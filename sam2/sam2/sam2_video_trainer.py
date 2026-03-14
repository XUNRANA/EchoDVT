"""
SAM2VideoTrainer - 支持训练的SAM2视频预测器

关键改动:
1. 继承SAM2VideoPredictor
2. 重写关键方法，移除@torch.inference_mode()装饰器
3. 添加训练模式支持 (允许梯度计算)
4. 复用所有自适应记忆模块
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames


class SAM2VideoTrainer(SAM2VideoPredictor):
    """
    支持训练的SAM2视频预测器

    与SAM2VideoPredictor的区别:
    - 关键方法不使用@torch.inference_mode()装饰器
    - 支持梯度计算和反向传播
    - 可以在训练模式下运行

    Usage:
        # 构建训练器
        trainer = SAM2VideoTrainer.from_predictor(predictor)

        # 或直接构建
        trainer = build_sam2_video_trainer(config, checkpoint)

        # 训练模式
        trainer.train()
        for frame_idx, loss in trainer.train_on_video(video_path, masks):
            loss.backward()
            optimizer.step()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_mode = False

    @classmethod
    def from_predictor(cls, predictor: SAM2VideoPredictor) -> "SAM2VideoTrainer":
        """从现有的SAM2VideoPredictor创建训练器，共享权重"""
        trainer = cls.__new__(cls)
        # 复制所有属性
        trainer.__dict__.update(predictor.__dict__)
        trainer._training_mode = False
        return trainer

    def set_training_mode(self, mode: bool = True):
        """设置训练模式"""
        self._training_mode = mode
        if mode:
            super().train()
        else:
            super().eval()

    def train(self, mode=True):
        """重写train方法以同时设置_training_mode"""
        super().train(mode)
        self._training_mode = mode
        return self

    def eval(self):
        """重写eval方法以同时设置_training_mode"""
        super().eval()
        self._training_mode = False
        return self

    # ========== 重写关键方法，移除inference_mode ==========

    def init_state_train(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """
        初始化训练状态 - 不使用@torch.inference_mode()

        与init_state相同，但支持梯度计算
        """
        compute_device = self.device
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )

        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device

        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device

        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        # Adaptive Memory State
        inference_state["use_adaptive_memory"] = self.use_adaptive_memory
        inference_state["use_separate_memory"] = self.use_separate_memory
        inference_state["prev_frame_outputs_per_obj"] = {}

        if self.use_separate_memory:
            from sam2.adaptive_memory import SeparateMemoryBank
            inference_state["separate_memory_banks"] = SeparateMemoryBank()
        else:
            inference_state["separate_memory_banks"] = None

        inference_state["quality_scores_per_obj"] = {}

        # Warm up backbone (这部分可以不计算梯度)
        with torch.no_grad():
            self._get_image_feature(inference_state, frame_idx=0, batch_size=1)

        return inference_state

    def add_new_points_or_box_train(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """
        添加点或框prompt - 不使用@torch.inference_mode()

        返回: (frame_idx, obj_ids, video_res_masks, low_res_masks)
        low_res_masks用于计算损失
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        if box is not None:
            if not clear_old_points:
                raise ValueError("cannot add box without clearing old points")
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)

        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            from sam2.utils.misc import concat_points
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None

        from sam2.utils.misc import concat_points
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked

        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]

        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

        # 关键: 使用训练版本的单帧推理
        current_out, pred_masks = self._run_single_frame_inference_train(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]

        # 返回low_res_masks用于计算损失
        return frame_idx, obj_ids, current_out.get("pred_masks"), current_out.get("low_res_masks")

    def _run_single_frame_inference_train(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        obj_id=None,
        prev_frame_output=None,
    ):
        """
        单帧推理 - 训练版本，不使用@torch.inference_mode()

        这是最关键的方法，需要支持梯度计算
        """
        # 获取图像特征 (可以缓存)
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # 准备记忆条件特征
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # 存储用于后续帧的记忆
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out.get("maskmem_features")
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)

        maskmem_pos_enc = current_out.get("maskmem_pos_enc")
        if maskmem_pos_enc is not None:
            maskmem_pos_enc = [x.to(storage_device, non_blocking=True) for x in maskmem_pos_enc]

        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)

        # 保存输出 (训练时保留梯度)
        # 注意: sam2_base.py中的low_res_masks存储在current_out["pred_masks"]中
        low_res_masks = current_out.get("pred_masks")  # 这才是真正的low_res_masks

        if self._training_mode:
            compact_current_out = {
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc,
                "pred_masks": pred_masks_gpu,
                "low_res_masks": low_res_masks,  # 用于计算损失
                "obj_ptr": current_out.get("obj_ptr"),
                "object_score_logits": current_out.get("object_score_logits"),
            }
        else:
            compact_current_out = {
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc,
                "pred_masks": pred_masks_gpu.to(storage_device, non_blocking=True),
                "low_res_masks": low_res_masks,
                "obj_ptr": current_out.get("obj_ptr"),
                "object_score_logits": current_out.get("object_score_logits"),
            }

        return compact_current_out, pred_masks_gpu

    def propagate_in_video_train(
        self,
        inference_state,
        gt_masks_dict: Dict[int, torch.Tensor],  # {frame_idx: gt_mask_tensor}
        loss_fn,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        detach_memory=True,  # 是否分离记忆的梯度 (减少显存)
        vein_weight=1.5,
        artery_weight=1.0,
    ):
        """
        视频传播 - 训练版本

        Args:
            inference_state: 推理状态
            gt_masks_dict: GT masks字典 {frame_idx: tensor[num_objs, H, W]}
            loss_fn: 损失函数
            start_frame_idx: 开始帧
            max_frame_num_to_track: 最大跟踪帧数
            reverse: 是否反向
            detach_memory: 是否分离记忆梯度 (减少显存但可能影响训练效果)
            vein_weight: 静脉损失权重
            artery_weight: 动脉损失权重

        Yields:
            (frame_idx, loss, pred_masks)
        """
        # 预处理
        self._propagate_in_video_preflight_train(
            inference_state,
            detach_memory=detach_memory,
        )

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # 设置帧范围
        if start_frame_idx is None:
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        prev_frame_outputs = [None] * batch_size

        for frame_idx in processing_order:
            pred_masks_per_obj = []
            low_res_masks_per_obj = []

            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)

                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    # 条件帧 (首帧)
                    current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"].to(inference_state["device"])
                    low_res_masks = current_out.get("low_res_masks")
                else:
                    # 非条件帧 - 需要传播
                    current_out, pred_masks = self._run_single_frame_inference_train(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                        obj_id=obj_id,
                        prev_frame_output=prev_frame_outputs[obj_idx],
                    )
                    low_res_masks = current_out.get("low_res_masks")

                    # 存储到output_dict
                    if detach_memory:
                        # 分离记忆的梯度以减少显存
                        storage_out = {
                            "maskmem_features": current_out["maskmem_features"].detach() if current_out.get("maskmem_features") is not None else None,
                            "maskmem_pos_enc": [x.detach() for x in current_out["maskmem_pos_enc"]] if current_out.get("maskmem_pos_enc") is not None else None,
                            "pred_masks": current_out["pred_masks"].detach(),
                            "obj_ptr": current_out["obj_ptr"].detach() if current_out.get("obj_ptr") is not None else None,
                        }
                    else:
                        storage_out = current_out

                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = storage_out
                    inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}

                    # 更新前一帧输出
                    prev_frame_outputs[obj_idx] = {
                        "pred_masks": pred_masks.detach() if detach_memory else pred_masks,
                        "frame_idx": frame_idx,
                    }

                pred_masks_per_obj.append(pred_masks)
                if low_res_masks is not None:
                    low_res_masks_per_obj.append(low_res_masks)

            # 计算损失 (如果该帧有GT)
            frame_loss = None
            if frame_idx in gt_masks_dict and low_res_masks_per_obj:
                gt_mask = gt_masks_dict[frame_idx]  # [num_objs, H, W] or [H, W]
                frame_loss = self._compute_frame_loss(
                    low_res_masks_per_obj,
                    gt_mask,
                    obj_ids,
                    loss_fn,
                    inference_state["device"],
                    vein_weight=vein_weight,
                    artery_weight=artery_weight,
                )

            # 合并预测
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0] if pred_masks_per_obj else None

            yield frame_idx, frame_loss, all_pred_masks

    def _propagate_in_video_preflight_train(self, inference_state, detach_memory=False):
        """预处理 - 训练版本"""
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError("No input points or masks provided")

        # 合并临时输出到output_dict
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]

            for is_cond in [True, False]:
                storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                temp_outputs = obj_temp_output_dict[storage_key]
                for frame_idx, out in temp_outputs.items():
                    # 编码记忆 (训练时保留梯度)
                    if out.get("maskmem_features") is None and out.get("pred_masks") is not None:
                        if detach_memory:
                            with torch.no_grad():
                                out = self._encode_frame_memory(
                                    inference_state, frame_idx, out
                                )
                        else:
                            out = self._encode_frame_memory(inference_state, frame_idx, out)
                    if detach_memory:
                        if out.get("maskmem_features") is not None:
                            out["maskmem_features"] = out["maskmem_features"].detach()
                        if out.get("maskmem_pos_enc") is not None:
                            out["maskmem_pos_enc"] = [
                                x.detach() for x in out["maskmem_pos_enc"]
                            ]
                        if out.get("obj_ptr") is not None:
                            out["obj_ptr"] = out["obj_ptr"].detach()
                    obj_output_dict[storage_key][frame_idx] = out

                temp_outputs.clear()

            # 检查是否有输入
            if (
                len(obj_output_dict["cond_frame_outputs"]) == 0 and
                len(obj_output_dict["non_cond_frame_outputs"]) == 0
            ):
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(f"No input for object {obj_id}")

            # 清理重叠帧
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    def _encode_frame_memory(self, inference_state, frame_idx, out):
        """编码帧记忆 - 训练版本"""
        device = inference_state["device"]

        # 获取图像特征
        (
            _,
            _,
            current_vision_feats,
            _,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size=1)

        pred_masks = out["pred_masks"]
        if pred_masks.shape[-2:] != (self.image_size, self.image_size):
            pred_masks = F.interpolate(
                pred_masks,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # 编码记忆
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=pred_masks,
            object_score_logits=out.get("object_score_logits"),
            is_mask_from_pts=True,
        )

        out["maskmem_features"] = maskmem_features
        out["maskmem_pos_enc"] = maskmem_pos_enc

        return out

    def _compute_frame_loss(
            self,
            low_res_masks_list,
            gt_mask,
            obj_ids,
            loss_fn,
            device,
            vein_weight=1.5,
            artery_weight=1.0,
    ):
        """
        计算单帧损失

        Args:
            low_res_masks_list: 预测masks列表，每个是 [1, 1, H, W]
            gt_mask: GT mask, [H, W] (类别图) 或 [num_objs, H, W]
            obj_ids: 对象ID列表
            loss_fn: 损失函数
            device: 设备
            vein_weight: 静脉权重
            artery_weight: 动脉权重

        Returns:
            total_loss: 加权总损失
        """
        total_loss = torch.zeros((), device=device)

        for i, (obj_id, low_res_mask) in enumerate(zip(obj_ids, low_res_masks_list)):
            if low_res_mask is None:
                continue

            # 获取该对象的GT
            if gt_mask.dim() == 2:
                # gt_mask是类别图 [H, W]
                gt_obj = (gt_mask == obj_id).float()
            elif gt_mask.dim() == 3:
                # gt_mask是多通道 [num_objs, H, W]
                gt_obj = gt_mask[i].float()
            else:
                # gt_mask是 [1, 1, H, W] 或其他
                if gt_mask.shape[0] == 1 and gt_mask.shape[1] == 1:
                    gt_obj = (gt_mask.squeeze(0).squeeze(0) == obj_id).float()
                else:
                    gt_obj = gt_mask[i].float()

            # 确保 gt_obj 是 4 维: [1, 1, H, W]
            if gt_obj.dim() == 2:
                # [H, W] -> [1, 1, H, W]
                gt_obj = gt_obj.unsqueeze(0).unsqueeze(0)
            elif gt_obj.dim() == 3:
                # [B, H, W] -> [B, 1, H, W]
                gt_obj = gt_obj.unsqueeze(1)
            # else: 已经是 4 维

            # Resize GT to match prediction size
            h_out, w_out = low_res_mask.shape[-2:]
            gt_obj_resized = F.interpolate(
                gt_obj,
                size=(h_out, w_out),
                mode="nearest"
            )  # 保持 [1, 1, h_out, w_out]

            # 🔥 关键: 不使用 squeeze()，直接保持 4 维
            # 确保形状完全匹配
            if gt_obj_resized.shape != low_res_mask.shape:
                # 如果 batch 维度不同
                if gt_obj_resized.shape[0] != low_res_mask.shape[0]:
                    gt_obj_resized = gt_obj_resized.expand(low_res_mask.shape[0], -1, -1, -1)
                # 如果 channel 维度不同
                if gt_obj_resized.shape[1] != low_res_mask.shape[1]:
                    gt_obj_resized = gt_obj_resized.expand(-1, low_res_mask.shape[1], -1, -1)

            gt_obj_resized = gt_obj_resized.to(device)

            # 调试信息 (可选，训练稳定后删除)
            # print(f"obj_id={obj_id}: pred={low_res_mask.shape}, gt={gt_obj_resized.shape}")

            # 计算损失
            loss = loss_fn(low_res_mask, gt_obj_resized)

            # 类别权重
            weight = vein_weight if obj_id == 2 else artery_weight
            total_loss = total_loss + loss * weight

        return total_loss



def fill_holes_in_mask_scores(mask, area_thresh):
    """填充mask中的小孔"""
    assert mask.dim() == 4 and mask.shape[1] == 1
    mask = mask[:, 0, :, :]  # [B, H, W]
    filled = mask.clone()

    for i in range(mask.shape[0]):
        m = mask[i].detach().cpu().numpy()
        filled_m = m.copy()

        # 简单的形态学填充
        import cv2
        binary = (m > 0).astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 只填充小于阈值的孔
        diff = closed - binary
        if diff.sum() <= area_thresh:
            filled_m = m * (closed > 0)

        filled[i] = torch.from_numpy(filled_m).to(mask.device)

    return filled.unsqueeze(1)  # [B, 1, H, W]


# ========== 构建函数 ==========

def build_sam2_video_trainer(
    config_file,
    ckpt_path=None,
    device="cuda",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    use_adaptive_memory=False,
    use_separate_memory=False,
    memory_quality_config=None,
    use_av_constraint=False,
):
    """
    构建SAM2VideoTrainer (支持训练的视频预测器)

    参数与build_sam2_video_predictor相同
    """
    from sam2.build_sam import build_sam2_video_predictor

    # 首先构建predictor (使用eval模式避免Hydra问题)
    predictor = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
        mode="eval",  # 先用eval模式构建，之后切换为train
        hydra_overrides_extra=hydra_overrides_extra,
        apply_postprocessing=apply_postprocessing,
        use_adaptive_memory=use_adaptive_memory,
        use_separate_memory=use_separate_memory,
        memory_quality_config=memory_quality_config,
        use_av_constraint=use_av_constraint,
    )

    # 转换为trainer并设置为训练模式
    trainer = SAM2VideoTrainer.from_predictor(predictor)
    trainer.train()  # 切换为训练模式
    return trainer
