"""
模块 1: 视频上传与预览
- 从 val 集中选择案例
- 上传自定义超声视频文件夹
- 显示首帧预览、帧数、标注帧信息
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.visualization import bgr_to_rgb


def _get_dataset_root() -> Path:
    """获取数据集根目录（兼容多种部署路径）"""
    candidates = [
        PROJECT_ROOT / "sam2" / "dataset",
        PROJECT_ROOT / "dataset",
        Path("/data1/ouyangxinglong/EchoDVT/sam2/dataset"),
        Path("/data1/ouyangxinglong/EchoDVT/dataset"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return PROJECT_ROOT / "dataset"


def _list_cases(split: str = "val") -> List[str]:
    """列出指定 split 下的所有 case"""
    root = _get_dataset_root() / split
    if not root.exists():
        return []
    cases = sorted([d.name for d in root.iterdir() if d.is_dir()])
    return cases


def _load_case_info(case_name: str, split: str = "val") -> Dict:
    """加载一个 case 的基本信息"""
    case_dir = _get_dataset_root() / split / case_name
    images_dir = case_dir / "images"
    masks_dir = case_dir / "masks"

    frame_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem)) if images_dir.exists() else []
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem)) if masks_dir.exists() else []

    info = {
        "case_name": case_name,
        "split": split,
        "case_dir": str(case_dir),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "num_frames": len(frame_files),
        "num_masks": len(mask_files),
        "frame_files": [str(f) for f in frame_files],
        "mask_files": [str(f) for f in mask_files],
        "mask_frame_indices": [int(f.stem) for f in mask_files],
    }
    return info


def _on_case_selected(case_name: str, split: str, state: dict):
    """当用户选择一个 case 时触发"""
    if not case_name:
        return state, None, "请选择一个案例"

    info = _load_case_info(case_name, split)
    if info["num_frames"] == 0:
        return state, None, f"⚠️ 案例 {case_name} 没有找到图像帧"

    # 加载首帧预览
    first_frame = cv2.imread(info["frame_files"][0])
    first_frame_rgb = bgr_to_rgb(first_frame)
    h, w = first_frame.shape[:2]

    # 加载首帧 GT mask（如果有的话）
    gt_preview = None
    first_mask_path = Path(info["masks_dir"]) / "00000.png"
    if first_mask_path.exists():
        gt_mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
        from web.utils.visualization import overlay_masks
        gt_preview = bgr_to_rgb(overlay_masks(first_frame, gt_mask, alpha=0.5))

    # 更新 state
    state["current_case"] = case_name
    state["split"] = split
    state["images_dir"] = info["images_dir"]
    state["masks_dir"] = info["masks_dir"]
    state["frame_files"] = info["frame_files"]
    state["mask_files"] = info["mask_files"]
    state["detections"] = None
    state["pred_masks"] = None
    state["frame_metrics"] = []
    state["vein_areas"] = []
    state["artery_areas"] = []

    # 生成信息文本
    info_md = f"""
### 📋 案例信息

| 属性 | 值 |
|------|------|
| **案例名** | `{case_name}` |
| **数据集** | `{split}` |
| **总帧数** | {info['num_frames']} |
| **标注帧数** | {info['num_masks']} |
| **分辨率** | {w} × {h} |
| **标注帧** | {', '.join(str(i) for i in info['mask_frame_indices'][:10])}{'...' if len(info['mask_frame_indices']) > 10 else ''} |
"""

    # 返回: 首帧 + GT 预览（如果有）
    preview = gt_preview if gt_preview is not None else first_frame_rgb

    return state, preview, info_md


def _on_split_changed(split: str):
    """切换数据集 split 时更新案例列表"""
    cases = _list_cases(split)
    if not cases:
        return gr.update(choices=[], value=None)
    return gr.update(choices=cases, value=cases[0])


def build_upload_tab(state: gr.State):
    """构建视频上传与预览 Tab"""

    gr.Markdown("""
    ### 选择超声视频案例
    从验证集或训练集中选择一个案例进行分析。每个案例包含完整的超声压缩视频序列。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            split_radio = gr.Radio(
                choices=["val", "train"],
                value="val",
                label="📂 数据集",
                info="val 含 76 例 (38正常+38患者), train 含 300 例 (全部正常)",
            )

            initial_cases = _list_cases("val")
            case_dropdown = gr.Dropdown(
                choices=initial_cases,
                value=initial_cases[0] if initial_cases else None,
                label="🔍 选择案例",
                info="从数据集中选择一个超声视频案例",
                filterable=True,
            )

            load_btn = gr.Button("📥 加载案例", variant="primary", size="lg")

            case_info = gr.Markdown("选择案例后将显示详细信息")

        with gr.Column(scale=2):
            preview_image = gr.Image(
                label="首帧预览（含 GT 标注叠加）",
                height=500,
                type="numpy",
            )

    # 事件绑定
    split_radio.change(
        fn=_on_split_changed,
        inputs=[split_radio],
        outputs=[case_dropdown],
    )

    load_btn.click(
        fn=_on_case_selected,
        inputs=[case_dropdown, split_radio, state],
        outputs=[state, preview_image, case_info],
    )

    # 也支持 dropdown 变化直接加载
    case_dropdown.change(
        fn=_on_case_selected,
        inputs=[case_dropdown, split_radio, state],
        outputs=[state, preview_image, case_info],
    )
