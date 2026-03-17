"""
模块 1: 数据输入
- 方式 A: 从 val/train 集中选择案例
- 方式 B: 上传本地超声视频文件 (mp4/avi/mov)
- 自动抽帧、预览、显示信息
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import Dict, List

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from web.utils.visualization import bgr_to_rgb


# ─── 统一 Tab 标题 HTML 模板 ───

_TAB_HEADER = """
<div style="padding:16px 20px; background:linear-gradient(135deg, {bg1}, {bg2});
            border-radius:12px; border:1px solid #e2e8f0; margin-bottom:8px;">
    <h3 style="margin:0 0 4px 0; color:#1e293b; font-size:16px;">
        {icon} {title}
    </h3>
    <p style="margin:0; color:#64748b; font-size:13px;">
        {desc}
    </p>
</div>
"""


def _get_dataset_root() -> Path:
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
    root = _get_dataset_root() / split
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def _load_case_info(case_name: str, split: str = "val") -> Dict:
    case_dir = _get_dataset_root() / split / case_name
    images_dir = case_dir / "images"
    masks_dir = case_dir / "masks"

    frame_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem)) if images_dir.exists() else []
    if not frame_files:
        frame_files = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem)) if images_dir.exists() else []
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem)) if masks_dir.exists() else []

    return {
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


def _on_case_selected(case_name: str, split: str, state: dict):
    if not case_name:
        return state, None, "请选择一个案例", []

    info = _load_case_info(case_name, split)
    if info["num_frames"] == 0:
        return state, None, "案例中没有找到图像帧", []

    first_frame = cv2.imread(info["frame_files"][0])
    first_frame_rgb = bgr_to_rgb(first_frame)
    h, w = first_frame.shape[:2]

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
    state["from_video"] = False
    state["detections"] = None
    state["pred_masks"] = None
    state["frame_metrics"] = []
    state["vein_areas"] = []
    state["artery_areas"] = []

    info_md = f"""### 📋 案例信息
| 属性 | 值 |
|------|------|
| **案例名** | `{case_name}` |
| **数据集** | `{split}` |
| **总帧数** | {info['num_frames']} |
| **标注帧数** | {info['num_masks']} |
| **分辨率** | {w} × {h} |
| **标注帧** | {', '.join(str(i) for i in info['mask_frame_indices'][:10])}{'...' if len(info['mask_frame_indices']) > 10 else ''} |
"""

    preview = gt_preview if gt_preview is not None else first_frame_rgb

    # 生成帧缩略图 gallery
    gallery = _build_frame_gallery(info["frame_files"], max_frames=12)

    return state, preview, info_md, gallery


def _check_disk_space(path: str, min_gb: float = 1.0) -> bool:
    """检查磁盘空间是否足够"""
    try:
        import shutil as shu
        usage = shu.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        return free_gb >= min_gb
    except Exception:
        return True  # 无法检查时不阻塞


def _on_video_uploaded(video_path: str, state: dict):
    """处理本地上传的视频文件"""
    if not video_path:
        return state, None, "⚠️ 请上传一个视频文件", []

    video_path = Path(video_path)
    if not video_path.exists():
        return state, None, f"❌ 视频文件不存在: `{video_path}`", []

    # ★ 检查文件大小
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 2048:
        return state, None, f"❌ 视频文件过大 ({file_size_mb:.0f} MB)，最大支持 2GB", []

    # ★ 检查磁盘空间
    tmp_base = tempfile.gettempdir()
    if not _check_disk_space(tmp_base, min_gb=1.0):
        return state, None, "❌ 临时目录磁盘空间不足（< 1GB），请清理 /tmp 后重试", []

    # 创建临时目录存放抽帧
    tmp_dir = Path(tempfile.mkdtemp(prefix="echodvt_"))
    images_dir = tmp_dir / "images"
    images_dir.mkdir()

    # 抽帧
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return state, None, f"❌ 无法打开视频文件，请检查格式。\n\n支持格式: MP4 / AVI / MOV / MKV", []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            cap.release()
            return state, None, "❌ 无法读取视频帧数，文件可能已损坏", []

        frame_files = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = images_dir / f"{idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            frame_files.append(str(out_path))
            idx += 1
        cap.release()

    except Exception as e:
        return state, None, f"❌ 视频解析失败:\n```\n{traceback.format_exc()}\n```", []

    if not frame_files:
        return state, None, "❌ 视频中没有提取到有效帧", []

    # 加载首帧预览
    first_frame = cv2.imread(frame_files[0])
    first_frame_rgb = bgr_to_rgb(first_frame)

    # 更新 state
    case_name = video_path.stem
    state["current_case"] = case_name
    state["split"] = "upload"
    state["images_dir"] = str(images_dir)
    state["masks_dir"] = str(tmp_dir / "masks")  # 无 GT
    state["frame_files"] = frame_files
    state["mask_files"] = []
    state["from_video"] = True
    state["detections"] = None
    state["pred_masks"] = None
    state["frame_metrics"] = []
    state["vein_areas"] = []
    state["artery_areas"] = []

    duration = total_frames / fps if fps > 0 else 0
    info_md = f"""### 📋 视频信息
| 属性 | 值 |
|------|------|
| **文件名** | `{video_path.name}` |
| **文件大小** | {file_size_mb:.1f} MB |
| **来源** | 本地上传 |
| **总帧数** | {len(frame_files)} |
| **帧率** | {fps:.1f} FPS |
| **时长** | {duration:.1f} 秒 |
| **分辨率** | {w} × {h} |
| **标注** | 无（上传视频无 GT） |
"""

    gallery = _build_frame_gallery(frame_files, max_frames=12)
    return state, first_frame_rgb, info_md, gallery


def _build_frame_gallery(frame_files: list, max_frames: int = 12) -> list:
    """构建帧缩略图 gallery"""
    if not frame_files:
        return []
    n = len(frame_files)
    step = max(1, n // max_frames)
    gallery = []
    for i in range(0, n, step):
        if len(gallery) >= max_frames:
            break
        img = cv2.imread(frame_files[i])
        if img is not None:
            gallery.append((bgr_to_rgb(img), f"Frame {i}"))
    return gallery


def _on_split_changed(split: str):
    cases = _list_cases(split)
    if not cases:
        return gr.update(choices=[], value=None)
    return gr.update(choices=cases, value=cases[0])


def build_upload_tab(state: gr.State):
    """构建数据输入 Tab"""

    with gr.Row(equal_height=False):
        # ========== 左栏：输入区 ==========
        with gr.Column(scale=2):
            gr.HTML(_TAB_HEADER.format(
                bg1="#f0f9ff", bg2="#eff6ff",
                icon="📤", title="数据输入",
                desc="从数据集选择案例，或上传本地超声视频",
            ))

            with gr.Tabs() as input_tabs:
                # ---- 方式 A: 数据集 ----
                with gr.Tab("从数据集选择", id="dataset"):
                    split_radio = gr.Radio(
                        choices=["val", "train"],
                        value="val",
                        label="数据集",
                        info="val: 76 例 (38 正常 + 38 患者) | train: 300 例 (全部正常)",
                    )

                    initial_cases = _list_cases("val")
                    case_dropdown = gr.Dropdown(
                        choices=initial_cases,
                        value=initial_cases[0] if initial_cases else None,
                        label="选择案例",
                        filterable=True,
                    )

                    load_btn = gr.Button(
                        "📂 加载案例",
                        variant="primary", size="lg",
                    )

                # ---- 方式 B: 上传视频 ----
                with gr.Tab("上传本地视频", id="video_upload"):
                    # ★ 使用 gr.File 替代 gr.Video，避免 "Video not playable" 错误
                    video_input = gr.File(
                        label="上传超声视频文件",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
                        type="filepath",
                    )

                    upload_btn = gr.Button(
                        "🎬 解析视频并加载",
                        variant="primary", size="lg",
                    )

                    gr.HTML("""
                    <div style="padding:12px; background:rgba(241,245,249,0.8); border-radius:8px;
                                border:1px solid #e2e8f0; margin-top:4px;">
                        <p style="color:#64748b; font-size:12px; margin:0; line-height:1.7;">
                            📁 支持格式：MP4 / AVI / MOV / MKV（最大 2GB）<br>
                            ⚙️ 系统将自动逐帧提取并加载到分析流程
                        </p>
                    </div>
                    """)

            # 案例信息
            case_info = gr.Markdown("""
> 💡 **快速开始**: 从左侧数据集中选择一个案例，或切换到"上传本地视频"标签页上传超声视频文件。
""")

        # ========== 右栏：预览区 ==========
        with gr.Column(scale=3):
            preview_image = gr.Image(
                label="首帧预览（含 GT 标注叠加）",
                height=420,
                type="numpy",
            )

            frame_gallery = gr.Gallery(
                label="帧序列预览（均匀采样）",
                columns=6, rows=2, height=200,
                object_fit="contain",
            )

    # ========== 事件绑定 ==========
    split_radio.change(fn=_on_split_changed, inputs=[split_radio], outputs=[case_dropdown])

    load_btn.click(
        fn=_on_case_selected,
        inputs=[case_dropdown, split_radio, state],
        outputs=[state, preview_image, case_info, frame_gallery],
    )

    case_dropdown.change(
        fn=_on_case_selected,
        inputs=[case_dropdown, split_radio, state],
        outputs=[state, preview_image, case_info, frame_gallery],
    )

    upload_btn.click(
        fn=_on_video_uploaded,
        inputs=[video_input, state],
        outputs=[state, preview_image, case_info, frame_gallery],
    )

