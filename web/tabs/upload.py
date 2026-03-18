"""
模块 1: 数据输入
- 方式 A: 从 train / val / test 数据集中选择案例
- 方式 B: 上传本地超声视频文件 (mp4/avi/mov)
- 自动抽帧、预览、显示信息
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
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


def _get_test_root() -> Path:
    candidates = [
        PROJECT_ROOT / "test",
        Path("/data1/ouyangxinglong/EchoDVT/test"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return PROJECT_ROOT / "test"


def _list_cases(split: str = "val", test_subset: str = "normal") -> List[str]:
    if split == "test":
        subset = "patient" if test_subset == "patient" else "normal"
        root = _get_test_root() / subset
    else:
        root = _get_dataset_root() / split
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def _load_case_info(case_name: str, split: str = "val", test_subset: str = "normal") -> Dict:
    if split == "test":
        subset = "patient" if test_subset == "patient" else "normal"
        case_dir = _get_test_root() / subset / case_name
        split_name = f"test/{subset}"
    else:
        case_dir = _get_dataset_root() / split / case_name
        split_name = split

    images_dir = case_dir / "images"
    masks_dir = case_dir / "masks"

    frame_files = sorted(images_dir.glob("*.jpg"), key=lambda p: int(p.stem)) if images_dir.exists() else []
    if not frame_files:
        frame_files = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem)) if images_dir.exists() else []
    mask_files = sorted(masks_dir.glob("*.png"), key=lambda p: int(p.stem)) if masks_dir.exists() else []

    return {
        "case_name": case_name,
        "split": split_name,
        "case_dir": str(case_dir),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "num_frames": len(frame_files),
        "num_masks": len(mask_files),
        "frame_files": [str(f) for f in frame_files],
        "mask_files": [str(f) for f in mask_files],
        "mask_frame_indices": [int(f.stem) for f in mask_files],
    }


def _on_case_selected(case_name: str, split: str, test_subset: str, state: dict):
    if not case_name:
        return state, None, "请选择一个案例", []

    info = _load_case_info(case_name, split, test_subset)
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
    state["split"] = info["split"]
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
| **数据集** | `{info['split']}` |
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
            gallery.append((bgr_to_rgb(img), f"第 {i} 帧"))
    return gallery


def _get_dataset_counts() -> Dict[str, int]:
    return {
        "train": len(_list_cases("train")),
        "val": len(_list_cases("val")),
        "test_normal": len(_list_cases("test", "normal")),
        "test_patient": len(_list_cases("test", "patient")),
    }


def _build_dataset_selector_status_html(split: str, test_subset: str = "normal") -> str:
    counts = _get_dataset_counts()
    split_label = {
        "train": "📚 训练集",
        "val": "🎯 验证集",
        "test": "🧪 测试集",
    }.get(split, "🎯 验证集")

    split_desc = {
        "train": f"{counts['train']} 例，全部正常，适合查看标准压缩超声序列。",
        "val": f"{counts['val']} 例，正常与患者各半，适合快速验证完整链路。",
        "test": (
            f"{counts['test_normal'] + counts['test_patient']} 例，"
            f"当前 test 子集：`{test_subset}`，适合直接抽取真实测试案例。"
        ),
    }.get(split, "")

    return f"""
    <div class="dataset-selector-status">
        <div class="dataset-selector-pill">当前入口：{split_label}</div>
        <div class="dataset-selector-text">{split_desc}</div>
    </div>
    """


def _get_dataset_selector_updates(
    split: str,
    test_subset: str = "normal",
    selected_case: str | None = None,
):
    effective_subset = test_subset if split == "test" else "normal"
    cases = _list_cases(split, effective_subset)
    case_value = selected_case if selected_case in cases else (cases[0] if cases else None)
    subset_update = gr.update(visible=(split == "test"), value=effective_subset)
    return (
        gr.update(value=split),
        gr.update(choices=cases, value=case_value),
        subset_update,
        gr.update(variant="primary" if split == "train" else "secondary"),
        gr.update(variant="primary" if split == "val" else "secondary"),
        gr.update(variant="primary" if split == "test" else "secondary"),
        _build_dataset_selector_status_html(split, effective_subset),
    )


def _on_source_changed(split: str, test_subset: str):
    _, case_update, subset_update, train_btn_update, val_btn_update, test_btn_update, selector_status = (
        _get_dataset_selector_updates(split, test_subset)
    )
    return case_update, subset_update, train_btn_update, val_btn_update, test_btn_update, selector_status


def _select_train_source():
    return _get_dataset_selector_updates("train", "normal")


def _select_val_source():
    return _get_dataset_selector_updates("val", "normal")


def _select_test_source(test_subset: str):
    return _get_dataset_selector_updates("test", test_subset)


def _build_preview_placeholder(title: str, subtitle: str, width: int = 1200, height: int = 680) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    cv2.rectangle(canvas, (48, 52), (width - 48, height - 52), (226, 232, 240), 2)
    cv2.rectangle(canvas, (92, 96), (width - 92, height - 96), (219, 234, 254), 2)
    cv2.putText(canvas, title, (118, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (37, 99, 235), 3, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (118, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 116, 139), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Load a case or upload a local ultrasound video to begin.",
        (118, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (148, 163, 184),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_upload_tab(state: gr.State):
    """构建数据输入 Tab"""
    counts = _get_dataset_counts()
    dataset_info = (
        f"train: {counts['train']} 例 (全部正常) | "
        f"val: {counts['val']} 例 (38 正常 + 38 患者) | "
        f"test: {counts['test_normal'] + counts['test_patient']} 例 "
        f"({counts['test_normal']} normal + {counts['test_patient']} patient)"
    )
    selector_status_html = _build_dataset_selector_status_html("val", "normal")
    preview_placeholder = _build_preview_placeholder(
        "Preview panel",
        "Supports train / val / test cases and local ultrasound video upload",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            gr.HTML(_TAB_HEADER.format(
                bg1="#f0f9ff", bg2="#eff6ff",
                icon="📤", title="数据输入",
                desc="从数据集选择案例，或上传本地超声视频",
            ))

            with gr.Tabs() as input_tabs:
                with gr.Tab("从数据集选择", id="dataset"):
                    split_radio = gr.Radio(
                        choices=["train", "val", "test"],
                        value="val",
                        label="数据集",
                        info=dataset_info,
                        visible=False,
                    )

                    gr.HTML("""
                    <div class="dataset-entry-strip">
                        <div class="dataset-entry-headline">三组并行入口</div>
                        <div class="dataset-entry-subtitle">先选择案例池，再进入具体病例与后续分析流程。</div>
                    </div>
                    """)

                    with gr.Row(equal_height=False, elem_classes=["dataset-entry-row"]):
                        with gr.Column(scale=1, min_width=0):
                            train_source_btn = gr.Button(
                                "📚 训练集",
                                variant="secondary",
                                elem_classes=["dataset-entry-btn"],
                            )
                            gr.HTML(
                                f'<div class="dataset-entry-meta"><strong>{counts["train"]} 例</strong><span>全部正常，适合查看标准序列</span></div>'
                            )

                        with gr.Column(scale=1, min_width=0):
                            val_source_btn = gr.Button(
                                "🎯 验证集",
                                variant="primary",
                                elem_classes=["dataset-entry-btn"],
                            )
                            gr.HTML(
                                f'<div class="dataset-entry-meta"><strong>{counts["val"]} 例</strong><span>38 正常 + 38 患者，适合快速验证</span></div>'
                            )

                        with gr.Column(scale=1, min_width=0):
                            test_source_btn = gr.Button(
                                "🧪 测试集",
                                variant="secondary",
                                elem_classes=["dataset-entry-btn"],
                            )
                            gr.HTML(
                                f'<div class="dataset-entry-meta"><strong>{counts["test_normal"] + counts["test_patient"]} 例</strong><span>{counts["test_normal"]} normal + {counts["test_patient"]} patient</span></div>'
                            )

                    dataset_selector_status = gr.HTML(selector_status_html)

                    test_subset_radio = gr.Radio(
                        choices=["normal", "patient"],
                        value="normal",
                        label="test 子集",
                        visible=False,
                        info="切换 test 入口下的 normal / patient 案例池",
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

                with gr.Tab("上传本地视频", id="video_upload"):
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

            case_info = gr.Markdown("""
> 💡 **快速开始**: 先点击上方 `train / val / test` 入口卡，再选择具体案例；也可以切换到「上传本地视频」直接加载本地超声视频。
""")

        with gr.Column(scale=3):
            preview_image = gr.Image(
                label="首帧预览（含 GT 标注叠加）",
                height=420,
                type="numpy",
                value=preview_placeholder,
            )

            frame_gallery = gr.Gallery(
                label="帧序列预览（均匀采样）",
                columns=6, rows=None, height="auto",
                object_fit="contain",
            )

    train_source_btn.click(
        fn=_select_train_source,
        outputs=[
            split_radio,
            case_dropdown,
            test_subset_radio,
            train_source_btn,
            val_source_btn,
            test_source_btn,
            dataset_selector_status,
        ],
    )
    val_source_btn.click(
        fn=_select_val_source,
        outputs=[
            split_radio,
            case_dropdown,
            test_subset_radio,
            train_source_btn,
            val_source_btn,
            test_source_btn,
            dataset_selector_status,
        ],
    )
    test_source_btn.click(
        fn=_select_test_source,
        inputs=[test_subset_radio],
        outputs=[
            split_radio,
            case_dropdown,
            test_subset_radio,
            train_source_btn,
            val_source_btn,
            test_source_btn,
            dataset_selector_status,
        ],
    )

    split_radio.change(
        fn=_on_source_changed,
        inputs=[split_radio, test_subset_radio],
        outputs=[
            case_dropdown,
            test_subset_radio,
            train_source_btn,
            val_source_btn,
            test_source_btn,
            dataset_selector_status,
        ],
    )
    test_subset_radio.change(
        fn=_on_source_changed,
        inputs=[split_radio, test_subset_radio],
        outputs=[
            case_dropdown,
            test_subset_radio,
            train_source_btn,
            val_source_btn,
            test_source_btn,
            dataset_selector_status,
        ],
    )

    load_btn.click(
        fn=_on_case_selected,
        inputs=[case_dropdown, split_radio, test_subset_radio, state],
        outputs=[state, preview_image, case_info, frame_gallery],
    )

    case_dropdown.change(
        fn=_on_case_selected,
        inputs=[case_dropdown, split_radio, test_subset_radio, state],
        outputs=[state, preview_image, case_info, frame_gallery],
    )

    upload_btn.click(
        fn=_on_video_uploaded,
        inputs=[video_input, state],
        outputs=[state, preview_image, case_info, frame_gallery],
    )

    return {
        "split_radio": split_radio,
        "test_subset_radio": test_subset_radio,
        "case_dropdown": case_dropdown,
        "preview_image": preview_image,
        "case_info": case_info,
        "frame_gallery": frame_gallery,
        "train_source_btn": train_source_btn,
        "val_source_btn": val_source_btn,
        "test_source_btn": test_source_btn,
        "dataset_selector_status": dataset_selector_status,
    }
