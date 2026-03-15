#!/usr/bin/env python3
"""
EchoDVT 超声诊断系统 — Gradio Web 应用入口

用法:
    cd EchoDVT/web
    python app.py

    # 或指定端口和共享
    python app.py --port 7860 --share
"""

import sys
import argparse
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sam2"))
sys.path.insert(0, str(PROJECT_ROOT / "yolo"))

import gradio as gr

from tabs.upload import build_upload_tab
from tabs.detection import build_detection_tab
from tabs.segmentation import build_segmentation_tab
from tabs.diagnosis import build_diagnosis_tab
from tabs.evaluation import build_evaluation_tab
from tabs.comparison import build_comparison_tab


CSS_PATH = Path(__file__).parent / "assets" / "custom.css"


def build_app() -> gr.Blocks:
    """构建完整的 Gradio 应用"""
    custom_css = ""
    if CSS_PATH.exists():
        custom_css = CSS_PATH.read_text(encoding="utf-8")

    app_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="EchoDVT 超声诊断系统") as app:
        # Store theme and css for launch
        app._custom_css = custom_css
        app._custom_theme = app_theme

        # ===== 顶部标题 =====
        gr.HTML("""
        <div class="app-header">
            <h1>🫀 EchoDVT — 超声深静脉血栓智能诊断系统</h1>
            <p>基于 YOLO 检测 + SAM2 视频分割 + 时序特征分析的自动化 DVT 诊断平台</p>
        </div>
        """)

        # ===== 全局状态 =====
        state = gr.State({
            "current_case": None,
            "images_dir": None,
            "masks_dir": None,
            "frame_files": [],
            "mask_files": [],
            "detections": None,
            "pred_masks": None,
            "frame_metrics": [],
            "vein_areas": [],
            "artery_areas": [],
        })

        # ===== 6 个功能 Tab =====
        with gr.Tabs() as tabs:
            with gr.Tab("📤 视频上传", id="upload"):
                build_upload_tab(state)

            with gr.Tab("🎯 YOLO 检测", id="detection"):
                build_detection_tab(state)

            with gr.Tab("🔬 SAM2 分割", id="segmentation"):
                build_segmentation_tab(state)

            with gr.Tab("🩺 DVT 诊断", id="diagnosis"):
                build_diagnosis_tab(state)

            with gr.Tab("📊 定量评估", id="evaluation"):
                build_evaluation_tab(state)

            with gr.Tab("⚖️ 模型对比", id="comparison"):
                build_comparison_tab(state)

    return app


def main():
    parser = argparse.ArgumentParser(description="EchoDVT Web 诊断系统")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app = build_app()
    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
    }
    # Gradio 6.x: pass theme/css to launch
    if hasattr(app, '_custom_css'):
        launch_kwargs["css"] = app._custom_css
    if hasattr(app, '_custom_theme'):
        launch_kwargs["theme"] = app._custom_theme
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
