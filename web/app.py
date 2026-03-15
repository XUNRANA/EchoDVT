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
from tabs.pipeline import build_pipeline_tab


CSS_PATH = Path(__file__).parent / "assets" / "custom.css"


def build_app():
    """构建完整的 Gradio 应用"""
    custom_css = ""
    if CSS_PATH.exists():
        custom_css = CSS_PATH.read_text(encoding="utf-8")

    # ★ Gradio 6.x: theme 和 css 在 launch() 中传入
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="EchoDVT 超声诊断系统") as app:

        # ===== 顶部标题 =====
        gr.HTML("""
        <div class="app-header">
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="font-size:42px; line-height:1;">🫀</div>
                <div>
                    <h1 style="margin:0; font-size:26px; font-weight:800; color:#fff;
                               letter-spacing:-0.5px;">
                        EchoDVT — 超声深静脉血栓智能诊断系统
                    </h1>
                    <p style="margin:4px 0 0 0; font-size:13px; color:rgba(255,255,255,0.75);">
                        YOLO 血管检测 &rarr; SAM2 LoRA 视频分割 &rarr; 19 维特征分析 &rarr; DVT 智能判断
                    </p>
                </div>
            </div>
        </div>
        """)

        # ===== 全局状态 =====
        state = gr.State({
            "current_case": None,
            "images_dir": None,
            "masks_dir": None,
            "frame_files": [],
            "mask_files": [],
            "from_video": False,
            "detections": None,
            "pred_masks": None,
            "frame_metrics": [],
            "vein_areas": [],
            "artery_areas": [],
        })

        # ===== Tab 布局 =====
        with gr.Tabs() as tabs:
            with gr.Tab("📤 数据输入", id="upload"):
                build_upload_tab(state)

            with gr.Tab("🚀 一键分析", id="pipeline"):
                build_pipeline_tab(state)

            with gr.Tab("🎯 YOLO 检测", id="detection"):
                build_detection_tab(state)

            with gr.Tab("🔬 SAM2 分割", id="segmentation"):
                build_segmentation_tab(state)

            with gr.Tab("🩺 诊断报告", id="diagnosis"):
                build_diagnosis_tab(state)

            with gr.Tab("📊 定量评估", id="evaluation"):
                build_evaluation_tab(state)

            with gr.Tab("⚖️ 模型对比", id="comparison"):
                build_comparison_tab(state)

        # ===== 底部 =====
        gr.HTML("""
        <div style="text-align:center; padding:16px; color:#64748b; font-size:12px; margin-top:12px;
                    border-top:1px solid #1e293b;">
            EchoDVT v2.0 &mdash; YOLO + SAM2 LoRA + MFP + 19-Feature Classification
        </div>
        """)

    return app, custom_css, theme


def main():
    parser = argparse.ArgumentParser(description="EchoDVT Web 诊断系统")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app, css, theme = build_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=css,
        theme=theme,
        max_file_size="2gb",      # ★ 允许上传最大 2GB 的视频文件
        allowed_paths=[           # ★ 允许 Gradio 从这些路径提供文件
            str(PROJECT_ROOT),
            "/tmp",
            "/data1/ouyangxinglong",
        ],
    )


if __name__ == "__main__":
    main()
