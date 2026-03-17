#!/usr/bin/env python3
"""
EchoDVT 超声诊断系统 — Gradio Web 应用入口 (Dashboard 版)

布局: 使用 CSS Grid 实现侧边栏 + Gradio Tabs 内容区
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sam2"))
sys.path.insert(0, str(PROJECT_ROOT / "yolo"))

import gradio as gr

from tabs.dashboard import (
    build_dashboard_panel,
    _refresh_dashboard,
    _quick_load_next_val_case,
    _quick_run_pipeline_from_dashboard,
)
from tabs.upload import build_upload_tab
from tabs.detection import build_detection_tab
from tabs.segmentation import build_segmentation_tab
from tabs.diagnosis import build_diagnosis_tab
from tabs.evaluation import build_evaluation_tab
from tabs.pipeline import build_pipeline_tab


CSS_PATH = Path(__file__).parent / "assets" / "custom.css"


def build_app():
    """构建 Dashboard 风格的 Gradio 应用"""
    custom_css = ""
    if CSS_PATH.exists():
        custom_css = CSS_PATH.read_text(encoding="utf-8")

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    # 北京时间时钟脚本 — 通过 head 参数注入，确保执行
    clock_head = """
    <script>
    document.addEventListener('DOMContentLoaded', function(){
        function tick(){
            var d = new Date();
            var utc = d.getTime() + d.getTimezoneOffset()*60000;
            var bj = new Date(utc + 8*3600000);
            var ds = bj.getFullYear()+'-'+String(bj.getMonth()+1).padStart(2,'0')+'-'+String(bj.getDate()).padStart(2,'0');
            var ts = String(bj.getHours()).padStart(2,'0')+':'+String(bj.getMinutes()).padStart(2,'0')+':'+String(bj.getSeconds()).padStart(2,'0');
            var el = document.getElementById('beijing-clock');
            if(el) el.innerHTML='<span class="clock-date">'+ds+'</span> '+ts;
        }
        var waitForClock = setInterval(function(){
            if(document.getElementById('beijing-clock')){
                clearInterval(waitForClock);
                tick();
                setInterval(tick, 1000);
            }
        }, 200);
    });
    </script>
    """

    with gr.Blocks(title="EchoDVT 智能诊断系统") as app:

        # ===== 顶部栏 =====
        gr.HTML("""
        <div class="topbar">
            <div class="topbar-left">
                <span class="topbar-logo">🫀</span>
                <h1 class="topbar-title">EchoDVT — 深静脉血栓智能诊断系统</h1>
            </div>
            <div class="topbar-right">
                <div class="topbar-clock" id="beijing-clock">
                    <span class="clock-date">加载中...</span> --:--:--
                </div>
                <div class="topbar-badge" title="通知">🔔<span class="notification-dot"></span></div>
            </div>
        </div>
        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
             style="display:none"
             onload="
                function bjTick(){
                    var d=new Date(),u=d.getTime()+d.getTimezoneOffset()*60000,
                        b=new Date(u+8*3600000),
                        ds=b.getFullYear()+'-'+String(b.getMonth()+1).padStart(2,'0')+'-'+String(b.getDate()).padStart(2,'0'),
                        ts=String(b.getHours()).padStart(2,'0')+':'+String(b.getMinutes()).padStart(2,'0')+':'+String(b.getSeconds()).padStart(2,'0'),
                        el=document.getElementById('beijing-clock');
                    if(el) el.innerHTML='<span class=clock-date>'+ds+'</span> '+ts;
                }
                bjTick(); setInterval(bjTick,1000);
             ">
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

        # ===== 使用 Gradio 原生 Tabs，用 CSS 改造成侧边栏外观 =====
        with gr.Tabs(elem_classes=["sidebar-tabs"]) as tabs:
            with gr.Tab("📊 仪表盘", id="dashboard", elem_classes=["sidebar-tab-item"]):
                dash_outs = build_dashboard_panel(state)

            # 主流程顺序
            with gr.Tab("📤 数据输入", id="upload", elem_classes=["sidebar-tab-item"]):
                upload_handles = build_upload_tab(state)

            with gr.Tab("🚀 一键分析", id="pipeline", elem_classes=["sidebar-tab-item"]):
                pipeline_handles = build_pipeline_tab(state)

            with gr.Tab("🎯 目标检测", id="detection", elem_classes=["sidebar-tab-item"]):
                build_detection_tab(state)

            with gr.Tab("🔬 视频分割", id="segmentation", elem_classes=["sidebar-tab-item"]):
                build_segmentation_tab(state)

            with gr.Tab("🩺 DVT 诊断", id="diagnosis", elem_classes=["sidebar-tab-item"]):
                build_diagnosis_tab(state)

            with gr.Tab("📈 定量评估", id="evaluation", elem_classes=["sidebar-tab-item"]):
                build_evaluation_tab(state)

        # Dashboard 自动刷新
        (
            dash_status,
            dash_dataset,
            dash_errors,
            dash_chart,
            dash_workflow,
            dash_refresh,
            dash_quick_load_btn,
            dash_quick_analyze_btn,
            dash_quick_status,
        ) = dash_outs
        app.load(
            fn=_refresh_dashboard,
            inputs=[state],
            outputs=[dash_status, dash_dataset, dash_errors, dash_chart, dash_workflow],
        )

        dash_quick_load_btn.click(
            fn=_quick_load_next_val_case,
            inputs=[state],
            outputs=[
                state,
                upload_handles["split_radio"],
                upload_handles["case_dropdown"],
                upload_handles["preview_image"],
                upload_handles["case_info"],
                upload_handles["frame_gallery"],
                dash_quick_status,
                dash_workflow,
                tabs,
                upload_handles["test_subset_radio"],
            ],
        )

        dash_quick_analyze_btn.click(
            fn=_quick_run_pipeline_from_dashboard,
            inputs=[state],
            outputs=[
                state,
                pipeline_handles["det_preview"],
                pipeline_handles["seg_gallery"],
                pipeline_handles["area_plot"],
                pipeline_handles["report_html"],
                pipeline_handles["diagnosis_summary"],
                dash_quick_status,
                dash_workflow,
                tabs,
            ],
        )

    return app, custom_css, theme, clock_head


def main():
    parser = argparse.ArgumentParser(description="EchoDVT Dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="指定端口（默认不固定端口，由 Gradio 自动选择可用端口）",
    )
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app, css, theme, head = build_app()
    launch_kwargs = dict(
        server_name=args.server_name,
        share=args.share,
        show_error=True,
        css=css,
        theme=theme,
        head=head,
        max_file_size="2gb",
        allowed_paths=[
            str(PROJECT_ROOT),
            "/tmp",
            "/data1/ouyangxinglong",
        ],
    )
    if args.port is not None:
        launch_kwargs["server_port"] = args.port

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
