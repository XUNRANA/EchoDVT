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

from tabs.dashboard import build_dashboard_panel, _refresh_dashboard
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

        # ===== 顶部流程导航 =====
        gr.HTML("""
        <div class="workflow-nav-header">
            <div class="workflow-nav-title">🧭 流程导航</div>
            <div class="workflow-nav-desc">
                推荐顺序：数据输入 → 目标检测 → 视频分割 → DVT 诊断 → 定量评估（可随时使用「一键分析」快速完成全流程）
            </div>
        </div>
        """)
        with gr.Row(elem_classes=["workflow-nav-row"]):
            nav_upload_btn = gr.Button("1️⃣ 📤 数据输入", variant="secondary", size="sm")
            nav_detection_btn = gr.Button("2️⃣ 🎯 目标检测", variant="secondary", size="sm")
            nav_seg_btn = gr.Button("3️⃣ 🔬 视频分割", variant="secondary", size="sm")
            nav_diag_btn = gr.Button("4️⃣ 🩺 DVT 诊断", variant="secondary", size="sm")
            nav_eval_btn = gr.Button("5️⃣ 📈 定量评估", variant="secondary", size="sm")
            nav_pipeline_btn = gr.Button("🚀 一键分析", variant="secondary", size="sm")
            nav_dash_btn = gr.Button("📊 仪表盘", variant="secondary", size="sm")

        # ===== 使用 Gradio 原生 Tabs，用 CSS 改造成侧边栏外观 =====
        with gr.Tabs(elem_classes=["sidebar-tabs"]) as tabs:
            # 主流程顺序（按视频分析的正常经历顺序）
            with gr.Tab("📤 数据输入", id="upload", elem_classes=["sidebar-tab-item"]):
                upload_handles = build_upload_tab(state)

            with gr.Tab("🎯 目标检测", id="detection", elem_classes=["sidebar-tab-item"]):
                build_detection_tab(state)

            with gr.Tab("🔬 视频分割", id="segmentation", elem_classes=["sidebar-tab-item"]):
                build_segmentation_tab(state)

            with gr.Tab("🩺 DVT 诊断", id="diagnosis", elem_classes=["sidebar-tab-item"]):
                build_diagnosis_tab(state)

            with gr.Tab("📈 定量评估", id="evaluation", elem_classes=["sidebar-tab-item"]):
                build_evaluation_tab(state)

            with gr.Tab("🚀 一键分析", id="pipeline", elem_classes=["sidebar-tab-item"]):
                pipeline_handles = build_pipeline_tab(state)

            with gr.Tab("📊 仪表盘", id="dashboard", elem_classes=["sidebar-tab-item"]):
                dash_outs = build_dashboard_panel(
                    state,
                    tabs=tabs,
                    upload_handles=upload_handles,
                    pipeline_handles=pipeline_handles,
                )

        # ===== 顶部流程导航按钮事件 =====
        nav_upload_btn.click(fn=lambda: gr.Tabs(selected="upload"), outputs=[tabs])
        nav_detection_btn.click(fn=lambda: gr.Tabs(selected="detection"), outputs=[tabs])
        nav_seg_btn.click(fn=lambda: gr.Tabs(selected="segmentation"), outputs=[tabs])
        nav_diag_btn.click(fn=lambda: gr.Tabs(selected="diagnosis"), outputs=[tabs])
        nav_eval_btn.click(fn=lambda: gr.Tabs(selected="evaluation"), outputs=[tabs])
        nav_pipeline_btn.click(fn=lambda: gr.Tabs(selected="pipeline"), outputs=[tabs])
        nav_dash_btn.click(fn=lambda: gr.Tabs(selected="dashboard"), outputs=[tabs])

        # Dashboard 自动刷新
        dash_status, dash_dataset, dash_recent, dash_errors, dash_chart, dash_workflow, dash_refresh = dash_outs
        app.load(
            fn=_refresh_dashboard,
            inputs=[state],
            outputs=[dash_status, dash_dataset, dash_recent, dash_errors, dash_chart, dash_workflow],
        )

    return app, custom_css, theme, clock_head


def main():
    parser = argparse.ArgumentParser(description="EchoDVT Dashboard")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    app, css, theme, head = build_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
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


if __name__ == "__main__":
    main()
