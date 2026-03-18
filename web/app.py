#!/usr/bin/env python3
"""EchoDVT Web app entrypoint."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "sam2"))
sys.path.insert(0, str(PROJECT_ROOT / "yolo"))

import gradio as gr

from tabs.dashboard import (
    build_dashboard_panel, _refresh_dashboard,
    _get_unified_model_meta, _get_dataset_stats,
    _refresh_dashboard_panel,
)
from tabs.upload import build_upload_tab
from tabs.detection import build_detection_tab
from tabs.segmentation import build_segmentation_tab
from tabs.diagnosis import build_diagnosis_tab
from tabs.evaluation import build_report_tab
from tabs.pipeline import build_pipeline_tab


CSS_PATH = Path(__file__).parent / "assets" / "custom.css"


def build_app():
    """构建 Dashboard 风格的 Gradio 应用"""
    custom_css = ""
    if CSS_PATH.exists():
        custom_css = CSS_PATH.read_text(encoding="utf-8")

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
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

    # ===== 构建时读取统计数据 =====
    meta = _get_unified_model_meta()
    try:
        ds = _get_dataset_stats()
    except Exception:
        ds = {}

    acc_val = meta.get("val_accuracy")
    acc_str = f"{acc_val * 100:.1f}%" if acc_val is not None else "—"
    feat_dim = len(meta.get("feature_cols", [])) or 21
    total_cases = ds.get("total_cases", "—")
    model_status = "已就绪" if meta else "未加载"
    model_status_cls = "dot-ok" if meta else "dot-off"

    with gr.Blocks(title="EchoDVT 智能诊断系统") as app:

        # ===== 顶部栏 =====
        gr.HTML(f"""
        <div class="topbar-shell">
            <div class="topbar">
                <div class="topbar-brand">
                    <span class="brand-mark" aria-hidden="true"></span>
                    <div class="brand-copy">
                        <div class="topbar-title">EchoDVT</div>
                        <div class="topbar-subtitle">深静脉血栓超声辅助诊断系统</div>
                    </div>
                </div>
                <div class="topbar-right">
                    <div class="topbar-status">
                        <span class="topbar-dot {model_status_cls}"></span>
                        <span>{model_status}</span>
                    </div>
                    <div class="topbar-clock" id="beijing-clock">
                        <span class="clock-date">加载中...</span> --:--:--
                    </div>
                </div>
            </div>
        </div>
        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
             style="display:none"
             onload="
                function bjTick(){{
                    var d=new Date(),u=d.getTime()+d.getTimezoneOffset()*60000,
                        b=new Date(u+8*3600000),
                        ds=b.getFullYear()+'-'+String(b.getMonth()+1).padStart(2,'0')+'-'+String(b.getDate()).padStart(2,'0'),
                        ts=String(b.getHours()).padStart(2,'0')+':'+String(b.getMinutes()).padStart(2,'0')+':'+String(b.getSeconds()).padStart(2,'0'),
                        el=document.getElementById('beijing-clock');
                    if(el) el.innerHTML='<span class=clock-date>'+ds+'</span> '+ts;
                }}
                bjTick(); setInterval(bjTick,1000);
             ">
        """)

        # ===== 全局摘要行 =====
        gr.HTML(f"""
        <div class="app-meta-shell">
            <div class="app-meta-strip">
                <div class="app-meta-item">
                    <div class="app-meta-label">验证准确率</div>
                    <div class="app-meta-value">{acc_str}</div>
                </div>
                <div class="app-meta-item">
                    <div class="app-meta-label">特征维度</div>
                    <div class="app-meta-value">{feat_dim}</div>
                </div>
                <div class="app-meta-item">
                    <div class="app-meta-label">样本总量</div>
                    <div class="app-meta-value">{total_cases}</div>
                </div>
                <div class="app-meta-item">
                    <div class="app-meta-label">统一模型</div>
                    <div class="app-meta-value app-meta-inline">
                        <span class="topbar-dot {model_status_cls}"></span>{model_status}
                    </div>
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

        # ===== 水平 Tabs =====
        with gr.Tabs(elem_classes=["top-tabs"]) as tabs:
            with gr.Tab("仪表盘", id="dashboard"):
                dash_outs = build_dashboard_panel(state)

            with gr.Tab("数据输入", id="upload"):
                upload_handles = build_upload_tab(state)

            with gr.Tab("一键分析", id="pipeline"):
                pipeline_handles = build_pipeline_tab(state)

            with gr.Tab("目标检测", id="detection"):
                build_detection_tab(state)

            with gr.Tab("视频分割", id="segmentation"):
                build_segmentation_tab(state)

            with gr.Tab("DVT 诊断", id="diagnosis"):
                build_diagnosis_tab(state)

            with gr.Tab("导出报告", id="report"):
                build_report_tab(state)

        # Dashboard 自动刷新
        (
            dash_intro,
            dash_status,
            dash_dataset,
            dash_errors,
            dash_chart,
            dash_refresh,
        ) = dash_outs
        app.load(
            fn=lambda s: _refresh_dashboard_panel(s),
            inputs=[state],
            outputs=[dash_intro, dash_status, dash_dataset, dash_errors, dash_chart],
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
