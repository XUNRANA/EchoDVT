"""Small HTML helpers for a more consistent Gradio UI."""

from html import escape


def render_page_header(title: str, desc: str, eyebrow: str | None = None) -> str:
    eyebrow_html = (
        f'<div class="ui-page-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    )
    return f"""
    <div class="ui-page-header">
        {eyebrow_html}
        <h2 class="ui-page-title">{escape(title)}</h2>
        <p class="ui-page-desc">{escape(desc)}</p>
    </div>
    """


def render_note(text: str) -> str:
    return f'<div class="ui-note">{escape(text)}</div>'


def render_empty_state(title: str, desc: str) -> str:
    return f"""
    <div class="ui-empty-state">
        <div class="ui-empty-title">{escape(title)}</div>
        <div class="ui-empty-desc">{escape(desc)}</div>
    </div>
    """


def render_summary_card(
    *,
    tone: str,
    title: str,
    metric: str,
    meta: str = "",
    detail: str = "",
    eyebrow: str = "诊断结果",
) -> str:
    return f"""
    <div class="ui-summary-card tone-{escape(tone)}">
        <div class="ui-summary-eyebrow">{escape(eyebrow)}</div>
        <div class="ui-summary-title">{escape(title)}</div>
        <div class="ui-summary-metric">{escape(metric)}</div>
        <div class="ui-summary-meta">{escape(meta)}</div>
        <div class="ui-summary-detail">{escape(detail)}</div>
    </div>
    """
