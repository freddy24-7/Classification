"""Builds the explanatory PDF for a topic (reportlab). All PDF generation lives here."""
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .code_ref import code_reference  # noqa: E402
from .registry import Topic  # noqa: E402

ACCENT = colors.HexColor("#2f5d8a")
STATUS_COLORS = {"pass": "#2e7d4f", "warn": "#b7791f", "fail": "#b83232"}
MAX_TABLE_ROWS = 25

_styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_styles["Title"], alignment=0, fontSize=20, spaceAfter=4, textColor=ACCENT)
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=ACCENT)
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=10, leading=14)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.5, leading=9.5, backColor=colors.HexColor("#f4f5f7"),
                      borderPadding=6, leftIndent=4, rightIndent=4)


def _esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _bullets(items):
    return ListFlowable([ListItem(Paragraph(_esc(i), BODY), leftIndent=12) for i in items],
                        bulletType="bullet", start="•", leftIndent=12)


def _chart_png(chart: dict) -> io.BytesIO:
    labels, values = chart["labels"], chart["values"]
    fig, ax = plt.subplots(figsize=(7, max(1.6, 0.28 * len(labels) + 0.8)), dpi=150)
    threshold = chart.get("threshold")
    bar_colors = ["#b83232" if threshold is not None and v >= threshold else "#2f5d8a" for v in values]
    ax.barh(labels[::-1], values[::-1], color=bar_colors[::-1])
    if threshold is not None:
        ax.axvline(threshold, color="#b83232", linestyle="--", linewidth=1)
        ax.text(threshold, len(labels) - 0.4, f" threshold {threshold:g}{chart.get('unit', '')}",
                color="#b83232", fontsize=7, va="bottom")
    ax.set_title(chart["title"], fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


MAX_TABLE_COLS = 9
GROUP_LABELS = {"data": "Dataset", "intuition": "Model intuition", "compare": "Model comparison",
                "metrics": "Evaluation metrics", "handson": "Hands-on", "workflow": "Analyst walkthrough", "extras": "Beyond the course"}
NOTEBOOKS = {"handson": "notebooks/10_7_classification_hands_on.ipynb"}


def _table(t: dict, width: float):
    cols = t["columns"][:MAX_TABLE_COLS]
    rows = [r[:MAX_TABLE_COLS] for r in t["rows"][:MAX_TABLE_ROWS]]
    data = [cols] + [["" if v is None else _esc(v) for v in r] for r in rows]
    data = [[Paragraph(str(c), SMALL) for c in row] for row in data]
    col_w = min(width / len(cols), 5 * cm)
    tbl = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[col_w] * len(cols))
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    title = ParagraphStyle("tt", parent=BODY, keepWithNext=True, spaceAfter=3)
    out = [Paragraph(f"<b>{_esc(t['title'])}</b>", title), tbl]
    notes = []
    if len(t["rows"]) > MAX_TABLE_ROWS:
        notes.append(f"{len(t['rows']) - MAX_TABLE_ROWS} more rows")
    if len(t["columns"]) > MAX_TABLE_COLS:
        notes.append(f"{len(t['columns']) - MAX_TABLE_COLS} more columns")
    if notes:
        out.append(Paragraph(" and ".join(notes) + " omitted; see the web page for the full table.", SMALL))
    return out + [Spacer(1, 8)]


def _image(png: io.BytesIO, width: float, max_height: float = 19 * cm):
    img = Image(png)
    scale = min(width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return img


def build_pdf(topic: Topic, result: dict, params: dict | None = None) -> bytes:
    refs = [code_reference(f) for f in [topic.func] + topic.helpers]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title=f"{topic.number} {topic.title}", author="Classification Lab")
    status = result.get("status")
    verdict = (f'<font color="{STATUS_COLORS[status]}"><b>{status.upper()}</b></font> on the diabetes dataset: '
               if status else "On the diabetes dataset: ")

    story = [
        Paragraph(f"{GROUP_LABELS[topic.group]} {topic.number}", SMALL),
        Paragraph(_esc(topic.title), H1),
        Paragraph(verdict + _esc(result["message"]), BODY),
    ]
    if params:
        shown = ", ".join(f"{k} = {'default' if v is None else v}" for k, v in params.items())
        story.append(Paragraph(f"Settings: <font face='Courier'>{_esc(shown)}</font>", SMALL))
    story += [
        Paragraph("What it is", H2), Paragraph(_esc(topic.what), BODY),
        Paragraph("Why it matters for machine learning", H2), Paragraph(_esc(topic.why), BODY),
        Paragraph("How it works", H2), _bullets(topic.how),
        Paragraph("Result on the diabetes dataset", H2),
    ]

    metrics = [[Paragraph(f"<b>{_esc(m['label'])}</b>", SMALL), Paragraph(_esc(m["value"]), SMALL)]
               for m in result["metrics"]]
    mt = Table(metrics, hAlign="LEFT", colWidths=[6 * cm, 5 * cm])
    mt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc"))]))
    story += [mt, Spacer(1, 8)]

    for t in [t for t in result["tables"] if t.get("top")]:
        story += _table(t, doc.width)
    if result.get("chart") and result["chart"]["labels"]:
        story += [_image(_chart_png(result["chart"]), doc.width * 0.9), Spacer(1, 6)]
    for im in result.get("images", []):
        title = ParagraphStyle("it", parent=BODY, keepWithNext=True, spaceAfter=3)
        story += [Paragraph(f"<b>{_esc(im['title'])}</b>", title),
                  _image(io.BytesIO(base64.b64decode(im["png"])), doc.width), Spacer(1, 8)]

    for t in [t for t in result["tables"] if not t.get("top")]:
        story += _table(t, doc.width)

    step_title = ParagraphStyle("st", parent=H2, fontSize=11.5, keepWithNext=True)
    decision = ParagraphStyle("dec", parent=BODY, backColor=colors.HexColor("#e8eef5"), borderPadding=5,
                              spaceBefore=4, spaceAfter=6)
    for i, s in enumerate(result.get("steps", []), 1):
        story += [Paragraph(f"Step {i}. {_esc(s['title'])}", step_title), Paragraph(_esc(s["text"]), BODY),
                  Paragraph(f"<b>Decision:</b> {_esc(s['decision'])}", decision)]
        if s["metrics"]:
            rows = [[Paragraph(f"<b>{_esc(m['label'])}</b>", SMALL), Paragraph(_esc(m["value"]), SMALL)]
                    for m in s["metrics"]]
            st = Table(rows, hAlign="LEFT", colWidths=[6 * cm, 5 * cm])
            st.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc"))]))
            story += [st, Spacer(1, 6)]
        for im in s["images"]:
            story += [_image(io.BytesIO(base64.b64decode(im["png"])), doc.width * 0.85, 11 * cm), Spacer(1, 6)]
        for t in s["tables"]:
            story += _table(t, doc.width)

    story += [Paragraph(topic.fix_title, H2), _bullets(topic.fix)]
    if topic.pitfalls:
        story += [Paragraph("Watch out for", H2), _bullets(topic.pitfalls)]

    source = f" Based on {NOTEBOOKS[topic.group]}." if topic.group in NOTEBOOKS else ""
    story += [Paragraph("Reference code", H2)]
    for i, ref in enumerate(refs):
        story += [
            Paragraph(f"<font face='Courier'>{_esc(ref['file'])}</font>, lines {ref['start_line']}–{ref['end_line']}, "
                      f"function <font face='Courier'>{ref['function']}()</font>."
                      + (source if i == 0 else ""), SMALL),
            Spacer(1, 4),
            Preformatted(ref["source"], CODE, maxLineLength=108, newLineChars="    "),
            Spacer(1, 8),
        ]

    doc.build(story)
    return buf.getvalue()
