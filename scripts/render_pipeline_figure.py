from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


PAGE_WIDTH = 560
PAGE_HEIGHT = 260

BG = colors.HexColor("#F7F4EE")
INK = colors.HexColor("#1E2430")
MUTED = colors.HexColor("#5C6675")
LINE = colors.HexColor("#B9C1CB")
ACCENT = colors.HexColor("#1F5E7A")
ACCENT_LIGHT = colors.HexColor("#D9ECF4")
WARM = colors.HexColor("#EAA15A")
WARM_LIGHT = colors.HexColor("#FCE7CF")
GREEN = colors.HexColor("#5E8B62")
GREEN_LIGHT = colors.HexColor("#E0ECDD")
SLATE = colors.HexColor("#6E7EA1")
SLATE_LIGHT = colors.HexColor("#E5EAF5")


def fit_text(c: canvas.Canvas, text: str, font_name: str, max_size: float, width: float) -> float:
    size = max_size
    while size > 6 and stringWidth(text, font_name, size) > width:
        size -= 0.25
    return size


def rounded_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: list[str],
    fill_color,
    stroke_color=LINE,
    title_color=INK,
    body_color=MUTED,
    title_size: float = 11,
    body_size: float = 8.4,
    radius: float = 14,
) -> None:
    c.setFillColor(fill_color)
    c.setStrokeColor(stroke_color)
    c.setLineWidth(1.0)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)

    title_fit = fit_text(c, title, "Helvetica-Bold", title_size, w - 18)
    c.setFillColor(title_color)
    c.setFont("Helvetica-Bold", title_fit)
    c.drawString(x + 9, y + h - 18, title)

    c.setFillColor(body_color)
    text_y = y + h - 33
    for line in body:
        fit = fit_text(c, line, "Helvetica", body_size, w - 18)
        c.setFont("Helvetica", fit)
        c.drawString(x + 9, text_y, line)
        text_y -= fit + 4


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=ACCENT, width: float = 2.0) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = 0
    if x2 != x1:
        angle = 0 if x2 > x1 else 180
    elif y2 > y1:
        angle = 90
    else:
        angle = -90
    size = 6
    if angle == 0:
        pts = [(x2, y2), (x2 - size * 1.6, y2 + size * 0.9), (x2 - size * 1.6, y2 - size * 0.9)]
    elif angle == 180:
        pts = [(x2, y2), (x2 + size * 1.6, y2 + size * 0.9), (x2 + size * 1.6, y2 - size * 0.9)]
    elif angle == 90:
        pts = [(x2, y2), (x2 - size * 0.9, y2 - size * 1.6), (x2 + size * 0.9, y2 - size * 1.6)]
    else:
        pts = [(x2, y2), (x2 - size * 0.9, y2 + size * 1.6), (x2 + size * 0.9, y2 + size * 1.6)]
    path = c.beginPath()
    path.moveTo(*pts[0])
    path.lineTo(*pts[1])
    path.lineTo(*pts[2])
    path.close()
    c.drawPath(path, stroke=0, fill=1)


def model_chip(c: canvas.Canvas, x: float, y: float, w: float, h: float, text: str, fill_color) -> None:
    c.setFillColor(fill_color)
    c.setStrokeColor(colors.white)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 10, fill=1, stroke=1)
    font_size = fit_text(c, text, "Helvetica-Bold", 8.2, w - 10)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", font_size)
    text_width = stringWidth(text, "Helvetica-Bold", font_size)
    c.drawString(x + (w - text_width) / 2, y + h / 2 - font_size / 3, text)


def draw_bracket(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str) -> None:
    c.setStrokeColor(LINE)
    c.setLineWidth(1.2)
    c.line(x, y + h, x, y)
    c.line(x, y + h, x + 12, y + h)
    c.line(x, y, x + 12, y)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 8.2)
    c.drawString(x + 16, y + h / 2 - 3, label)


def render_pipeline(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=(PAGE_WIDTH, PAGE_HEIGHT))

    c.setFillColor(BG)
    c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)

    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(28, PAGE_HEIGHT - 28, "MI-EEG Classification Pipeline")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.8)
    c.drawString(28, PAGE_HEIGHT - 42, "Shared preprocessing, model-specific decoding, and 4-class prediction.")

    left_x = 30
    y = 78
    box_w = 114
    box_h = 112
    gap = 16

    rounded_box(
        c,
        left_x,
        y,
        box_w,
        box_h,
        "Raw MI-EEG",
        ["22 channels", "2 sessions / subject", "Cue window: 2-6 s"],
        SLATE_LIGHT,
    )
    rounded_box(
        c,
        left_x + box_w + gap,
        y,
        box_w,
        box_h,
        "Preprocess",
        ["Butterworth 8-30 Hz", "Trial-wise standardize", "Downsample to 125 Hz"],
        ACCENT_LIGHT,
    )
    rounded_box(
        c,
        left_x + 2 * (box_w + gap),
        y,
        150,
        box_h,
        "Model Bank",
        ["Shared train/val/test split", "Sequence readout for CfC/LSTM", "EEGNet-style frontend for hybrid"],
        WARM_LIGHT,
    )
    rounded_box(
        c,
        PAGE_WIDTH - 130,
        y,
        100,
        box_h,
        "Prediction",
        ["Left hand", "Right hand", "Feet", "Tongue"],
        GREEN_LIGHT,
    )

    arrow(c, left_x + box_w, y + box_h / 2, left_x + box_w + gap - 2, y + box_h / 2)
    arrow(c, left_x + 2 * box_w + gap, y + box_h / 2, left_x + 2 * box_w + 2 * gap - 2, y + box_h / 2)
    arrow(c, left_x + 2 * (box_w + gap) + 150, y + box_h / 2, PAGE_WIDTH - 132, y + box_h / 2)

    chip_x = left_x + 2 * (box_w + gap) + 12
    chip_y = y + 18
    chip_w = 58
    chip_h = 24
    chip_gap_x = 12
    chip_gap_y = 10
    model_chip(c, chip_x, chip_y + chip_h + chip_gap_y, chip_w, chip_h, "CfC", colors.white)
    model_chip(c, chip_x + chip_w + chip_gap_x, chip_y + chip_h + chip_gap_y, chip_w, chip_h, "LSTM", colors.white)
    model_chip(c, chip_x, chip_y, chip_w, chip_h, "EEGNet", colors.white)
    model_chip(c, chip_x + chip_w + chip_gap_x, chip_y, chip_w, chip_h, "Hybrid-CfC", colors.white)

    draw_bracket(c, 24, y - 24, 0, box_h + 48, "Input + preprocessing")
    draw_bracket(c, left_x + 2 * (box_w + gap) - 10, y - 24, 0, box_h + 48, "Decoding models")

    c.setFont("Helvetica", 8.2)
    c.setFillColor(MUTED)
    footer = "Evaluation: subject-dependent 5-fold CV, Adam optimizer, early stopping, accuracy and macro F1."
    c.drawString(28, 24, footer)

    c.showPage()
    c.save()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    render_pipeline(repo_root / "pipeline_placeholder.pdf")
