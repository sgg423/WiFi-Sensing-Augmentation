"""Create a paper-ready BeamDiff and TimeVAE ratio comparison plot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_prefix", type=Path, help="output path without extension")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load(path: Path) -> dict[str, list[float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty input CSV: {path}")
    fields = (
        "ratio_percent",
        "beamdiff_mean",
        "beamdiff_sd",
        "timevae_mean",
        "timevae_sd",
    )
    return {field: [float(row[field]) for row in rows] for field in fields}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Times New Roman Bold.ttf" if bold else "Times New Roman.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    return ImageFont.truetype(str(path), size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str = "black",
) -> None:
    box = draw.textbbox((0, 0), value, font=text_font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        value,
        font=text_font,
        fill=fill,
    )


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    offset = 0.0
    while offset < length:
        stop = min(offset + dash, length)
        draw.line(
            (x1 + ux * offset, y1 + uy * offset, x1 + ux * stop, y1 + uy * stop),
            fill=fill,
            width=width,
        )
        offset += dash + gap


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")

    data = load(args.input_csv)
    ratios = data["ratio_percent"]

    width, height = 2880, 1800
    left, right, top, bottom = 360, 120, 130, 280
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    all_low = [
        mean - sd
        for means, sds in (
            (data["beamdiff_mean"], data["beamdiff_sd"]),
            (data["timevae_mean"], data["timevae_sd"]),
        )
        for mean, sd in zip(means, sds)
    ]
    all_high = [
        mean + sd
        for means, sds in (
            (data["beamdiff_mean"], data["beamdiff_sd"]),
            (data["timevae_mean"], data["timevae_sd"]),
        )
        for mean, sd in zip(means, sds)
    ]
    y_min = int(min(all_low) - 1)
    y_max = int(max(all_high) + 1)
    y_min -= y_min % 2
    if y_max % 2:
        y_max += 1

    def sx(value: float) -> float:
        # Leave room for a small horizontal dodge between the two methods.
        return plot_left + ((value + 4.0) / 108.0) * plot_width

    def sy(value: float) -> float:
        return plot_bottom - ((value - y_min) / (y_max - y_min)) * plot_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label_font = font(62)
    tick_font = font(50)
    legend_font = font(52)

    # Horizontal grid and y-axis labels.
    for value in range(y_min, y_max + 1, 2):
        y = sy(value)
        draw.line((plot_left, y, plot_right, y), fill="#c6c6c6", width=3)
        box = draw.textbbox((0, 0), f"{value}%", font=tick_font)
        draw.text(
            (plot_left - 32 - (box[2] - box[0]), y - (box[3] - box[1]) / 2),
            f"{value}%",
            font=tick_font,
            fill="black",
        )

    # Axes and x-axis labels.
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=5)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=5)
    for ratio in ratios:
        x = sx(ratio)
        draw.line((x, plot_bottom, x, plot_bottom + 18), fill="black", width=4)
        centered_text(draw, (x, plot_bottom + 72), f"+{ratio:g}%", tick_font)

    # Axis titles.
    centered_text(draw, ((plot_left + plot_right) / 2, height - 95), "Augmented Data", label_font)
    y_label = Image.new("RGBA", (600, 120), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    centered_text(y_draw, (300, 60), "Accuracy", label_font)
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (70, int((height - y_label.height) / 2)), y_label)

    beamdiff_color = "#0072B2"
    timevae_color = "#D55E00"
    series = (
        (
            "BeamDiff",
            data["beamdiff_mean"],
            data["beamdiff_sd"],
            beamdiff_color,
            "solid",
            "circle",
            -1.25,
        ),
        (
            "TimeVAE",
            data["timevae_mean"],
            data["timevae_sd"],
            timevae_color,
            "dashed",
            "square",
            1.25,
        ),
    )

    for _, means, sds, color, style, marker, offset in series:
        points = [(sx(ratio + offset), sy(mean)) for ratio, mean in zip(ratios, means)]
        for first, second in zip(points, points[1:]):
            if style == "solid":
                draw.line((*first, *second), fill=color, width=10)
            else:
                dashed_line(draw, first, second, color, 10, 42, 25)

        for (x, y), mean, sd in zip(points, means, sds):
            low_y, high_y = sy(mean - sd), sy(mean + sd)
            if style == "solid":
                draw.line((x, high_y, x, low_y), fill=color, width=6)
            else:
                dashed_line(
                    draw,
                    (x, high_y),
                    (x, low_y),
                    color,
                    6,
                    25,
                    16,
                )
            draw.line((x - 22, high_y, x + 22, high_y), fill=color, width=6)
            draw.line((x - 22, low_y, x + 22, low_y), fill=color, width=6)
            radius = 18
            if marker == "circle":
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            else:
                draw.rectangle(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill="white",
                    outline=color,
                    width=7,
                )

    # Legend with line and marker encodings that remain distinct in grayscale.
    legend_x, legend_y = plot_left + 45, plot_top + 38
    draw.rectangle(
        (legend_x - 25, legend_y - 28, legend_x + 820, legend_y + 88),
        fill="white",
        outline="#666666",
        width=3,
    )
    draw.line(
        (legend_x, legend_y + 28, legend_x + 130, legend_y + 28),
        fill=beamdiff_color,
        width=9,
    )
    draw.ellipse(
        (legend_x + 47, legend_y + 10, legend_x + 83, legend_y + 46),
        fill=beamdiff_color,
    )
    draw.text((legend_x + 150, legend_y - 3), "BeamDiff", font=legend_font, fill="black")
    start_x = legend_x + 405
    dashed_line(
        draw,
        (start_x, legend_y + 28),
        (start_x + 130, legend_y + 28),
        timevae_color,
        9,
        38,
        20,
    )
    draw.rectangle(
        (start_x + 47, legend_y + 10, start_x + 83, legend_y + 46),
        fill="white",
        outline=timevae_color,
        width=7,
    )
    draw.text((start_x + 150, legend_y - 3), "TimeVAE", font=legend_font, fill="black")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    image.save(png_path, dpi=(args.dpi, args.dpi))
    image.save(pdf_path, "PDF", resolution=args.dpi)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
