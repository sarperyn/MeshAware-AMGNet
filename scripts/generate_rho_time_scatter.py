from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from meshaware_data.reporting import SampleRecordRepository

DEFAULT_INPUT_GLOB = "datasets/diffusion/large/**/diffusion_reports/*.csv"
@dataclass(frozen=True)
class Point:
    h: float
    pattern: str
    epsilon: float
    theta: float
    rho: float
    elapsed: float
    count: int


def parse_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not raw or raw.strip().lower() == "auto":
        return default
    return tuple(float(v.strip()) for v in raw.split(",") if v.strip())


def available_h_values(points: list[Point]) -> tuple[float, ...]:
    return tuple(sorted({point.h for point in points}, reverse=True))


def load_points(input_glob: str) -> tuple[list[Point], list[str]]:
    paths = sorted(glob.glob(input_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files matched {input_glob}")
    records = SampleRecordRepository.from_glob(input_glob).all()
    grouped: dict[tuple[float, str, float, float], list[tuple[float, float]]] = defaultdict(list)
    for record in records:
        try:
            meta = record.sample_meta
            metrics = record.metrics
            key = (
                round(float(meta["h"]), 8),
                str(meta.get("pattern", "")),
                round(float(meta["epsilon"]), 8),
                round(float(meta["theta"]), 8),
            )
            grouped[key].append((float(metrics["rho"]), float(metrics["elapsed_sec"])))
        except (KeyError, TypeError, ValueError):
            continue
    points = [
        Point(
            h=key[0],
            pattern=key[1],
            epsilon=key[2],
            theta=key[3],
            rho=statistics.mean(v[0] for v in values),
            elapsed=statistics.mean(v[1] for v in values),
            count=len(values),
        )
        for key, values in grouped.items()
    ]
    return points, paths


def normalize_by_test_case(points: list[Point]) -> tuple[list[Point], list[float], list[float], int, int]:
    groups: dict[tuple[float, str, float], list[Point]] = defaultdict(list)
    for point in points:
        groups[(point.h, point.pattern, point.epsilon)].append(point)

    kept: list[Point] = []
    norm_rho: list[float] = []
    norm_time: list[float] = []
    skipped = 0
    used_groups = 0
    for group in groups.values():
        if len(group) < 2:
            skipped += len(group)
            continue
        rhos = [p.rho for p in group]
        times = [p.elapsed for p in group]
        rho_mean = statistics.mean(rhos)
        time_mean = statistics.mean(times)
        rho_std = statistics.pstdev(rhos)
        time_std = statistics.pstdev(times)
        if rho_std <= 0 or time_std <= 0:
            skipped += len(group)
            continue
        used_groups += 1
        for point in group:
            kept.append(point)
            norm_rho.append((point.rho - rho_mean) / rho_std)
            norm_time.append((point.elapsed - time_mean) / time_std)
    return kept, norm_rho, norm_time, skipped, used_groups


def axis_limits(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1e-9)
    return lo - 0.08 * span, hi + 0.08 * span


def ticks(lo: float, hi: float) -> list[int]:
    return [v for v in range(math.ceil(lo), math.floor(hi) + 1) if lo <= v <= hi]


def fmt_h(h: float) -> str:
    return f"{h:.3e}"


def write_csv(path: Path, points: list[Point], x: list[float], y: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "h",
                "pattern",
                "epsilon",
                "theta",
                "rho",
                "elapsed_sec",
                "normalized_rho",
                "normalized_elapsed_sec",
                "count",
            ]
        )
        for point, nx, ny in zip(points, x, y):
            writer.writerow([point.h, point.pattern, point.epsilon, point.theta, point.rho, point.elapsed, nx, ny, point.count])


def write_svg(
    path: Path,
    points: list[Point],
    x_values: list[float],
    y_values: list[float],
    h_values: tuple[float, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 800, 560
    left, right, top, bottom = 90, 765, 35, 485
    plot_width = right - left
    plot_height = bottom - top
    x_lo, x_hi = axis_limits(x_values)
    y_lo, y_hi = axis_limits(y_values)

    def sx(value: float) -> float:
        return left + (value - x_lo) * plot_width / (x_hi - x_lo)

    def sy(value: float) -> float:
        return bottom - (value - y_lo) * plot_height / (y_hi - y_lo)

    colors = ("#440154", "#21918c", "#fde725", "#3366cc", "#dc3912")
    color_by_h = {
        round(h, 8): colors[index % len(colors)]
        for index, h in enumerate(h_values)
    }
    lines = [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'),
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for tick in ticks(x_lo, x_hi):
        x = sx(tick)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
            'stroke="#dddddd" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{bottom + 20}" text-anchor="middle" '
            f'font-family="serif" font-size="12">{tick}</text>'
        )
    for tick in ticks(y_lo, y_hi):
        y = sy(tick)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
            'stroke="#dddddd" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="serif" font-size="12">{tick}</text>'
        )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="none" stroke="black" stroke-width="1"/>'
    )
    for point, x_value, y_value in zip(points, x_values, y_values):
        color = color_by_h[round(point.h, 8)]
        lines.append(
            f'<circle cx="{sx(x_value):.2f}" cy="{sy(y_value):.2f}" r="3" '
            f'fill="{color}" stroke="black" stroke-width="0.35"/>'
        )
    lines.extend(
        [
            (f'<text x="{(left + right) / 2:.2f}" y="535" text-anchor="middle" '
            'font-family="serif" font-size="15">normalized rho</text>'),
            (f'<text x="24" y="{(top + bottom) / 2:.2f}" text-anchor="middle" '
            'font-family="serif" font-size="15" '
            f'transform="rotate(-90 24 {(top + bottom) / 2:.2f})">'
            "normalized solve time</text>"),
        ]
    )
    legend_x = right - 150
    legend_y = top + 18
    lines.append(
        f'<rect x="{legend_x - 10}" y="{legend_y - 15}" width="155" '
        f'height="{24 * len(h_values) + 10}" fill="white" fill-opacity="0.92" '
        'stroke="#777777" stroke-width="0.7"/>'
    )
    for index, h in enumerate(h_values):
        y = legend_y + index * 24
        color = color_by_h[round(h, 8)]
        lines.append(
            f'<circle cx="{legend_x}" cy="{y}" r="4" fill="{color}" '
            'stroke="black" stroke-width="0.4"/>'
        )
        lines.append(
            f'<text x="{legend_x + 12}" y="{y + 4}" font-family="serif" '
            f'font-size="11">h={fmt_h(h)}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(path: Path, points: list[Point], x_values: list[float], y_values: list[float], h_values: tuple[float, ...], scale: int) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNG output uses matplotlib. Install project requirements, e.g. `pip install -r requirements.txt`, "
            "or run with the repo .venv if matplotlib is installed there."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    x_lo, x_hi = axis_limits(x_values)
    y_lo, y_hi = axis_limits(y_values)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )
    dpi = 100 * max(1, scale)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=dpi)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.13)

    color_map = plt.get_cmap("viridis")
    color_denominator = max(1, len(h_values) - 1)
    for color_index, h in enumerate(h_values):
        h_key = round(h, 8)
        xs = [x for point, x in zip(points, x_values) if round(point.h, 8) == h_key]
        ys = [y for point, y in zip(points, y_values) if round(point.h, 8) == h_key]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=7.0,
            c=[color_map(color_index / color_denominator)],
            edgecolors="black",
            linewidths=0.25,
            label=f"h={fmt_h(h)}",
            zorder=3,
        )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(r"normalized $\rho$", fontsize=9)
    ax.set_ylabel(r"normalized $t$", fontsize=9)
    ax.set_xticks(ticks(x_lo, x_hi))
    ax.set_yticks(ticks(y_lo, y_hi))
    ax.tick_params(axis="both", labelsize=8, length=3)
    ax.grid(True, color="#d4d4d4", linewidth=0.55, zorder=0)
    legend = ax.legend(
        loc="upper right",
        fontsize=6.2,
        frameon=True,
        framealpha=0.9,
        fancybox=False,
        edgecolor="#777777",
        markerscale=0.8,
        borderpad=0.35,
        labelspacing=0.35,
        handletextpad=0.35,
    )
    legend.get_frame().set_linewidth(0.5)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rho/time scatter normalized inside each same test case: h + pattern + epsilon."
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--out-dir", default="results/figures/rho_time_scatter")
    parser.add_argument("--h-values", default="auto")
    parser.add_argument("--png-name", default="rho_time_scatter.png")
    parser.add_argument("--svg-name", default="rho_time_scatter.svg")
    parser.add_argument("--csv-name", default="rho_time_scatter.csv")
    parser.add_argument("--png-scale", type=int, default=3)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    points, paths = load_points(args.input_glob)
    all_h_values = available_h_values(points)
    h_values = parse_floats(args.h_values, all_h_values)
    if not h_values:
        raise ValueError(
            "Could not discover h values from the input reports. Supply "
            "--h-values explicitly or check the input glob."
        )
    h_set = {round(v, 8) for v in h_values}
    points = [point for point in points if round(point.h, 8) in h_set]
    if not points:
        if not all_h_values:
            raise ValueError("No compatible points found in the input reports.")
        print("WARNING: requested h values had no compatible points; using all available h values.")
        h_values = all_h_values
        h_set = {round(v, 8) for v in h_values}
        points, _ = load_points(args.input_glob)
        points = [point for point in points if round(point.h, 8) in h_set]
        if not points:
            raise ValueError("No compatible points found after falling back to all available h values.")

    points, norm_rho, norm_time, skipped, used_groups = normalize_by_test_case(points)
    if not points:
        raise ValueError("No same-test-case subset had enough variation to normalize.")

    print(f"Loaded {len(points)} averaged points from {len(paths)} file(s).")
    print(f"Normalization: each same test case independently: h + pattern + epsilon ({used_groups} groups).")
    if skipped:
        print(f"WARNING: skipped {skipped} point(s) from same-test-case subsets without variation.")
    if max((p.count for p in points), default=0) < 2:
        print("WARNING: no repeated samples were found; each averaged point is based on one record.")

    csv_path = out_dir / args.csv_name
    svg_path = out_dir / args.svg_name
    png_path = out_dir / args.png_name
    write_csv(csv_path, points, norm_rho, norm_time)
    print(f"Wrote {csv_path}")
    write_svg(svg_path, points, norm_rho, norm_time, h_values)
    print(f"Wrote {svg_path}")
    if not args.no_png:
        try:
            write_png(png_path, points, norm_rho, norm_time, h_values, max(1, args.png_scale))
        except RuntimeError as exc:
            print(f"WARNING: could not write {png_path}: {exc}")
        else:
            print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
