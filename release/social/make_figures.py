#!/usr/bin/env python3
"""Render the launch figures for X and LinkedIn.

    python release/social/make_figures.py

Numbers come from the paper's own LaTeX -- the score table for the scores, the
appendix's model table for release dates -- so a change to the paper shows up
here as a changed figure rather than as a figure that quietly disagrees with it.
`verify_figures.py` checks them against arXiv and the website.

Singles (1600x900):
    ranking_<metric>.png   overall, the three dimensions, the three domains
    gap_vs_frontier.png    how far each open size class sits behind the frontier
    score_over_time.png    score against release date

Composites, because X allows four images per post:
    panel_domains.png      the three domain rankings
    panel_dimensions.png   the three dimension rankings
    panel_gap_time.png     the gap chart beside the trend chart

Type is Inter (fonts/, OFL). Colour comes from the validated categorical
palette: two hues for rankings, and for the gap chart a three-step blue ramp,
because size classes are an ordered quantity rather than unrelated categories,
plus one accent for the all-open summary.
"""
import re
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath

REPO = Path(__file__).resolve().parents[2]
LATEX = REPO / "claude_docs" / "latex" / "sections"
TABLE = LATEX / "Tables" / "results.tex"
APPENDIX = LATEX / "appendix.tex"
OUT = Path(__file__).resolve().parent
FONTS = OUT / "fonts"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
FRONTIER = "#2a78d6"
OPEN = "#eb6834"
RAMP = ["#9dc3f0", "#5a9ae3", "#1f5fa9"]
ACCENT = "#eb6834"

DPI = 200
FIGSIZE = (8, 4.5)          # -> 1600 x 900
SOURCE = "arxiv.org/abs/2510.20091  ·  joeyhou.github.io/CreativityPrism"

METRICS = {
    "overall": "Overall creativity score",
    "quality": "Quality",
    "novelty": "Novelty",
    "diversity": "Diversity",
    "cw": "Creative writing",
    "dt": "Divergent thinking",
    "lr": "Logical reasoning",
}
DIMENSIONS = ["quality", "novelty", "diversity"]
DOMAINS = ["cw", "dt", "lr"]
SIZE_CLASSES = ["<10B", "10-40B", "40-80B"]
TREND = [("quality", "Quality", "#2a78d6"), ("novelty", "Novelty", "#eb6834"),
         ("diversity", "Diversity", "#1baf7a")]


# ---------- data ----------

def use_inter():
    for ttf in sorted(FONTS.glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    if any(FONTS.glob("Inter-*.ttf")):
        plt.rcParams["font.family"] = "Inter"
    plt.rcParams["axes.unicode_minus"] = False


def load_table():
    """Model -> scores + size class, read out of the paper's own table."""
    body = TABLE.read_text().split("\\begin{table}[h]")[1]
    cols = list(METRICS)
    data, group = {}, None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("%") or "&" not in line:
            continue
        cells = [c.strip().replace("\\\\", "").strip() for c in line.split("&")]
        name = cells[0].replace("\\textbf{", "").replace("}", "").strip()
        vals = [c.replace("\\textbf{", "").replace("}", "").strip() for c in cells[1:]]
        nums = [float("0" + v) for v in vals if re.fullmatch(r"\.\d+", v)]
        if len(nums) == len(cols):
            data[name] = dict(zip(cols, nums), group=group)
        elif not any(vals):
            group = name
    if len(data) != 17:
        raise SystemExit("expected 17 models, parsed {}".format(len(data)))
    return data


def load_release_dates():
    """Model -> release date, from the appendix's model table (MM/YYYY)."""
    out = {}
    for line in APPENDIX.read_text().splitlines():
        m = re.match(r"\s*([\w.\-]+)\s*&.*&\s*(\d{2})/(\d{4})\s*\\\\", line)
        if m:
            out[m.group(1)] = date(int(m.group(3)), int(m.group(2)), 1)
    return out


def open_groups(data):
    members = {c: [m for m in data if data[m]["group"] == c] for c in SIZE_CLASSES}
    members["Open (all)"] = [m for m in data if data[m]["group"] != "Frontier-scale"]
    return members


# ---------- drawing primitives ----------

def _px(ax, px):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((px, px))
    return abs(x1 - x0), abs(y1 - y0)


def rounded_bar(ax, x0, y, width, height, color, round_px=4, horizontal=True):
    """A bar with a rounded data-end and a square baseline end."""
    rx, ry = _px(ax, round_px)
    if horizontal:
        r, ry = min(rx, abs(width) / 2), min(ry, height / 2)
        x1, y0, y1 = x0 + width, y - height / 2, y + height / 2
        verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + ry),
                 (x1, y1 - ry), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
    else:
        r, rx = min(ry, abs(height) / 2), min(rx, width / 2)
        x1, y0, y1 = x0 + width, y, y + height
        verts = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + rx, y1),
                 (x1 - rx, y1), (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none"))


def bare(ax, axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    getattr(ax, "{}axis".format(axis)).grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=7.5, colors=INK_2)


def titles(fig, title, subtitle, note, y=0.95):
    fig.text(0.045, y, title, fontsize=15.5, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.045, y - 0.074, subtitle, fontsize=9.5, color=INK_2, va="top")
    fig.text(0.045, 0.04, note, fontsize=7.5, color=INK_2, va="bottom")


def group_legend(target, loc="lower right", **kw):
    handles = [Patch(facecolor=FRONTIER, label="Frontier-scale"),
               Patch(facecolor=OPEN, label="Open")]
    leg = target.legend(handles=handles, loc=loc, frameon=False, fontsize=8,
                        handlelength=0.9, handleheight=0.9, borderpad=0.2, **kw)
    for t in leg.get_texts():
        t.set_color(INK_2)


# ---------- panels ----------

def draw_ranking(ax, data, metric, label_size=8, bar_px=20, compact=False):
    """One ranked bar chart. Used standalone and inside the composites."""
    models = sorted(data, key=lambda m: -data[m][metric])
    top = max(data[m][metric] for m in models)
    ax.set_xlim(0, top * (1.3 if compact else 1.2))
    ax.set_ylim(-0.7, len(models) - 0.3)
    _, height = _px(ax, bar_px)

    best_open = max((m for m in models if data[m]["group"] != "Frontier-scale"),
                    key=lambda m: data[m][metric])
    marked = {models[0]} if compact else {models[0], best_open}
    for i, m in enumerate(models):
        v = data[m][metric]
        color = FRONTIER if data[m]["group"] == "Frontier-scale" else OPEN
        rounded_bar(ax, 0, len(models) - 1 - i, v, height, color)
        if m in marked:
            ax.text(v + top * 0.015, len(models) - 1 - i, "{:.3f}".format(v).lstrip("0"),
                    va="center", ha="left", fontsize=label_size, color=INK,
                    fontweight="bold")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(list(reversed(models)), fontsize=label_size, color=INK)
    bare(ax, "x")
    ticks = [t for t in (0, 0.2, 0.4, 0.6, 0.8) if t <= top * 1.05]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + ["{:.1f}".format(t).lstrip("0") for t in ticks[1:]])


def draw_gap(ax, data, label_size=8.5, value_labels=True):
    """Each open size class hanging below the frontier mean, per metric."""
    front = [m for m in data if data[m]["group"] == "Frontier-scale"]
    members = open_groups(data)
    classes = list(members)
    colors = dict(zip(classes, RAMP + [ACCENT]))
    avg = lambda ms, c: sum(data[m][c] for m in ms) / len(ms)
    keys = list(METRICS)

    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(-0.34, 0.012)
    width, _ = _px(ax, 13)
    gap, _ = _px(ax, 2)
    span = 4 * width + 3 * gap

    for i, k in enumerate(keys):
        worst = 0
        for j, c in enumerate(classes):
            v = avg(members[c], k) - avg(front, k)
            rounded_bar(ax, i - span / 2 + j * (width + gap), 0, width, v,
                        colors[c], horizontal=False)
            worst = min(worst, v)
        if value_labels:
            ax.text(i, worst - 0.012, "{:.2f}".format(worst), ha="center", va="top",
                    fontsize=7.5, color=INK_2)

    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([METRICS[k].replace("Overall creativity score", "Overall").replace(" ", "\n")
                        for k in keys], fontsize=label_size, color=INK)
    bare(ax, "y")
    ax.set_yticks([0, -0.1, -0.2, -0.3])
    ax.set_yticklabels(["0", "-.1", "-.2", "-.3"])
    ax.tick_params(axis="x", pad=4)
    leg = ax.legend(handles=[Patch(facecolor=colors[c], label=c) for c in classes],
                    loc="lower left", bbox_to_anchor=(0.0, 1.01), frameon=False,
                    fontsize=8, ncol=4, handlelength=0.9, handleheight=0.9,
                    columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK_2)


def draw_time(ax, data, dates, label_size=9):
    """Score against release date, one least-squares trend per dimension."""
    models = [m for m in data if m in dates]
    xs = np.array([dates[m].toordinal() for m in models], dtype=float)
    ends = []
    for key, label, color in TREND:
        ys = np.array([data[m][key] for m in models])
        ax.scatter(xs, ys, s=16, color=color, alpha=0.5, linewidths=0, zorder=2)
        b, a = np.polyfit(xs, ys, 1)
        grid = np.array([xs.min(), xs.max()])
        ax.plot(grid, a + b * grid, color=color, linewidth=2, solid_capstyle="round",
                zorder=3)
        ends.append((a + b * grid[-1], label))

    ends.sort()
    placed = []
    for y, label in ends:
        if placed and y - placed[-1] < 0.04:
            y = placed[-1] + 0.04
        placed.append(y)
        ax.text(xs.max() + 20, y, label, color=INK, fontsize=label_size,
                va="center", ha="left")

    ticks = [date(y, m, 1) for y, m in [(2024, 1), (2024, 7), (2025, 1)]]
    ax.set_xticks([t.toordinal() for t in ticks])
    ax.set_xticklabels([t.strftime("%b %Y") for t in ticks])
    ax.set_xlim(xs.min() - 30, xs.max() + 30)
    ax.set_ylim(0.2, 0.85)
    bare(ax, "y")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([".2", ".4", ".6", ".8"])


# ---------- figures ----------

def fig_ranking(data, metric):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.165, right=0.965, top=0.80, bottom=0.14)
    draw_ranking(ax, data, metric)
    group_legend(ax)
    name = METRICS[metric]
    titles(fig, name if metric == "overall" else "{}: 17 models".format(name),
           "CreativityPrism: 8 tasks, averaged over quality, novelty and diversity. Higher is better."
           if metric == "overall" else
           "CreativityPrism score on {}. Higher is better.".format(name.lower()),
           SOURCE)
    fig.savefig(OUT / "ranking_{}.png".format(metric), facecolor=SURFACE)
    plt.close(fig)


def fig_gap(data):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.19)
    draw_gap(ax, data)
    titles(fig, "How far open models sit behind the frontier",
           "Mean score of each open size class minus the frontier-scale mean. Zero is parity.",
           "Label marks the widest gap in each column  ·  arxiv.org/abs/2510.20091")
    fig.savefig(OUT / "gap_vs_frontier.png", facecolor=SURFACE)
    plt.close(fig)


def fig_time(data, dates):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.78, bottom=0.15)
    draw_time(ax, data, dates)
    titles(fig, "Newer models score higher",
           "Each model's score against its release date, with a least-squares trend per dimension.",
           "17 models; release dates from the paper's appendix  ·  arxiv.org/abs/2510.20091")
    fig.savefig(OUT / "score_over_time.png", facecolor=SURFACE)
    plt.close(fig)


def fig_panel_rankings(data, metrics, filename, title, subtitle):
    """Three rankings side by side: X allows four images, so these travel together.

    Each panel is sorted on its own metric -- the orders differ, which is the
    point -- so every panel carries its own model names.
    """
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    # Room for three sets of model names, three value labels, and the footer.
    fig.subplots_adjust(left=0.107, right=0.973, top=0.745, bottom=0.145, wspace=0.62)
    for ax, metric in zip(axes, metrics):
        draw_ranking(ax, data, metric, label_size=6.5, bar_px=13, compact=True)
        ax.set_title(METRICS[metric], fontsize=10, color=INK, fontweight="bold", pad=6)
    group_legend(fig, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2)
    titles(fig, title, subtitle, SOURCE)
    fig.savefig(OUT / filename, facecolor=SURFACE)
    plt.close(fig)


def fig_panel_gap_time(data, dates):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI,
                             gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.22})
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.055, right=0.90, top=0.70, bottom=0.17)
    draw_gap(axes[0], data, label_size=7, value_labels=False)
    axes[0].set_title("Gap to the frontier, by metric", fontsize=10, color=INK,
                      fontweight="bold", pad=26)
    draw_time(axes[1], data, dates, label_size=8)
    # Same pad as the left panel, whose title has to clear its legend: the two
    # titles then sit on one line instead of stepping down across the figure.
    axes[1].set_title("Score against release date", fontsize=10, color=INK,
                      fontweight="bold", pad=26)
    titles(fig, "Where the gap is, and where the field is heading",
           "Left: each open size class minus the frontier-scale mean, by metric. "
           "Right: every model's score against its release date.", SOURCE)
    fig.savefig(OUT / "panel_gap_time.png", facecolor=SURFACE)
    plt.close(fig)


def fig_panel_heatmaps():
    """The paper's two correlation figures, side by side.

    These are the paper's own renders, not redrawn: the correlation matrices
    behind them are not in this repository, and redrawing from the picture
    would be inventing numbers. They are dense, so the canvas is wider than the
    rest -- at 1600x900 the cell labels would be unreadable even zoomed.
    """
    import matplotlib.image as mpimg
    src = REPO / "docs" / "assets" / "img"
    left = mpimg.imread(src / "figure_4_heatmap_task_full.png")
    right = mpimg.imread(src / "figure_5_heatmap_task_sub.png")

    # Same height for both, so the pair reads as one figure.
    h = 900
    wl = h * left.shape[1] / left.shape[0]
    wr = h * right.shape[1] / right.shape[0]
    pad, top, bottom = 40, 150, 120
    W, H = wl + wr + 3 * pad, h + top + bottom

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    for img, x0, w, label, caption in (
            (left, pad, wl, "By task",
             "Metrics from the same task agree; across tasks and domains they do not."),
            (right, 2 * pad + wl, wr, "By dimension",
             "Quality and diversity hold together. Novelty does not.")):
        ax = fig.add_axes([x0 / W, bottom / H, w / W, h / H])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(label, fontsize=13, fontweight="bold", color=INK, pad=10, loc="left")
        ax.text(0, -0.035, caption, transform=ax.transAxes, fontsize=9,
                color=INK_2, va="top", ha="left")

    fig.text(pad / W, 1 - 22 / H, "What correlates with what",
             fontsize=18, fontweight="bold", color=INK, va="top")
    fig.text(pad / W, 1 - 62 / H,
             "Pearson correlation of 17 models' scores, over every pair of metrics in CreativityPrism.",
             fontsize=11, color=INK_2, va="top")
    fig.text(pad / W, 24 / H,
             "Figures 4 and 5 of the paper  ·  " + SOURCE, fontsize=8.5, color=INK_2, va="bottom")
    fig.savefig(OUT / "panel_heatmaps.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    use_inter()
    data = load_table()
    dates = load_release_dates()
    missing = [m for m in data if m not in dates]
    if missing:
        raise SystemExit("no release date for: {}".format(missing))

    for metric in METRICS:
        fig_ranking(data, metric)
    fig_gap(data)
    fig_time(data, dates)
    fig_panel_rankings(data, DOMAINS, "panel_domains.png",
                       "Creativity by domain",
                       "CreativityPrism scores, ranked within each domain. Higher is better.")
    fig_panel_rankings(data, DIMENSIONS, "panel_dimensions.png",
                       "Creativity by dimension",
                       "CreativityPrism scores, ranked within each dimension. Higher is better.")
    fig_panel_gap_time(data, dates)
    fig_panel_heatmaps()
    for f in sorted(OUT.glob("*.png")):
        print("{:26s} {:>5.0f} KB".format(f.name, f.stat().st_size / 1024))


if __name__ == "__main__":
    main()
