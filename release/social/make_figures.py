#!/usr/bin/env python3
"""Render the launch figures for X and LinkedIn.

    python release/social/make_figures.py

Everything is 1600x900 and built from the paper's own LaTeX -- the score table
and the model-details table -- so a change to the paper shows up here as a
changed figure rather than as a figure that quietly disagrees with it.

    ranking_overall.png        17 models, ranked
    ranking_<dimension>.png    quality / novelty / diversity
    ranking_<domain>.png       cw / dt / lr
    gap_vs_frontier.png        how far each open size class sits behind the frontier
    score_over_time.png        score against model release date

Type is Inter (fonts/, OFL). Colours come from the validated categorical
palette: two hues for the ranking charts, and for the gap chart a three-step
blue ramp, because size classes are an ordered quantity rather than unrelated
categories, plus one accent for the all-open average.
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
FRONTIER = "#2a78d6"   # categorical slot 1
OPEN = "#eb6834"       # categorical slot 2
RAMP = ["#9dc3f0", "#5a9ae3", "#1f5fa9"]   # ordered sizes: one hue, light -> dark
ACCENT = "#eb6834"

DPI = 200
FIGSIZE = (8, 4.5)     # -> 1600 x 900

METRICS = {
    "overall": "Overall creativity score",
    "quality": "Quality",
    "novelty": "Novelty",
    "diversity": "Diversity",
    "cw": "Creative writing",
    "dt": "Divergent thinking",
    "lr": "Logical reasoning",
}
SIZE_CLASSES = ["<10B", "10-40B", "40-80B"]


def use_inter():
    for ttf in sorted(FONTS.glob("Inter-*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    if any(FONTS.glob("Inter-*.ttf")):
        plt.rcParams["font.family"] = "Inter"
    plt.rcParams["axes.unicode_minus"] = False


def load_table():
    """Model -> scores + size class, read out of the paper's own table."""
    body = TABLE.read_text().split("\\begin{table}[h]")[1]
    cols = ["overall", "quality", "novelty", "diversity", "cw", "dt", "lr"]
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


def titles(fig, title, subtitle, note):
    fig.text(0.045, 0.95, title, fontsize=15.5, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.876, subtitle, fontsize=9.5, color=INK_2, va="top")
    fig.text(0.045, 0.04, note, fontsize=7.5, color=INK_2, va="bottom")


def fig_ranking(data, metric):
    models = sorted(data, key=lambda m: -data[m][metric])
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.165, right=0.965, top=0.80, bottom=0.14)

    top = max(data[m][metric] for m in models)
    ax.set_xlim(0, top * 1.16)
    ax.set_ylim(-0.7, len(models) - 0.3)
    _, height = _px(ax, 20)

    best_open = max((m for m in models if data[m]["group"] != "Frontier-scale"),
                    key=lambda m: data[m][metric])
    for i, m in enumerate(models):
        v = data[m][metric]
        color = FRONTIER if data[m]["group"] == "Frontier-scale" else OPEN
        rounded_bar(ax, 0, len(models) - 1 - i, v, height, color)
        if m in (models[0], best_open):
            ax.text(v + top * 0.012, len(models) - 1 - i, "{:.3f}".format(v).lstrip("0"),
                    va="center", ha="left", fontsize=8, color=INK, fontweight="bold")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(list(reversed(models)), fontsize=8, color=INK)
    bare(ax, "x")
    ticks = [t for t in (0, 0.2, 0.4, 0.6, 0.8) if t <= top * 1.16]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + ["{:.1f}".format(t).lstrip("0") for t in ticks[1:]])

    leg = ax.legend(handles=[Patch(facecolor=FRONTIER, label="Frontier-scale"),
                             Patch(facecolor=OPEN, label="Open")],
                    loc="lower right", frameon=False, fontsize=8,
                    handlelength=0.9, handleheight=0.9, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK_2)

    name = METRICS[metric]
    title = name if metric == "overall" else "{}: 17 models".format(name)
    sub = ("CreativityPrism: 8 tasks, averaged over quality, novelty and diversity. Higher is better."
           if metric == "overall" else
           "CreativityPrism score on {}. Higher is better.".format(name.lower()))
    titles(fig, title, sub, "arxiv.org/abs/2510.20091  ·  joeyhou.github.io/CreativityPrism")
    fig.savefig(OUT / "ranking_{}.png".format(metric), facecolor=SURFACE)
    plt.close(fig)


def fig_gap(data):
    """How far each open size class sits behind the frontier, per metric.

    The paper's Figure 3(b) hangs four bars downward per metric. Same data; the
    size classes ride one blue ramp because they are an ordered quantity, and
    the all-open average takes the accent hue because it is a summary rather
    than a fourth size.
    """
    front = [m for m in data if data[m]["group"] == "Frontier-scale"]
    avg = lambda ms, c: sum(data[m][c] for m in ms) / len(ms)
    classes = SIZE_CLASSES + ["Open (all)"]
    members = {c: [m for m in data if data[m]["group"] == c] for c in SIZE_CLASSES}
    members["Open (all)"] = [m for m in data if data[m]["group"] != "Frontier-scale"]
    colors = dict(zip(classes, RAMP + [ACCENT]))
    keys = list(METRICS)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.19)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(-0.34, 0.012)
    width, _ = _px(ax, 13)
    gap, _ = _px(ax, 2)
    span = 4 * width + 3 * gap

    for i, k in enumerate(keys):
        worst = 0
        for j, c in enumerate(classes):
            v = avg(members[c], k) - avg(front, k)      # negative: behind
            rounded_bar(ax, i - span / 2 + j * (width + gap), 0, width, v,
                        colors[c], horizontal=False)
            worst = min(worst, v)
        ax.text(i, worst - 0.012, "{:.2f}".format(worst), ha="center", va="top",
                fontsize=7.5, color=INK_2)

    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([METRICS[k].replace("Overall creativity score", "Overall").replace(" ", "\n")
                        for k in keys], fontsize=8.5, color=INK)
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
    titles(fig, "How far open models sit behind the frontier",
           "Mean score of each open size class minus the frontier-scale mean. Zero is parity.",
           "Label marks the widest gap in each column  ·  arxiv.org/abs/2510.20091")
    fig.savefig(OUT / "gap_vs_frontier.png", facecolor=SURFACE)
    plt.close(fig)


def fig_time(data, dates):
    """Score against release date, one trend line per dimension."""
    series = [("quality", "Quality", "#2a78d6"), ("novelty", "Novelty", "#eb6834"),
              ("diversity", "Diversity", "#1baf7a")]
    models = [m for m in data if m in dates]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.78, bottom=0.15)

    xs = np.array([dates[m].toordinal() for m in models], dtype=float)
    ends = []
    for key, label, color in series:
        ys = np.array([data[m][key] for m in models])
        ax.scatter(xs, ys, s=16, color=color, alpha=0.5, linewidths=0, zorder=2)
        b, a = np.polyfit(xs, ys, 1)
        grid = np.array([xs.min(), xs.max()])
        ax.plot(grid, a + b * grid, color=color, linewidth=2, solid_capstyle="round",
                zorder=3)
        ends.append((a + b * grid[-1], label))

    # Two trends can finish at the same height; stacked labels are unreadable,
    # so nudge them apart rather than letting them overprint.
    ends.sort()
    placed = []
    for y, label in ends:
        if placed and y - placed[-1] < 0.035:
            y = placed[-1] + 0.035
        placed.append(y)
        ax.text(xs.max() + 20, y, label, color=INK, fontsize=9,
                va="center", ha="left")

    ticks = [date(y, m, 1) for y, m in [(2024, 1), (2024, 7), (2025, 1), (2025, 7)]]
    ax.set_xticks([t.toordinal() for t in ticks])
    ax.set_xticklabels([t.strftime("%b %Y") for t in ticks])
    ax.set_xlim(xs.min() - 30, xs.max() + 30)
    ax.set_ylim(0.2, 0.85)
    bare(ax, "y")
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels([".2", ".4", ".6", ".8"])
    titles(fig, "Newer models score higher",
           "Each model's score against its release date, with a least-squares trend per dimension.",
           "17 models; release dates from the paper's appendix  ·  arxiv.org/abs/2510.20091")
    fig.savefig(OUT / "score_over_time.png", facecolor=SURFACE)
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
    for f in sorted(OUT.glob("*.png")):
        print("{:26s} {:>5.0f} KB".format(f.name, f.stat().st_size / 1024))


if __name__ == "__main__":
    main()
