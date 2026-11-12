#!/usr/bin/env python3
"""Render the launch figures for X and LinkedIn.

    python release/social/make_figures.py

Three 1600x900 PNGs, built from the paper's Table 1 rather than retyped:

    overall_ranking.png    17 models, ranked, frontier-scale against open
    gap_by_domain.png      where the frontier's lead actually is
    gap_by_dimension.png   and which dimension carries it

The numbers are parsed out of the paper's LaTeX table, so a change to the paper
shows up here as a changed figure rather than as a figure that quietly disagrees
with it. Colours are slots 1 and 2 of the validated categorical palette (blue,
orange); only two series are ever on screen at once.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "claude_docs" / "latex" / "sections" / "Tables" / "results.tex"
OUT = Path(__file__).resolve().parent

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
FRONTIER = "#2a78d6"   # categorical slot 1
OPEN = "#eb6834"       # categorical slot 2

DPI = 200
FIGSIZE = (8, 4.5)     # -> 1600 x 900


def load_table():
    """Model -> scores, read out of the paper's own table."""
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


def style(ax, xmax):
    """Recessive axes: hairline solid gridlines, no frame, ink-token text."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, solid_capstyle="butt")
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="both", length=0, labelsize=7.5, colors=INK_2)


def _px_to_data(ax, px):
    """(dx, dy) in data units for a length of `px` pixels on each axis.

    The two axes have different data scales, so one radius cannot serve both --
    using a single number is what made the first pass render square ends.
    """
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((px, px))
    return abs(x1 - x0), abs(y1 - y0)


def rounded_bar(ax, x0, y, width, height, color, round_px=4, horizontal=True):
    """A bar with a rounded data-end and a square baseline end.

    matplotlib has no such primitive, so the path is built by hand: the two
    corners at the data end get a quarter-circle, the baseline stays square.
    """
    rx, ry = _px_to_data(ax, round_px)
    if horizontal:
        r = min(rx, abs(width) / 2)
        ry = min(ry, height / 2)
        x1, y0, y1 = x0 + width, y - height / 2, y + height / 2
        verts = [(x0, y0), (x1 - r, y0), (x1, y0), (x1, y0 + ry),
                 (x1, y1 - ry), (x1, y1), (x1 - r, y1), (x0, y1), (x0, y0)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    else:
        r = min(ry, abs(height) / 2)
        rx = min(rx, width / 2)
        x1, y0, y1 = x0 + width, y, y + height
        verts = [(x0, y0), (x0, y1 - r), (x0, y1), (x0 + rx, y1),
                 (x1 - rx, y1), (x1, y1), (x1, y1 - r), (x1, y0), (x0, y0)]
        codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
                 MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none"))


def titles(fig, title, subtitle, note):
    fig.text(0.045, 0.945, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.875, subtitle, fontsize=9.5, color=INK_2, va="top")
    fig.text(0.045, 0.045, note, fontsize=7.5, color=INK_2, va="bottom")


def legend(ax, loc="lower right"):
    from matplotlib.patches import Patch
    leg = ax.legend(handles=[Patch(facecolor=FRONTIER, label="Frontier-scale"),
                             Patch(facecolor=OPEN, label="Open")],
                    loc=loc, frameon=False, fontsize=8, labelcolor=INK_2,
                    handlelength=0.9, handleheight=0.9, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK_2)


def fig_ranking(data):
    models = sorted(data, key=lambda m: -data[m]["overall"])
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.165, right=0.965, top=0.80, bottom=0.14)

    height = 0.56   # leaves the band's remainder as air; ~20px at this size
    best_open = max((m for m in models if data[m]["group"] != "Frontier-scale"),
                    key=lambda m: data[m]["overall"])
    for i, m in enumerate(models):
        v = data[m]["overall"]
        color = FRONTIER if data[m]["group"] == "Frontier-scale" else OPEN
        rounded_bar(ax, 0, len(models) - 1 - i, v, height, color)
        # Label only the two the story is about; the axis carries the rest.
        if m in (models[0], best_open):
            ax.text(v + 0.008, len(models) - 1 - i, "{:.3f}".format(v).lstrip("0"),
                    va="center", ha="left", fontsize=8, color=INK, fontweight="bold")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(list(reversed(models)), fontsize=8, color=INK)
    ax.set_ylim(-0.7, len(models) - 0.3)
    style(ax, 0.78)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0", ".2", ".4", ".6"])
    legend(ax)
    titles(fig, "Overall creativity score, 17 models",
           "CreativityPrism: 8 tasks, averaged over quality, novelty and diversity. Higher is better.",
           "arxiv.org/abs/2510.20091  ·  joeyhou.github.io/CreativityPrism")
    fig.savefig(OUT / "overall_ranking.png", facecolor=SURFACE)
    plt.close(fig)


def fig_groups(data, keys, labels, filename, title, subtitle):
    front = [m for m in data if data[m]["group"] == "Frontier-scale"]
    openm = [m for m in data if data[m]["group"] != "Frontier-scale"]
    avg = lambda ms, c: sum(data[m][c] for m in ms) / len(ms)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.965, top=0.80, bottom=0.16)

    # Limits first: the bar width is specified in pixels, and converting it to
    # data units needs the final transform.
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0, 0.92)
    width, _ = _px_to_data(ax, 24)       # mark spec: columns cap at 24px
    gap, _ = _px_to_data(ax, 2)          # the surface gap between adjacent bars

    for i, k in enumerate(keys):
        f, o = avg(front, k), avg(openm, k)
        rounded_bar(ax, i - width - gap / 2, 0, width, f, FRONTIER, horizontal=False)
        rounded_bar(ax, i + gap / 2, 0, width, o, OPEN, horizontal=False)
        # Each value sits on the far side of its own bar: a 24px column is
        # narrower than the label, so centring both puts them in each other's way.
        pad, _ = _px_to_data(ax, 3)
        ax.text(i - gap / 2 - width - pad, f + 0.012, "{:.2f}".format(f).lstrip("0"),
                ha="right", va="bottom", fontsize=8, color=INK)
        ax.text(i + gap / 2 + width + pad, o + 0.012, "{:.2f}".format(o).lstrip("0"),
                ha="left", va="bottom", fontsize=8, color=INK)
        # The gap is the point of the chart, so it rides above the pair rather
        # than below the axis, where it collided with the tick labels.
        ax.text(i, max(f, o) + 0.085, "gap {:+.2f}".format(f - o), ha="center",
                va="bottom", fontsize=9, color=INK, fontweight="bold")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=9.5, color=INK)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0", ".2", ".4", ".6", ".8"])
    ax.tick_params(axis="both", length=0, labelsize=7.5, colors=INK_2)
    ax.tick_params(axis="x", labelsize=9.5, pad=6)
    legend(ax, loc="lower right")
    titles(fig, title, subtitle,
           "Group means over 7 frontier-scale and 10 open models  ·  arxiv.org/abs/2510.20091")
    fig.savefig(OUT / filename, facecolor=SURFACE)
    plt.close(fig)


def main():
    data = load_table()
    fig_ranking(data)
    fig_groups(data, ["cw", "dt", "lr"],
               ["Creative writing", "Divergent thinking", "Logical reasoning"],
               "gap_by_domain.png",
               "The frontier's lead is not evenly spread",
               "Frontier-scale models pull ahead on writing and reasoning — and barely at all on divergent thinking.")
    fig_groups(data, ["quality", "novelty", "diversity"],
               ["Quality", "Novelty", "Diversity"],
               "gap_by_dimension.png",
               "Quality is where the frontier pulls away",
               "The same models, scored on the three creativity dimensions.")
    for f in sorted(OUT.glob("*.png")):
        print("{}  {:.0f} KB".format(f.name, f.stat().st_size / 1024))


if __name__ == "__main__":
    main()
