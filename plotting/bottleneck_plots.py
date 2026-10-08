"""Render Figure 4's estimated capacity fields for two bottleneck grids."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from plotting.figure_palette import FIELD_EDGE
from plotting.plot_utils import style_colorbar, style_grid_axes

def plot_mi_heatmap_grid(configs, results_by_gamma):
    """Render one or both discount groups, each with two doorway locations."""
    if len(configs) != 2:
        raise ValueError("Figure 4 expects exactly two bottleneck configurations")
    panel, panel_gap, left = 2.68, 0.32, 0.22
    group_width, group_gap = 7.65, 0.25
    width, height = len(results_by_gamma) * group_width + (len(results_by_gamma) - 1) * group_gap, 3.96
    font_size = 22
    fig = plt.figure(figsize=(width, height))
    # Colors from the heatmaps embedded in the geom-for-arxiv figure PDFs.
    colors = {0.5: "#d55e00", 0.95: "#634ba5"}
    for group, (gamma, results) in enumerate(results_by_gamma.items()):
        offset = group * (group_width + group_gap)
        fields = [np.asarray(results[name]["I_s0"]) for name, _ in configs]
        lo, hi = min(v.min() for v in fields), max(v.max() for v in fields)
        cmap = LinearSegmentedColormap.from_list(f"gamma_{gamma}", ["white", colors[gamma]])
        center = offset + left + (2 * panel + panel_gap) / 2
        fig.text((offset + left) / width, 3.61 / height, "(a)" if gamma == 0.5 else "(b)",
                 fontsize=font_size, fontweight="bold")
        fig.text(center / width, 3.61 / height, rf"$\gamma = {gamma}$", ha="center", fontsize=font_size)
        for index, ((_, env), values) in enumerate(zip(configs, fields)):
            x = offset + left + index * (panel + panel_gap)
            ax = fig.add_axes([x / width, 0.18 / height, panel / width, panel / height])
            draw_mi_heatmap_panel(ax, env, values.reshape(env.rows, env.cols), vmin=lo, vmax=hi, cmap=cmap)
            fig.text((x + panel / 2) / width, 3.03 / height,
                     f"Avg. MI = {values.mean():.2f} nats", ha="center", fontsize=font_size)
        cax = fig.add_axes([(offset + 6.08) / width, 0.48 / height, 0.15 / width, 2.08 / height])
        bar = fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap=cmap), cax=cax)
        ticks = MaxNLocator(nbins=4, steps=[1, 2.5, 5, 10]).tick_values(lo, hi)
        bar.set_ticks([tick for tick in ticks if lo < tick < hi])
        style_colorbar(bar, label="nats", labelpad=24, labelsize=font_size, tick_labelsize=font_size,
                       frame_color=FIELD_EDGE, frame_lw=0.9, tick_guides=False)
        bar.ax.grid(False)
        bar.ax.yaxis.label.set_verticalalignment("center")
    return fig


def draw_walls(ax, walls):
    """Draw each blocked edge as a line between its two cells."""
    for (r1, c1), (r2, c2) in walls:
        if r1 == r2:
            # Horizontal neighbors share a vertical wall.
            x = max(c1, c2) - 0.5
            xs, ys = [x, x], [r1 - 0.5, r1 + 0.5]
        else:
            y = max(r1, r2) - 0.5
            xs, ys = [c1 - 0.5, c1 + 0.5], [y, y]
        ax.plot(
            xs,
            ys,
            color=FIELD_EDGE,
            lw=1.4,
            alpha=0.92,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            zorder=3.5,
            clip_on=False,
        )


def draw_mi_heatmap_panel(ax, env, mi_grid, *, vmin, vmax, cmap):
    """Draw the supplied MI field with grid walls overlaid."""
    image = ax.imshow(mi_grid, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    draw_walls(ax, env.walls)
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            env.cols,
            env.rows,
            fill=False,
            edgecolor=FIELD_EDGE,
            linewidth=0.9,
            alpha=0.85,
            zorder=3.4,
        )
    )
    style_grid_axes(ax, env.rows, env.cols)
    return image


def plot_contiguous_fields(configs, fields):
    """Two doorway configurations, with one shared contiguous-MI color scale."""
    fig = plt.figure(figsize=(7.65, 3.96))
    cmap = LinearSegmentedColormap.from_list("contiguous", ["white", "#634ba5"])
    lo, hi = min(v.min() for v in fields), max(v.max() for v in fields)
    for i, ((_, env), values) in enumerate(zip(configs, fields)):
        x = 0.22 + 3 * i
        ax = fig.add_axes([x / 7.65, 0.18 / 3.96, 2.68 / 7.65, 2.68 / 3.96])
        draw_mi_heatmap_panel(ax, env, values.reshape(env.rows, env.cols), vmin=lo, vmax=hi, cmap=cmap)
        fig.text(x / 7.65, 3.61 / 3.96, f"({chr(97 + i)})", fontsize=22, fontweight="bold")
        fig.text((x + 1.34) / 7.65, 3.03 / 3.96, f"Avg. MI = {values.mean():.2f} nats",
                 ha="center", fontsize=22)
    cax = fig.add_axes([6.08 / 7.65, 0.48 / 3.96, 0.15 / 7.65, 2.08 / 3.96])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap=cmap), cax=cax)
    style_colorbar(bar, label="nats", labelpad=24, labelsize=22, tick_labelsize=22,
                   frame_color=FIELD_EDGE, frame_lw=0.9, tick_guides=False)
    bar.ax.grid(False)
    return fig
