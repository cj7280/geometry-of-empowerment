"""Drawing primitives shared by the canonical width figures (Figures 5 and 9)."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from plotting.plot_style import MUTED_CYCLE_HEX
from plotting.plot_utils import PLOT_THEME
from plotting.plot_nodes import sharp_geometric_marker_path

# Transition probability controls opacity; arrow width is constant.
LW_FIXED = 2.40
ALPHA_MIN, ALPHA_MAX = 0.22, 1.0
ALPHA_FULL_P = 0.50
CARTOON_NODE_GAP = 0.026

MUTED_PURPLE, MUTED_ORANGE, MUTED_BROWN, MUTED_GOLD = MUTED_CYCLE_HEX[:4]
NODE_COLORS_4 = [MUTED_GOLD, MUTED_ORANGE, MUTED_PURPLE, MUTED_BROWN]


def chart_ticks(halfwidth):
    """Return symmetric ticks [-a, 0, a], or [0] when halfwidth < 0.01."""
    for step in (5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        count = int(np.floor(halfwidth / step + 1e-9))
        if count >= 1:
            return [-count * step, 0.0, count * step]
    return [0.0]


def filled_arrow(tail_width=LW_FIXED, head_w=2.90, head_l=3.10):
    """Return a single-path filled-arrow style with no shaft/head alpha seam."""
    arrowstyle = (
        f"Simple,tail_width={tail_width:.3f},head_width={head_w * tail_width:.3f},"
        f"head_length={head_l * tail_width:.3f}"
    )
    return {
        "arrowstyle": arrowstyle,
        "mutation_scale": 1.0,
        "linewidth": 0,
        "shrinkA": 0,
        "shrinkB": 0,
        "joinstyle": "miter",
    }


def weight_ink(probability):
    """Return fixed width and opacity increasing linearly up to probability ALPHA_FULL_P."""
    probability = float(np.clip(probability, 0.0, 1.0))
    weight = min(probability / ALPHA_FULL_P, 1.0)
    return LW_FIXED, ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * weight


def weight_ink_scale(values, floor=ALPHA_MIN):
    """Map min(values)..max(values) to floor..1 opacity; equal values use opacity 1."""
    drawn = np.asarray(list(values), dtype=float)
    low, high = float(drawn.min()), float(drawn.max())
    span = high - low

    def ink(probability):
        if span <= 1e-9:
            return LW_FIXED, ALPHA_MAX
        fraction = float(np.clip((float(probability) - low) / span, 0.0, 1.0))
        return LW_FIXED, floor + (ALPHA_MAX - floor) * fraction

    return ink


def occupancy_glyph(ax, center, weights, colors, width, height, *, edge="0.55", zorder=9):
    """Draw a frameless bar chart of probabilities on a shared [0, 1] scale, in data coordinates."""
    weights = np.asarray(weights, dtype=float)
    x0, y0 = center[0] - width / 2.0, center[1] - height / 2.0
    pad_x, pad_y = 0.11 * width, 0.13 * height
    inner_width = width - 2 * pad_x
    inner_height = height - 2 * pad_y
    slot = inner_width / len(weights)
    bar_width = 0.68 * slot
    for index, (weight, color) in enumerate(zip(weights, colors)):
        ax.add_patch(
            Rectangle(
                (x0 + pad_x + index * slot + (slot - bar_width) / 2.0, y0 + pad_y),
                bar_width,
                inner_height * float(np.clip(weight, 0.0, 1.0)),
                fc=color,
                ec="none",
                zorder=zorder + 1,
            )
        )
    ax.plot(
        [x0 + pad_x * 0.6, x0 + width - pad_x * 0.6],
        [y0 + pad_y] * 2,
        color=edge,
        lw=0.9,
        zorder=zorder + 1,
        solid_capstyle="butt",
    )


def lane_perpendicular(direction):
    """Rotate direction by 90 degrees, choosing positive y (or positive x when y=0)."""
    perpendicular = np.array([-direction[1], direction[0]], dtype=float)
    if perpendicular[1] < -1e-12 or (abs(perpendicular[1]) <= 1e-12 and perpendicular[0] < 0):
        perpendicular = -perpendicular
    return perpendicular


def _commitment_inlay(ax, commitment, *, annotate, annotate_fontsize):
    """Draw the small block-MDP used above the Figure 9 commitment sweep; block 0 is the focal one."""
    ink = PLOT_THEME["ink"]
    n_blocks = 4
    colors = [to_rgb(color) for color in NODE_COLORS_4]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    center = np.array([0.5, 0.5])
    ring_radius, node_radius, start_radius = 0.40, 0.095, 0.05
    points = [
        np.array(
            [
                0.5 + ring_radius * np.cos(np.radians(90 + 360 * index / n_blocks)),
                0.5 + ring_radius * np.sin(np.radians(90 + 360 * index / n_blocks)),
            ]
        )
        for index in range(n_blocks)
    ]

    other_mass = (1.0 - commitment) / n_blocks
    for index, point in enumerate(points):
        mass = commitment + other_mass if index == 0 else other_mass
        if mass < 5e-3:
            continue
        direction = (point - center) / np.hypot(*(point - center))
        _, alpha = weight_ink(mass)
        ax.add_patch(
            FancyArrowPatch(
                center + direction * (start_radius + CARTOON_NODE_GAP),
                point - direction * (node_radius + CARTOON_NODE_GAP),
                color=colors[0] if index == 0 else "#777777",
                alpha=alpha,
                zorder=4 if index == 0 else 2,
                **filled_arrow(),
            )
        )

    for index, point in enumerate(points):
        color = colors[index]
        face = tuple(np.clip(0.12 * np.asarray(color) + 0.88, 0, 1))
        ax.add_patch(Circle(point, node_radius, fc=face, ec=color, lw=1.2, zorder=5))
    ax.scatter(
        [points[0][0]],
        [points[0][1]],
        s=82,
        marker=sharp_geometric_marker_path("star"),
        facecolor=colors[0],
        edgecolor="none",
        zorder=6,
    )
    ax.add_patch(Circle(tuple(center), start_radius, fc="white", ec=ink, lw=1.0, zorder=6))

    if annotate:
        for target, label, position in (
            ((0.5, 0.72), rf"$t+\frac{{1-t}}{{{n_blocks}}}$", (0.66, 0.80)),
            ((0.5, 0.28), rf"$\frac{{1-t}}{{{n_blocks}}}$", (0.66, 0.20)),
        ):
            ax.annotate(
                label,
                xy=target,
                xytext=position,
                ha="left",
                va="center",
                fontsize=annotate_fontsize,
                color=ink,
                arrowprops={"arrowstyle": "-", "color": ink, "lw": 0.8, "shrinkA": 2.0, "shrinkB": 2.5},
                annotation_clip=False,
                zorder=7,
            )


def add_commitment_inlays(ax, t_values, x_fracs, *, y_frac, size, fontsize):
    """Add labeled block-MDP inlays above a commitment sweep; the first carries the mass labels."""
    for index, (commitment, x_fraction) in enumerate(zip(t_values, x_fracs)):
        inset = ax.inset_axes([x_fraction - size / 2, y_frac, size, size])
        _commitment_inlay(inset, float(commitment), annotate=index == 0, annotate_fontsize=fontsize * 0.92)
        ax.text(
            x_fraction,
            y_frac + size + 0.026,
            f"$t = {commitment:g}$",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=PLOT_THEME["ink"],
        )
