"""Grid-figure style, action markers, grid axes, and colorbars."""

from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from plotting.plot_nodes import fontawesome_marker_path
from plotting.plot_style import PAPER_FONT_PARAMS, prepare_pub_style, save_pub_figure

PLOT_THEME = {
    "grid": "#c9d0dd",
    "spine": "#111111",
    "tick": "#111111",
    "ink": "#111111",
    "secondary": "#644ba5",
}

ACTION_ICON_NAMES = {"S": "pause", "P": "hand", "O": "minus"}
ACTION_ARROW_ANGLES = {"R": 0.0, "U": 90.0, "L": 180.0, "D": 270.0}


@cache
def sharp_arrow_marker_path(angle_deg: float = 0.0) -> MplPath:
    """Return a seven-vertex arrow rotated by angle_deg; at 0 degrees it points right and has length 1."""
    shaft_width, head_length, head_width = 0.26, 0.46, 0.80
    hs, hw, x_head = 0.5 * shaft_width, 0.5 * head_width, 0.5 - head_length
    vertices = np.array(
        [
            (-0.5, -hs),
            (x_head, -hs),
            (x_head, -hw),
            (0.5, 0.0),
            (x_head, hw),
            (x_head, hs),
            (-0.5, hs),
            (-0.5, -hs),
        ],
        dtype=float,
    )
    codes = np.array([MplPath.MOVETO] + [MplPath.LINETO] * 6 + [MplPath.CLOSEPOLY], dtype=np.uint8)
    return MplPath(Affine2D().rotate_deg(angle_deg).transform(vertices), codes)


def action_marker_path(action: str) -> MplPath:
    """Sharp arrows for moves, Font Awesome icons for stay, pick up, and drop."""
    if action in ACTION_ARROW_ANGLES:
        return sharp_arrow_marker_path(ACTION_ARROW_ANGLES[action])
    return fontawesome_marker_path(ACTION_ICON_NAMES.get(action, "minus"))


def draw_action_marker(ax, x: float, y: float, action: str, *, color, size: float = 180.0):
    ax.scatter(
        [x],
        [y],
        marker=action_marker_path(action),
        s=size,
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
        zorder=5.0,
    )
    return ax


def draw_square_grid(ax, rows: int, cols: int):
    """Draw the cell borders of a ``rows x cols`` grid."""
    style = {"colors": PLOT_THEME["grid"], "linewidth": 0.7, "alpha": 0.9, "zorder": 0.0, "clip_on": False}
    ax.vlines(np.arange(cols + 1, dtype=float) - 0.5, -0.5, rows - 0.5, **style)
    ax.hlines(np.arange(rows + 1, dtype=float) - 0.5, -0.5, cols - 0.5, **style)
    return ax


def style_grid_axes(ax, rows: int, cols: int):
    border_pad = 0.02
    ax.set_facecolor("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-0.5 - border_pad, cols - 0.5 + border_pad])
    ax.set_ylim([rows - 0.5 + border_pad, -0.5 - border_pad])
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def style_colorbar(
    cbar,
    *,
    label: str,
    labelpad: float,
    labelsize: float,
    tick_labelsize: float,
    frame_color: str,
    frame_lw: float,
    frame_alpha: float = 1.0,
    tick_guides: bool = True,
):
    """Frame a vertical colorbar; ``tick_guides`` replaces ticks by dotted white lines across the bar."""
    if getattr(cbar, "solids", None) is not None:
        cbar.solids.set_edgecolor("face")
    cbar.ax.set_facecolor("none")
    # Matplotlib suppresses the outline when frame_on is false.
    cbar.ax.set_frame_on(True)
    for name, spine in cbar.ax.spines.items():
        if name != "outline":
            spine.set_visible(False)
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor(frame_color)
    cbar.outline.set_linewidth(frame_lw)
    cbar.outline.set_alpha(frame_alpha)
    cbar.ax.tick_params(color=PLOT_THEME["tick"], labelcolor=PLOT_THEME["tick"], length=0.0, width=0.0)
    if tick_guides:
        # Locators may return ticks outside the color limits.
        bar_lo, bar_hi = cbar.mappable.get_clim()
        for tick in cbar.get_ticks():
            if not (bar_lo < tick < bar_hi):
                continue
            guide = cbar.ax.axhline(
                tick,
                xmin=0.10,
                xmax=0.90,
                color="white",
                linestyle=":",
                linewidth=0.9,
                alpha=0.95,
                zorder=10,
            )
            guide.set_dash_capstyle("butt")
    cbar.ax.yaxis.label.set_color(PLOT_THEME["tick"])
    cbar.ax.yaxis.label.set_rotation(270)
    cbar.ax.yaxis.label.set_verticalalignment("bottom")
    cbar.ax.yaxis.labelpad = labelpad
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.set_label(label, color=PLOT_THEME["tick"], rotation=270, labelpad=labelpad, fontsize=labelsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)
    return cbar


def init_style():
    """Apply the fixed style used by the grid and bound figures."""
    scale = 1.25
    prepare_pub_style()
    plt.rcParams.update(
        {
            "figure.figsize": (4.2, 3.4),
            "figure.dpi": 220,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.edgecolor": "none",
            "savefig.transparent": True,
            "axes.facecolor": "none",
            "axes.titlesize": 10.5 * scale,
            "axes.titleweight": "normal",
            "axes.labelsize": 9.2 * scale,
            "axes.edgecolor": PLOT_THEME["spine"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": PLOT_THEME["ink"],
            "axes.titlecolor": PLOT_THEME["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "xtick.labelsize": 8.0 * scale,
            "ytick.labelsize": 8.0 * scale,
            "xtick.color": PLOT_THEME["tick"],
            "ytick.color": PLOT_THEME["tick"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.fontsize": 8.0 * scale,
            "legend.frameon": False,
            "legend.facecolor": "none",
            "legend.labelcolor": PLOT_THEME["ink"],
            "grid.color": PLOT_THEME["grid"],
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "patch.edgecolor": "none",
            "patch.force_edgecolor": False,
            "image.cmap": "viridis",
            "image.interpolation": "nearest",
            **PAPER_FONT_PARAMS,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.formatter.use_mathtext": True,
        }
    )


def save_fig(name: str, fig, *, plot_dir: str | Path):
    """Save the grid figures as transparent PDF and PNG."""
    return save_pub_figure(fig, Path(plot_dir) / name, transparent=True, facecolor="none")
