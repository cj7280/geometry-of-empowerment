"""Rendering-only builders for the KL-versus-Gaussian-width figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm, to_hex
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
    NullLocator,
)

from plotting.figure_palette import teal_opaque_cmap
from plotting.figure_sizing import figure_sizes
from plotting.plot_style import MUTED_TEAL, muted_rgb01
from plotting.plot_utils import PLOT_THEME, style_colorbar
from plotting.width_mi_plots import add_commitment_inlays


def _decimal_log_ticks(ax, axes="xy"):
    """Format major ticks with Python's general number format and hide minor labels."""
    formatter = FuncFormatter(lambda value, _position: f"{value:g}")
    for name in axes:
        axis = ax.xaxis if name == "x" else ax.yaxis
        axis.set_major_formatter(formatter)
        axis.set_minor_formatter(NullFormatter())


def plot_kl_width_quality(ts, width, lower_bound, *, inlay_ts):
    """Render figure 8 from an already-computed commitment sweep."""
    ts = np.asarray(ts, dtype=float)
    width = np.asarray(width, dtype=float)
    lower_bound = np.asarray(lower_bound, dtype=float)
    if ts.shape != width.shape or width.shape != lower_bound.shape:
        raise ValueError("ts, width, and lower_bound must have the same shape")

    ink = PLOT_THEME["ink"]
    fig_w, fig_h = 6.6, 4.85
    font_sizes = figure_sizes(fig_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.145, right=0.97, bottom=0.135, top=0.715)
    ax.fill_between(
        ts,
        lower_bound,
        width,
        color="0.6",
        alpha=0.22,
        lw=0,
        zorder=1,
        label="Gap",
    )
    ax.plot(
        ts,
        width,
        "o-",
        ms=9.5,
        lw=1.9,
        color=ink,
        mec="none",
        zorder=3,
        label=r"Adaptation $J$",
    )
    ax.plot(
        ts,
        lower_bound,
        "s-",
        ms=8.5,
        lw=1.7,
        color=MUTED_TEAL,
        mec="none",
        zorder=3,
        label=r"Lower Bound $I/(c\sqrt{2\pi})$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=float(width.max()) * 1.35)
    _decimal_log_ticks(ax)
    ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])

    add_commitment_inlays(
        ax,
        list(inlay_ts),
        [0.17, 0.50, 0.83],
        y_frac=1.02,
        size=0.32,
        fontsize=font_sizes["annotation"],
    )
    ax.set_xlabel(r"Commitment $t$", fontsize=font_sizes["label"])
    ax.set_ylabel("Return", fontsize=font_sizes["label"])
    legend = ax.legend(fontsize=font_sizes["legend"], loc="lower right", frameon=False)
    for label in legend.get_texts():
        text_x, text_y = label.get_position()
        label.set_position((text_x, text_y + 0.14 * font_sizes["legend"]))
    return fig


def plot_kl_width_vs_bound(information, width, state_counts, reference_slope):
    """Render Figure 10 for K absorbing targets, with reference curve J = I/reference_slope."""
    information = np.asarray(information, dtype=float)
    width = np.asarray(width, dtype=float)
    state_counts = np.asarray(state_counts, dtype=int)
    if information.shape != width.shape or width.shape != state_counts.shape:
        raise ValueError("information, width, and state_counts must have the same shape")

    ink = PLOT_THEME["ink"]
    categorical = muted_rgb01(3)
    fig_w, fig_h = 5.0, 3.40
    font_sizes = figure_sizes(fig_w)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.105, 0.150, 0.765, 0.815])
    colorbar_ax = fig.add_axes([0.895, 0.150, 0.026, 0.815])

    x_low = 0.5
    x_high = float(np.ceil(information.max() + 0.5))
    y_high = float(np.ceil(width.max() + 1.5))
    ax.set_xscale("log")
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(0.0, y_high)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    _decimal_log_ticks(ax, axes="x")
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: "" if value == 0 else f"{value:g}"))

    equality_color = to_hex(categorical[1])
    grid_x, grid_y = np.meshgrid(
        np.geomspace(x_low, x_high, 300),
        np.linspace(0.0, y_high, 300),
    )
    ax.contour(
        grid_x,
        grid_y,
        grid_y - grid_x / reference_slope,
        levels=[0.0],
        colors=[equality_color],
        linewidths=1.7,
        linestyles="--",
        zorder=2,
    )
    label_at = 2.45
    screen_low = ax.transData.transform(
        (0.96 * label_at, 0.96 * label_at / reference_slope)
    )
    screen_high = ax.transData.transform(
        (1.04 * label_at, 1.04 * label_at / reference_slope)
    )
    label_angle = np.degrees(
        np.arctan2(screen_high[1] - screen_low[1], screen_high[0] - screen_low[0])
    )
    ax.annotate(
        r"$J = I/(c\sqrt{2\pi})$",
        (label_at, label_at / reference_slope),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        rotation=label_angle,
        rotation_mode="anchor",
        color=equality_color,
        fontsize=font_sizes["annotation"],
    )

    norm_pad = 1.5
    color_map = ListedColormap(
        teal_opaque_cmap()(np.linspace(0.30, 1.0, 256)),
        name="K_opaque",
    )
    color_norm = LogNorm(
        vmin=float(state_counts.min()) / norm_pad,
        vmax=float(state_counts.max()) * norm_pad,
    )
    ax.plot(information, width, "-", color=ink, lw=1.1, solid_capstyle="round", zorder=2.5)
    ax.scatter(
        information,
        width,
        c=state_counts,
        cmap=color_map,
        norm=color_norm,
        s=85,
        edgecolors="none",
        zorder=3,
    )
    ax.tick_params(axis="both", which="major", labelsize=font_sizes["tick"])
    ax.set_xlabel(r"$I(Z; S_+ \mid S_0)$ (nats)", fontsize=font_sizes["label"])
    ax.set_ylabel(r"Adaptation $J$", fontsize=font_sizes["label"])

    colorbar = fig.colorbar(ScalarMappable(norm=color_norm, cmap=color_map), cax=colorbar_ax)
    colorbar.ax.yaxis.set_major_locator(FixedLocator(state_counts.tolist()))
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
    colorbar.ax.yaxis.set_minor_locator(NullLocator())
    style_colorbar(
        colorbar,
        label=r"Absorbing states $K$",
        labelsize=font_sizes["label"],
        labelpad=1.2 * font_sizes["label"],
        tick_labelsize=font_sizes["tick"],
        frame_color=PLOT_THEME["spine"],
        frame_lw=plt.rcParams["axes.linewidth"],
    )
    colorbar.ax.grid(False, which="both")
    return fig, {"x_limits": [x_low, x_high], "y_limits": [0.0, y_high]}
