"""Publication style, saving, colors, fonts, and layout assertions."""

from __future__ import annotations

import logging
import os
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.legend import Legend

PLOTTING_ROOT = Path(__file__).resolve().parent
FONT_ROOT = PLOTTING_ROOT / "fonts"
FONTAWESOME_SVG_ROOT = PLOTTING_ROOT / "assets" / "fontawesome" / "svgs" / "solid"
SHARP_GEOMETRIC_ROOT = PLOTTING_ROOT / "assets" / "sharp_geometric"
PUB_DENSE_PDF_RASTER_DPI = 180
PUB_SUBTEXT_SIZE = 13

# TeX Gyre Pagella text with serif fallbacks and Computer Modern mathtext.
PAPER_FONT_PARAMS = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": [
        "TeX Gyre Pagella",
        "Palatino",
        "Palatino Linotype",
        "P052",
        "Computer Modern Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
}

PUB_RCPARAMS = {
    **PAPER_FONT_PARAMS,
    "axes.formatter.use_mathtext": True,
    "image.cmap": "viridis",
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#333333",
    "axes.titlesize": 17,
    "axes.labelsize": PUB_SUBTEXT_SIZE,
    "figure.titlesize": 19,
    "figure.titleweight": "normal",
    "xtick.labelsize": PUB_SUBTEXT_SIZE,
    "ytick.labelsize": PUB_SUBTEXT_SIZE,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "legend.fontsize": PUB_SUBTEXT_SIZE,
    "legend.title_fontsize": PUB_SUBTEXT_SIZE,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 0.92,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#d0d0d0",
    "legend.borderpad": 0.58,
    "legend.borderaxespad": 0.90,
    "legend.handlelength": 1.55,
    "legend.handleheight": 1.05,
    "legend.handletextpad": 0.60,
    "legend.labelspacing": 0.72,
    "legend.columnspacing": 1.25,
    "figure.figsize": (7.2, 7.2),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "axes.titlepad": 7,
    "grid.color": "#b8b8b8",
    "grid.linewidth": 0.65,
    "grid.alpha": 0.24,
}

MUTED_CYCLE_HEX = (
    "#8172B3",  # purple
    "#DD8452",  # orange
    "#937860",  # brown
    "#DDAA33",  # gold
    "#DA8BC3",  # pink
    "#4C72B0",  # blue
    "#55A868",  # green
    "#8C8C8C",  # gray
    "#3F8F8F",  # teal
)
MUTED_TEAL = MUTED_CYCLE_HEX[8]
MUTED_RGB01 = np.asarray([to_rgb(color) for color in MUTED_CYCLE_HEX], dtype=np.float32)


def muted_rgb01(count: int) -> np.ndarray:
    """Return the first ``count`` muted categorical colors (purple, orange, brown, ...)."""
    return MUTED_RGB01[:count].copy()


def fontawesome_icon_path(name: str) -> Path:
    """Return the bundled solid Font Awesome SVG for an icon."""
    icon_path = FONTAWESOME_SVG_ROOT / f"{name}.svg"
    if not icon_path.exists():
        raise FileNotFoundError(f"Font Awesome icon not found: {icon_path}")
    return icon_path


def sharp_geometric_icon_path(name: str) -> Path:
    """Return the path to a bundled geometric SVG; raise if it is absent."""
    icon_path = SHARP_GEOMETRIC_ROOT / f"{name}.svg"
    if not icon_path.exists():
        raise FileNotFoundError(f"Sharp geometric icon not found: {icon_path}")
    return icon_path


def save_pub_figure(fig, output_base: str | os.PathLike[str], **savefig_kwargs) -> list[Path]:
    """Save PDF and PNG and return their paths; preserve page size unless bbox_inches is supplied."""
    savefig_kwargs.setdefault("bbox_inches", None)
    savefig_kwargs.setdefault("pad_inches", 0.0)
    savefig_kwargs.setdefault("facecolor", "white")
    savefig_kwargs.setdefault("edgecolor", "none")
    base = Path(output_base)
    output_paths = [base.with_suffix(f".{fmt}") for fmt in ("pdf", "png")]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, **savefig_kwargs)
    return output_paths


def apply_pub_legend_alignment(legend):
    """Center legend titles while keeping legend entries left-aligned."""
    for text in legend.get_texts():
        text.set_horizontalalignment("left")
        text.set_multialignment("left")
    title = legend.get_title()
    if title is not None:
        title.set_horizontalalignment("center")
        title.set_multialignment("center")
    if hasattr(legend, "_legend_box"):
        legend._legend_box.align = "center"
        children = legend._legend_box.get_children()
        if len(children) >= 2 and hasattr(children[1], "align"):
            children[1].align = "left"
    legend.stale = True
    return legend


@cache
def prepare_pub_style():
    """Register bundled fonts, filter fontTools logs, and patch legend alignment once per process."""
    for logger_name in ("fontTools", "fontTools.subset", "fontTools.ttLib"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.getLogger("fontTools.ttLib.tables._h_e_a_d").addFilter(
        lambda record: "timestamp seems very low" not in record.getMessage()
    )
    for font_path in FONT_ROOT.glob("*.otf"):
        font_manager.fontManager.addfont(str(font_path))

    original_init = Legend.__init__
    original_draw = Legend.draw

    def aligned_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        apply_pub_legend_alignment(self)

    def aligned_draw(self, renderer):
        apply_pub_legend_alignment(self)
        return original_draw(self, renderer)

    Legend.__init__ = aligned_init
    Legend.draw = aligned_draw


def apply_pub_style():
    """Apply the Figure 5 style using bundled fonts and mathtext."""
    prepare_pub_style()
    plt.rcParams.update(PUB_RCPARAMS)


def assert_pub_axes_same_size_inches(fig, axes, *, tol: float = 0.015) -> None:
    """Raise if panel widths or heights differ by more than tol inches."""
    fig_w, fig_h = fig.get_size_inches()
    sizes = [(ax.get_position().width * fig_w, ax.get_position().height * fig_h) for ax in axes]
    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    if max(widths) - min(widths) > float(tol) or max(heights) - min(heights) > float(tol):
        detail = ", ".join(f"{w:.3f}x{h:.3f}" for w, h in sizes)
        raise AssertionError(f"Panel widths or heights differ by more than {tol} inches: {detail} inches")


def assert_pub_tight_bboxes_separated(fig, first_ax, second_ax, *, axis: str, min_gap_px: float = 6.0) -> None:
    """Raise if axes bounding boxes, including labels, have less than min_gap_px separation.

    ``first_ax`` is the left one for ``axis="x"`` and the lower one for ``axis="y"``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    first = first_ax.get_tightbbox(renderer)
    second = second_ax.get_tightbbox(renderer)
    gap_px = float(second.x0 - first.x1) if axis == "x" else float(second.y0 - first.y1)
    if gap_px < float(min_gap_px):
        raise AssertionError(
            f"{axis.upper()} layout overlap: {gap_px:.2f}px gap between "
            f"{first_ax.get_label() or first_ax!r} and {second_ax.get_label() or second_ax!r}."
        )


def _pub_artist_bbox(artist, renderer):
    if artist is None or not artist.get_visible():
        return None
    if hasattr(artist, "get_tightbbox"):
        box = artist.get_tightbbox(renderer)
    elif hasattr(artist, "get_window_extent"):
        box = artist.get_window_extent(renderer)
    else:
        return None
    if box is None:
        return None
    if not np.all(np.isfinite([box.x0, box.x1, box.y0, box.y1])):
        return None
    if float(box.width) <= 0.0 or float(box.height) <= 0.0:
        return None
    return box


def assert_pub_artists_inside_figure(fig, artists, *, min_margin_px: float = 1.0) -> None:
    """Raise if a visible artist's bounding box enters the page margin of min_margin_px pixels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    min_margin_px = float(min_margin_px)
    page = fig.bbox
    for artist in artists:
        box = _pub_artist_bbox(artist, renderer)
        if box is None:
            continue
        overflow = {
            "left": float(page.x0 + min_margin_px - box.x0),
            "right": float(box.x1 - (page.x1 - min_margin_px)),
            "bottom": float(page.y0 + min_margin_px - box.y0),
            "top": float(box.y1 - (page.y1 - min_margin_px)),
        }
        bad = {side: value for side, value in overflow.items() if value > 0.0}
        if bad:
            detail = ", ".join(f"{side}={value:.2f}px" for side, value in bad.items())
            raise AssertionError(
                f"Rendered artist spills outside figure page ({detail}): "
                f"{getattr(artist, 'get_label', lambda: repr(artist))()}."
            )


def _pub_visible_text_bboxes(texts, renderer) -> list:
    boxes = []
    for text in texts:
        if text is None or not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box is None:
            continue
        if not np.all(np.isfinite([box.x0, box.x1, box.y0, box.y1])):
            continue
        if float(box.width) <= 0.0 or float(box.height) <= 0.0:
            continue
        boxes.append(box)
    return boxes


def assert_pub_axis_tick_labels_clear(fig, ax, *, min_gap_px: float = 1.0) -> None:
    """Require min_gap_px pixels between adjacent tick labels and between ticks and axis labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    min_gap_px = float(min_gap_px)
    ax_name = ax.get_label() or repr(ax)
    tick_labels = {
        "x": [*ax.get_xticklabels(), *ax.get_xticklabels(minor=True)],
        "y": [*ax.get_yticklabels(), *ax.get_yticklabels(minor=True)],
    }
    for along, (name, axis) in enumerate((("x", ax.xaxis), ("y", ax.yaxis))):
        # along indexes the axis direction (x=0, y=1); across indexes its perpendicular.
        across = 1 - along
        tick_boxes = sorted(_pub_visible_text_bboxes(tick_labels[name], renderer), key=lambda box: box.p0[along])
        for lower_box, upper_box in zip(tick_boxes, tick_boxes[1:]):
            gap_px = float(upper_box.p0[along] - lower_box.p1[along])
            if gap_px < min_gap_px:
                raise AssertionError(
                    f"{name.upper()} tick labels overlap in {ax_name}: {gap_px:.2f}px gap, min_gap={min_gap_px:.2f}px."
                )
        if not (axis.label.get_visible() and axis.label.get_text()):
            continue
        label_box = axis.label.get_window_extent(renderer)
        for tick_box in tick_boxes:
            overlaps = min(tick_box.p1[along], label_box.p1[along]) > max(tick_box.p0[along], label_box.p0[along])
            gap_px = float(max(tick_box.p0[across] - label_box.p1[across], label_box.p0[across] - tick_box.p1[across]))
            if overlaps and gap_px < min_gap_px:
                raise AssertionError(
                    f"{name.upper()} tick labels collide with {name}-axis label in {ax_name}: "
                    f"{gap_px:.2f}px gap, min_gap={min_gap_px:.2f}px."
                )
