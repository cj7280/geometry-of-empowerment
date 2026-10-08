"""Shared typography, panel-letter, and colorbar sizing."""

PAPER_TEXT_WIDTH_IN = 6.9
BASE_ON_PAGE_PT = 15.0

ON_PAGE_PT = {
    role: BASE_ON_PAGE_PT
    for role in ("panel_letter", "title", "label", "tick", "legend", "annotation")
}

PANEL_LETTER_DX_IN = 0.0
PANEL_LETTER_TITLE_GAP_IN = 0.12
PANEL_LETTER_DY_IN = 0.46
COLORBAR_LABEL_PAD_EM = 1.20


def figure_sizes(fig_width_in):
    """Scale ON_PAGE_PT by fig_width_in / PAPER_TEXT_WIDTH_IN to preserve printed font sizes."""
    scale = float(fig_width_in) / PAPER_TEXT_WIDTH_IN
    return {role: points * scale for role, points in ON_PAGE_PT.items()}


def colorbar_labelpad(sizes):
    return COLORBAR_LABEL_PAD_EM * sizes["label"]


def panel_letter(
    fig,
    ax,
    label,
    sizes,
    dx_in=PANEL_LETTER_DX_IN,
    dy_in=PANEL_LETTER_DY_IN,
    color=None,
    anchor="axes",
    title_gap_in=PANEL_LETTER_TITLE_GAP_IN,
):
    """Align a bold ``(label)`` with the panel’s left edge, above its axes or title."""
    fig.draw_without_rendering()  # Resolve equal-aspect panel positions before placing labels.
    figure_width, figure_height = fig.get_size_inches()
    text = f"({label})"
    kwargs = {
        "fontsize": sizes["panel_letter"],
        "fontweight": "bold",
        "ha": "left",
        "va": "baseline",
    }
    if color is not None:
        kwargs["color"] = color

    if anchor == "title":
        title_box = ax.title.get_window_extent().transformed(fig.transFigure.inverted())
        return fig.text(
            ax.get_position().x0 + dx_in / figure_width,
            title_box.y1 + title_gap_in / figure_height,
            text,
            **kwargs,
        )

    axes_box = ax.get_position()
    cap_height_in = 0.72 * sizes["panel_letter"] / 72.0
    y = axes_box.y1 + dy_in / figure_height
    y = min(y, 1.0 - cap_height_in / figure_height)
    return fig.text(axes_box.x0 + dx_in / figure_width, y, text, **kwargs)
