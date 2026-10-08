"""Rendering-only builder for the mutual-information collapse figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb, rgb_to_hsv
from matplotlib.patches import Circle, Rectangle

from plotting.figure_sizing import PANEL_LETTER_DY_IN, figure_sizes, panel_letter
from plotting.plot_utils import PLOT_THEME


def plot_mi_information_collapse(
    panel_results,
    panel_means,
    *,
    center,
    ring_radius,
    reward_direction,
):
    """Render Figure 6 from precomputed skill locations and information values."""
    results = list(panel_results)
    means = [np.asarray(value, dtype=float) for value in panel_means]
    center = np.asarray(center, dtype=float)
    reward_direction = np.asarray(reward_direction, dtype=float)
    if len(results) != 3 or len(means) != 3:
        raise ValueError("the canonical information-collapse figure has three panels")

    ink = PLOT_THEME["ink"]
    fig_w = 8.8
    panel_fraction = 0.247
    panel_inches = panel_fraction * fig_w
    font_sizes = figure_sizes(fig_w)
    top_margin_inches = (
        PANEL_LETTER_DY_IN + 0.7 * font_sizes["panel_letter"] / 72 + 0.10
    )
    bottom_margin_inches = 0.99
    fig_h = top_margin_inches + panel_inches + bottom_margin_inches
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": False,
            "axes.titlesize": font_sizes["title"],
            "axes.titleweight": "normal",
            "xtick.labelsize": font_sizes["tick"],
            "ytick.labelsize": font_sizes["tick"],
            "axes.labelsize": font_sizes["label"],
            "legend.fontsize": font_sizes["legend"],
        }
    )
    title_size = plt.rcParams["axes.titlesize"]
    subtitle_size = title_size
    contour_size = font_sizes["annotation"]
    letter_daylight_inches = PANEL_LETTER_DY_IN - 0.08 - title_size / 72
    assert letter_daylight_inches > 0.10, "Figure 6's panel letters need more than 0.10 inches of clearance above titles"

    view_half_width = 3.6
    cyclic_map = plt.get_cmap("twilight_shifted")
    skill_saturation = 1.45
    skill_value = 0.88
    contour_radii = [2.0, 1.6, 1.2, 0.8, 0.4]
    contour_alphas = [0.14, 0.20, 0.27, 0.35, 0.44]
    inset_radii = [1.7, 1.25, 0.85, 0.45]
    inset_alphas = [0.24, 0.36, 0.52, 0.72]
    blob_radius_sigma = max(contour_radii)
    ink_width, ink_height = 0.98, 0.78
    label_candidates = [-3, -2, -1, 0, 1, 2, 3]
    label_min_gap = 0.12
    inset_box = [0.41, 0.54, 0.59, 0.44]

    def format_width(value):
        if value >= 0.01:
            return f"{value:.2f}"
        mantissa, exponent = f"{value:.0e}".split("e")
        return f"{mantissa}\\cdot 10^{{{int(exponent)}}}"

    def skill_color(index, count):
        hue, saturation, value = rgb_to_hsv(np.array(cyclic_map(index / count)[:3]))
        return tuple(
            hsv_to_rgb(
                (
                    hue,
                    min(1.0, saturation * skill_saturation),
                    skill_value * value,
                )
            )
        )

    def draw_skills(ax, locations, sigma, *, inset=False):
        radii, alphas = (
            (inset_radii, inset_alphas)
            if inset
            else (contour_radii, contour_alphas)
        )
        for index, location in enumerate(locations):
            color = skill_color(index, len(locations))
            for radius_sigma, alpha in zip(radii, alphas):
                ax.add_patch(
                    Circle(
                        location,
                        radius_sigma * sigma,
                        fc=(*color, alpha),
                        ec="none",
                        zorder=3,
                    )
                )

    axis_ticks = np.linspace(-view_half_width, view_half_width, 201)
    grid_x, grid_y = np.meshgrid(axis_ticks, axis_ticks)
    reward_field = reward_direction[0] * grid_x + reward_direction[1] * grid_y

    def iso_format(value):
        return f"$r$={int(value)}"

    def label_clears_rect(points, extent_x, extent_y, half_tangent, half_normal, tangent, rect, pad):
        x0, y0, x1, y1 = rect
        rect_center = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        rect_half = np.array([0.5 * (x1 - x0), 0.5 * (y1 - y0)])
        clear = np.zeros(len(points), dtype=bool)
        for normal, label_radius in (
            (np.array([1.0, 0.0]), extent_x),
            (np.array([0.0, 1.0]), extent_y),
            (tangent, half_tangent),
            (reward_direction, half_normal),
        ):
            clear |= (
                np.abs(points @ normal - rect_center @ normal)
                > label_radius + rect_half @ np.abs(normal) + pad
            )
        return clear

    def plan_iso_labels(ax, blobs, blob_radius, keep_out=(), pad=0.10):
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inverse = ax.transData.inverted()
        tangent = np.array([-reward_direction[1], reward_direction[0]])
        rail_positions = np.linspace(-2.2 * view_half_width, 2.2 * view_half_width, 3001)
        points_by_level = {}
        gaps_by_level = {}
        half_sizes = {}
        for level in label_candidates:
            text = ax.text(0, 0, iso_format(level), fontsize=contour_size)
            bbox = text.get_window_extent(renderer=renderer)
            text.remove()
            (x0, y0), (x1, y1) = inverse.transform(
                [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
            )
            half_tangent = 0.5 * ink_width * abs(x1 - x0)
            half_normal = 0.5 * ink_height * abs(y1 - y0)
            extent_x = (
                half_tangent * abs(tangent[0])
                + half_normal * abs(reward_direction[0])
            )
            extent_y = (
                half_tangent * abs(tangent[1])
                + half_normal * abs(reward_direction[1])
            )
            points = level * reward_direction + rail_positions[:, None] * tangent
            valid = (
                (np.abs(points[:, 0]) <= view_half_width - extent_x - pad)
                & (np.abs(points[:, 1]) <= view_half_width - extent_y - pad)
            )
            displacement = blobs[None, :, :] - points[:, None, :]
            gap = (
                np.hypot(
                    np.maximum(np.abs(displacement @ tangent) - half_tangent, 0.0),
                    np.maximum(
                        np.abs(displacement @ reward_direction) - half_normal,
                        0.0,
                    ),
                ).min(axis=1)
                - blob_radius
            )
            for rect in keep_out:
                valid &= label_clears_rect(
                    points,
                    extent_x,
                    extent_y,
                    half_tangent,
                    half_normal,
                    tangent,
                    rect,
                    pad,
                )
            points_by_level[level] = points
            gaps_by_level[level] = np.where(valid, gap, -np.inf)
            half_sizes[level] = (half_tangent, half_normal)

        assert max(size[1] for size in half_sizes.values()) * 2 + pad < 1.0, (
            "Figure 6's contour labels, including padding, must fit between unit-spaced reward lines"
        )
        order = sorted(label_candidates)
        rail_step = rail_positions[1] - rail_positions[0]
        best = None
        best_key = None
        for slope in np.linspace(-1.0, 1.0, 201):
            shifts = np.rint(slope * np.asarray(order) / rail_step).astype(int)
            gap_matrix = []
            y_matrix = []
            base_indices = np.arange(len(rail_positions))
            for level, shift in zip(order, shifts):
                shifted = base_indices + shift
                index = np.clip(shifted, 0, len(rail_positions) - 1)
                outside = (shifted < 0) | (shifted >= len(rail_positions))
                gap_matrix.append(
                    np.where(outside, -np.inf, gaps_by_level[level][index])
                )
                y_matrix.append(points_by_level[level][index, 1])
            gap_matrix = np.stack(gap_matrix)
            y_matrix = np.stack(y_matrix)
            keep_matrix = gap_matrix >= label_min_gap
            minimum_count = best_key[0] if best_key else 1
            for column in np.flatnonzero(keep_matrix.sum(axis=0) >= minimum_count):
                selected = keep_matrix[:, column]
                selected_levels = [
                    level for level, keep in zip(order, selected) if keep
                ]
                key = (
                    int(selected.sum()),
                    max(selected_levels) - min(selected_levels)
                    == len(selected_levels) - 1,
                    round(float(gap_matrix[selected, column].min()), 2),
                    -abs(slope),
                    -float(y_matrix[selected, column].mean()),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = [
                        (
                            level,
                            tuple(
                                points_by_level[level][
                                    min(
                                        max(column + shift, 0),
                                        len(rail_positions) - 1,
                                    )
                                ]
                            ),
                            float(gap_matrix[index, column]),
                        )
                        for index, (level, shift) in enumerate(zip(order, shifts))
                        if selected[index]
                    ]
        if best is None:
            return []
        return [
            (level, (float(point[0]), float(point[1])), gap)
            for level, point, gap in best
        ]

    def draw_reward_lines(ax, positions=()):
        contours = ax.contour(
            grid_x,
            grid_y,
            reward_field,
            levels=range(-4, 5),
            colors="0.55",
            linewidths=0.8,
            linestyles="solid",
            alpha=0.55,
            zorder=1,
        )
        if positions:
            ax.clabel(
                contours,
                fmt=iso_format,
                fontsize=contour_size,
                colors="0.45",
                inline=True,
                inline_spacing=6,
                manual=[point for _, point, _ in positions],
            )

    fig = plt.figure(figsize=(fig_w, fig_h))
    left_fraction, right_fraction = 0.045, 0.995
    spacing_fraction = (
        (right_fraction - left_fraction) * fig_w / panel_inches - 3
    ) / 2
    grid = fig.add_gridspec(
        1,
        3,
        wspace=spacing_fraction,
        top=1 - top_margin_inches / fig_h,
        bottom=bottom_margin_inches / fig_h,
        left=left_fraction,
        right=right_fraction,
    )
    panel_axes = []
    texts_to_clear = []
    keep_out = [(-0.22, -0.22, 0.22, 0.22)]
    for index, (result, locations) in enumerate(zip(results, means)):
        scale = result["s"]
        sigma = result["sigma"]
        ax = fig.add_subplot(grid[0, index])
        panel_axes.append(ax)
        draw_skills(ax, locations, sigma)
        ax.plot(*center, marker="+", color=ink, ms=8, mew=1.6, zorder=4)
        if index == 0:
            texts_to_clear.append(
                ax.annotate(
                    "$s$",
                    center,
                    xytext=(7, -9),
                    textcoords="offset points",
                    fontsize=title_size,
                    color=ink,
                )
            )
        ax.set_xlim(-view_half_width, view_half_width)
        ax.set_ylim(-view_half_width, view_half_width)
        ax.set_aspect("equal")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        ax.grid(False)
        ax.set_title(
            f"$\\mathcal{{E}}(s) = {result['I']:.2f}$ nats",
            fontsize=title_size,
        )
        width_label = f"Width $\\approx {format_width(result['width'])}$"
        if index == 0:
            texts_to_clear.append(
                ax.text(
                    0.035,
                    0.960,
                    width_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color="0.35",
                    fontsize=subtitle_size,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.7, "pad": 1.5},
                )
            )
        else:
            inset_half_height = 2.15 * ring_radius * scale
            inset = ax.inset_axes(inset_box)
            inset.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=inset.transAxes,
                    fc="white",
                    ec="none",
                    zorder=0,
                )
            )
            # Lift the displayed ring above the two-line width label.
            inset_locations = center + 0.62 * (locations - center) + np.array(
                [0.0, 0.63 * inset_half_height]
            )
            draw_skills(inset, inset_locations, sigma, inset=True)
            inset_half_width = inset_half_height * inset_box[2] / inset_box[3]
            inset.set_xlim(center[0] - inset_half_width, center[0] + inset_half_width)
            inset.set_ylim(
                center[1] - inset_half_height,
                center[1] + inset_half_height,
            )
            inset.set_aspect("equal")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.grid(False)
            for spine in inset.spines.values():
                spine.set(color="0.35", lw=0.9, visible=True)
            inset.text(
                0.5,
                0.03,
                f"Width\n$\\approx {format_width(result['width'])}$",
                transform=inset.transAxes,
                ha="center",
                va="bottom",
                multialignment="center",
                color="0.35",
                fontsize=subtitle_size,
            )
            ax.indicate_inset_zoom(inset, edgecolor="0.35", lw=0.9, alpha=0.8)
            keep_out.append(
                (
                    view_half_width * (2 * inset_box[0] - 1),
                    view_half_width * (2 * inset_box[1] - 1),
                    view_half_width * (2 * (inset_box[0] + inset_box[2]) - 1),
                    view_half_width * (2 * (inset_box[1] + inset_box[3]) - 1),
                )
            )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in texts_to_clear:
        bbox = text.get_window_extent(renderer=renderer)
        (box_x0, box_y0), (box_x1, box_y1) = panel_axes[0].transData.inverted().transform(
            [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
        )
        keep_out.append((box_x0, box_y0, box_x1, box_y1))

    iso_positions = plan_iso_labels(
        panel_axes[0],
        means[0],
        blob_radius_sigma * results[0]["sigma"],
        keep_out=keep_out,
    )
    for ax in panel_axes:
        draw_reward_lines(ax, iso_positions)

    panel_position = panel_axes[0].get_position()
    drawn_width = panel_position.width * fig_w
    drawn_height = panel_position.height * fig_h
    assert abs(drawn_width - panel_inches) < 0.02, "Figure 6's panel width differs from the intended size by >= 0.02 inches"
    assert abs(drawn_height - panel_inches) < 0.02, "Figure 6's panel height differs from the intended size by >= 0.02 inches"
    for letter, ax in zip("abc", panel_axes):
        panel_letter(fig, ax, letter, font_sizes, color=ink)

    hue_map = ListedColormap(
        np.array([cyclic_map(value)[:3] for value in np.linspace(0, 1, 256)])
    )
    colorbar_ax = fig.add_axes(
        [
            0.405,
            0.470 / fig_h,
            0.19,
            0.112 / fig_h,
        ]
    )
    hue_mappable = ScalarMappable(cmap=hue_map, norm=Normalize(0, 2 * np.pi))
    hue_mappable.set_array([])
    colorbar = fig.colorbar(hue_mappable, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_label(
        "skill $z \\;\\to\\; p^{\\pi}_{\\gamma}(s_+ \\mid s,\\, z)$",
        fontsize=subtitle_size,
        labelpad=3,
    )
    colorbar.set_ticks([])
    colorbar.outline.set(edgecolor="0.35", linewidth=0.8)

    render_record = {
        "colors": {
            "cyclic_skills": "twilight_shifted",
            "ink_blend": skill_value,
            "contour_radii_sigma": contour_radii,
            "contour_fill_alpha_stacked": contour_alphas,
            "inset_radii_sigma": inset_radii,
            "inset_fill_alpha_stacked": inset_alphas,
        },
        "iso_labels": [
            {
                "level": level,
                "xy": [point[0], point[1]],
            }
            for level, point, _ in iso_positions
        ],
        "view_half_width": view_half_width,
        "typography": {
            "subtitle_and_legend_fontsize_pt": subtitle_size,
            "figure_size_in": [fig_w, fig_h],
            "panel_side_in": panel_inches,
        },
    }
    return fig, render_record
