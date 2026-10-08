"""Render Figure 5: symmetric MDPs, Gaussian width, and KL level sets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path

from plotting.figure_sizing import figure_sizes, panel_letter
from plotting.plot_style import (
    assert_pub_artists_inside_figure,
    assert_pub_axes_same_size_inches,
    assert_pub_axis_tick_labels_clear,
    assert_pub_tight_bboxes_separated,
    muted_rgb01,
)
from plotting.plot_utils import PLOT_THEME
from plotting.width_mi_plots import (
    chart_ticks,
    filled_arrow,
    lane_perpendicular,
    occupancy_glyph,
    weight_ink_scale,
)

INK = PLOT_THEME["ink"]
TITLE_PAD_PT = 1.5
WINNER_FIELD_GRID = 500
WINNER_FIELD_MIX = 0.34


def assert_letters_clear_titles(fig, axes, gap_in=0.10):
    """Assert each panel letter is at least gap_in inches above its title's bounding box."""
    fig.draw_without_rendering()
    to_in = fig.dpi_scale_trans.inverted()
    letters = [t for t in fig.texts if t.get_text().startswith("(")]
    axes = list(axes)
    assert len(letters) == len(axes), f"Expected one letter per panel; found {len(letters)} letters for {len(axes)} panels"
    for ax, letter in zip(axes, letters):
        if not ax.get_title():
            continue
        title = ax.title.get_window_extent().transformed(to_in)
        box = letter.get_window_extent().transformed(to_in)
        clearance = box.y0 - title.y1
        assert clearance >= gap_in - 1e-6, (
            f"panel letter {letter.get_text()!r} sits {clearance:.3f} "
            f"in above its title, under the {gap_in} in minimum"
        )


def _absorbing_state_mdp(ax, family, colors, fs):
    """Draw start-to-absorbing transitions with probability >= 0.005; opacity encodes probability."""
    K = family.n_absorbing
    center = np.array([0.5, 0.48])
    ring = 0.315
    state_radius = 0.092
    start_radius = 0.064
    angles = [180.0, 0.0] if K == 2 else [90.0, 210.0, 330.0]
    states = [center + ring * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for a in angles]

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    skills = np.asarray(family.skill_distributions, float)
    lane_step = 0.035
    gap = 0.018
    drawn_by_absorbing = {j: [z for z in range(K) if skills[z, j] >= 5e-3] for j in range(K)}
    ink_of = weight_ink_scale([skills[z, j] for j, senders in drawn_by_absorbing.items() for z in senders])
    for j, (state, color) in enumerate(zip(states, colors[:K])):
        direction = (state - center) / np.linalg.norm(state - center)
        perpendicular = lane_perpendicular(direction)
        senders = drawn_by_absorbing[j]
        for lane, z in enumerate(senders):
            offset = (lane - (len(senders) - 1) / 2.0) * lane_step
            tail_out = float(np.sqrt(max(start_radius**2 - offset**2, 1e-6)))
            head_back = float(np.sqrt(max(state_radius**2 - offset**2, 1e-6)))
            line_width, alpha = ink_of(skills[z, j])
            ax.add_patch(
                FancyArrowPatch(
                    center + direction * (tail_out + gap) + perpendicular * offset,
                    state - direction * (head_back + gap) + perpendicular * offset,
                    color=colors[z],
                    alpha=alpha,
                    zorder=2,
                    **filled_arrow(tail_width=line_width),
                )
            )
        face = 0.17 * np.asarray(color) + 0.83 * np.ones(3)
        ax.add_patch(Circle(state, state_radius, fc=face, ec=color, lw=2.0, zorder=3))
        ax.text(*state, rf"$s_{{{j + 1}}}$", ha="center", va="center", fontsize=fs["legend"], zorder=4)
    ax.add_patch(Circle(center, start_radius, fc="white", ec=INK, lw=1.8, zorder=5))
    ax.text(*center, "$s$", ha="center", va="center", fontsize=fs["legend"], zorder=6)
    # Blank second line aligns all row titles.
    ax.set_title(rf"{K} Absorbing States" + "\n" + r"$\,$", fontsize=fs["title"])


def _occupancy_axes(ax, halfwidth, fs, *, title, show_xlabel):
    """Set equal-aspect u_1/u_2 axes with both limits [-halfwidth, halfwidth]."""
    ax.set_xlim(-halfwidth, halfwidth)
    ax.set_ylim(-halfwidth, halfwidth)
    ax.set_aspect("equal")
    ticks = chart_ticks(halfwidth)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=fs["legend"])
    ax.grid(False)
    ax.set_title(title, fontsize=fs["title"])
    if show_xlabel:
        ax.set_xlabel(r"Occupancy $u_1$", fontsize=fs["label"])
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(r"Occupancy $u_2$", fontsize=fs["label"])
    for spine in ax.spines.values():
        spine.set_color(PLOT_THEME["spine"])
        spine.set_linewidth(0.9)


def _length_underbrace(ax, first, second, *, outward, label, fs):
    """Draw a labeled curly brace parallel to the hull edge, offset in the outward direction."""
    brace_offset, brace_depth, label_gap = 0.13, 0.085, 0.30
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    edge = second - first
    edge_length = float(np.linalg.norm(edge))
    tangent = edge / edge_length
    normal = np.array([-tangent[1], tangent[0]])
    if np.dot(normal, outward) < 0.0:
        normal = -normal

    # Edge-relative cubic Bezier underbrace: (fraction along edge, depth multiple).
    profile = np.array(
        [
            [0.00, 0.00],
            [0.00, 0.55],
            [0.03, 1.00],
            [0.10, 1.00],
            [0.24, 1.00],
            [0.40, 1.00],
            [0.47, 1.05],
            [0.49, 1.08],
            [0.497, 1.30],
            [0.50, 1.35],
            [0.503, 1.30],
            [0.51, 1.08],
            [0.53, 1.05],
            [0.60, 1.00],
            [0.76, 1.00],
            [0.90, 1.00],
            [0.97, 1.00],
            [1.00, 0.55],
            [1.00, 0.00],
        ]
    )
    along = edge_length * profile[:, :1] * tangent[None, :]
    depth = (brace_offset + profile[:, 1:] * brace_depth) * normal[None, :]
    vertices = first[None, :] + along + depth
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(vertices) - 1)
    ax.add_patch(
        PathPatch(Path(vertices, codes), fill=False, color=INK, lw=1.35, capstyle="round", joinstyle="round", zorder=5)
    )
    angle = float(np.degrees(np.arctan2(tangent[1], tangent[0])))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    label_position = 0.5 * (first + second) + (brace_offset + 1.35 * brace_depth + label_gap) * normal
    ax.text(
        label_position[0],
        label_position[1],
        label,
        ha="center",
        va="center",
        rotation=angle,
        rotation_mode="anchor",
        fontsize=fs["legend"],
        color=INK,
        zorder=6,
    )


def _gaussian_width_panel(ax, geometry, colors, halfwidth, fs, *, show_xlabel):
    """Draw the skill hull U, with Gaussian width J = E_g[max_z <g,u_z>]."""
    coords = np.asarray(geometry["coords"], float)
    hull = coords[geometry["hull_order"]]
    _occupancy_axes(
        ax,
        halfwidth,
        fs,
        title=r"Gaussian Width $J$" + "\n"
        + rf"$\propto\;{geometry['perimeter_multiplier']}L={geometry['hull_perimeter']:.2f}$",
        show_xlabel=show_xlabel,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0.0)

    # Hue identifies argmax_z <g, u_z> over the displayed reward plane.
    field_axis = np.linspace(-halfwidth, halfwidth, WINNER_FIELD_GRID)
    field_xx, field_yy = np.meshgrid(field_axis, field_axis)
    field_winners = np.argmax(np.column_stack([field_xx.ravel(), field_yy.ravel()]) @ coords.T, axis=1)
    field_rgb = 1.0 + WINNER_FIELD_MIX * (np.asarray(colors, float)[field_winners] - 1.0)
    field_rgba = np.concatenate([field_rgb, np.ones((len(field_rgb), 1))], axis=1)
    ax.imshow(
        field_rgba.reshape(WINNER_FIELD_GRID, WINNER_FIELD_GRID, 4),
        extent=(-halfwidth, halfwidth, -halfwidth, halfwidth),
        origin="lower",
        interpolation="nearest",
        zorder=-1,
    )

    n_edges = 1 if len(hull) == 2 else len(hull)
    for edge in range(n_edges):
        first = hull[edge]
        second = hull[(edge + 1) % len(hull)]
        ax.plot([first[0], second[0]], [first[1], second[1]], color=INK, lw=2.0, solid_capstyle="round", zorder=2)

    if len(hull) == 2:
        first, second, outward = hull[0], hull[1], np.array([0.0, -1.0])
    else:
        # One side determines an equilateral triangle's perimeter; label the lowest.
        edge_index = int(np.argmin([0.5 * (hull[i, 1] + hull[(i + 1) % len(hull), 1]) for i in range(len(hull))]))
        first, second = hull[edge_index], hull[(edge_index + 1) % len(hull)]
        outward = 0.5 * (first + second) - hull.mean(axis=0)
    _length_underbrace(
        ax, first, second, outward=outward, label=rf"$L={float(np.linalg.norm(second - first)):.2f}$", fs=fs
    )

    for point, color in zip(coords, colors):
        ax.plot(point[0], point[1], "o", ms=10.5, mfc="white", mec=color, mew=2.6, zorder=7)
    ax.plot(0.0, 0.0, "o", ms=6.0, color=INK, zorder=7)
    return {
        "grid_size": WINNER_FIELD_GRID,
        "constant_color_mix": WINNER_FIELD_MIX,
    }


def _information_radius_panel(ax, family, geometry, colors, halfwidth, glyph_scale, fs, *, show_xlabel):
    """Draw the supplied D_KL(P || q) = I level set in u_1/u_2 coordinates."""
    coords = np.asarray(geometry["coords"], float)
    q = np.asarray(family.mixture, float)
    skills = np.asarray(family.skill_distributions, float)
    _occupancy_axes(
        ax,
        halfwidth,
        fs,
        title="Information Radius\n" + rf"$I={family.information_nats:.2f}$ nats",
        show_xlabel=show_xlabel,
    )

    if geometry["rank"] == 1:
        # For the symmetric two-state family, KL=I occurs at the two skill endpoints.
        radii = np.linalg.norm(coords, axis=1)
        direction = coords[np.argmax(radii)] / float(radii.max())
        perpendicular = np.array([-direction[1], direction[0]])
        tick = 0.085 * halfwidth
        for point in coords:
            endpoints = np.vstack([point - tick * perpendicular, point + tick * perpendicular])
            ax.plot(endpoints[:, 0], endpoints[:, 1], color=INK, lw=2.0, ls="solid", zorder=3)
    else:
        contour = np.vstack([geometry["kl_contour"], geometry["kl_contour"][0]])
        ax.plot(contour[:, 0], contour[:, 1], color=INK, lw=2.0, ls="solid", solid_joinstyle="round", zorder=3)

    for point, color in zip(coords, colors):
        ax.annotate(
            "",
            xy=point,
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=12.0),
            zorder=5,
        )
        ax.plot(point[0], point[1], "o", ms=10.5, mfc="white", mec=color, mew=2.6, zorder=6)
    ax.plot(0.0, 0.0, "o", ms=6.0, color=INK, zorder=7)

    # All probability bars use the [0,1] scale; colors identify absorbing states.
    glyph_width = 0.30 * glyph_scale
    glyph_height = 0.17 * glyph_scale
    glyph_offset = 0.24 * glyph_scale
    glyph_margin = 0.025 * glyph_scale
    glyph_nodes = [*coords, np.zeros(2)]
    glyph_distributions = [*skills, q]
    if geometry["rank"] == 1:
        glyph_directions = [np.array([0.0, 1.0]) for _ in coords] + [np.array([0.0, -1.0])]
    else:
        glyph_directions = [point / float(np.linalg.norm(point)) for point in coords] + [np.array([0.0, -1.0])]

    glyph_centers = []
    for glyph_index, (node, distribution, direction) in enumerate(
        zip(glyph_nodes, glyph_distributions, glyph_directions)
    ):
        glyph_center = np.asarray(node, float) + glyph_offset * direction
        glyph_center[0] = np.clip(
            glyph_center[0],
            -halfwidth + 0.5 * glyph_width + glyph_margin,
            halfwidth - 0.5 * glyph_width - glyph_margin,
        )
        glyph_center[1] = np.clip(
            glyph_center[1],
            -halfwidth + 0.5 * glyph_height + glyph_margin,
            halfwidth - 0.5 * glyph_height - glyph_margin,
        )
        leader = glyph_center - node
        leader_norm = float(np.linalg.norm(leader))
        # Connect q to its offset probability bars with a line.
        if glyph_index == len(coords) and leader_norm > 1e-12:
            leader_direction = leader / leader_norm
            distance_to_card = min(
                (0.5 * glyph_width) / max(abs(leader_direction[0]), 1e-12),
                (0.5 * glyph_height) / max(abs(leader_direction[1]), 1e-12),
            )
            card_edge = glyph_center - distance_to_card * leader_direction
            ax.plot(
                [node[0], card_edge[0]],
                [node[1], card_edge[1]],
                color=INK,
                lw=0.9,
                alpha=0.55,
                zorder=4,
                solid_capstyle="butt",
            )
        occupancy_glyph(ax, glyph_center, distribution, colors[: len(distribution)], glyph_width, glyph_height)
        glyph_centers.append(np.asarray(glyph_center, float).tolist())
    return {
        "centers": glyph_centers,
        "order": [*[f"skill_{index + 1}" for index in range(len(coords))], "q"],
        "shared_probability_vmax": 1.0,
        "glyph_width": float(glyph_width),
        "glyph_height": float(glyph_height),
    }


def plot_symmetric_adaptation_geometry(two_state, three_state):
    """Draw the MDP, Gaussian width, and information radius for both families.

    Each input pairs a family with its computed chart geometry.
    Return the figure and the data describing its marks and layout.
    """
    (x_family, x_geometry), (y_family, y_geometry) = two_state, three_state
    if (x_family.n_absorbing, y_family.n_absorbing) != (2, 3):
        raise ValueError("the canonical adaptation figure compares 2 and 3 absorbing states")

    fig_w, fig_h = 14.20, 11.30
    # Compute font sizes using a 13.40-inch reference width.
    fs = figure_sizes(13.40)
    colors = muted_rgb01(3)
    max_radius = max(float(np.linalg.norm(g["coords"], axis=1).max()) for g in (x_geometry, y_geometry))
    occupancy_halfwidth = 1.28 * max_radius

    fig = plt.figure(figsize=(fig_w, fig_h))
    side = 3.35
    col_lefts = [0.50, 5.35, 10.45]
    row_bottoms = [6.38, 1.12]

    def cell(column, row):
        return fig.add_axes([col_lefts[column] / fig_w, row_bottoms[row] / fig_h, side / fig_w, side / fig_h])

    ax_x_mdp = cell(0, 0)
    ax_y_mdp = cell(0, 1)
    _absorbing_state_mdp(ax_x_mdp, x_family, colors[:2], fs)
    _absorbing_state_mdp(ax_y_mdp, y_family, colors[:3], fs)

    ax_x_width = cell(1, 0)
    ax_y_width = cell(1, 1)
    x_winner_field = _gaussian_width_panel(ax_x_width, x_geometry, colors[:2], occupancy_halfwidth, fs, show_xlabel=False)
    y_winner_field = _gaussian_width_panel(ax_y_width, y_geometry, colors[:3], occupancy_halfwidth, fs, show_xlabel=True)

    ax_x_info = cell(2, 0)
    ax_y_info = cell(2, 1)
    x_glyphs = _information_radius_panel(
        ax_x_info, x_family, x_geometry, colors[:2], occupancy_halfwidth, max_radius, fs, show_xlabel=False
    )
    y_glyphs = _information_radius_panel(
        ax_y_info, y_family, y_geometry, colors[:3], occupancy_halfwidth, max_radius, fs, show_xlabel=True
    )

    # Check identical coordinate limits and physical sizes in panels b, c, e, and f.
    geometry_axes = (ax_x_width, ax_x_info, ax_y_width, ax_y_info)
    x_limits = np.asarray([axis.get_xlim() for axis in geometry_axes])
    y_limits = np.asarray([axis.get_ylim() for axis in geometry_axes])
    panel_sizes = np.asarray(
        [[axis.get_position().width * fig_w, axis.get_position().height * fig_h] for axis in geometry_axes]
    )
    assert np.allclose(x_limits, x_limits[0], atol=1e-12), "Figure 5's geometry panels must share the same x range"
    assert np.allclose(y_limits, y_limits[0], atol=1e-12), "Figure 5's geometry panels must share the same y range"
    assert np.allclose(panel_sizes, np.array([side, side]), atol=1e-12), (
        f"Figure 5's geometry panels must each measure {side} by {side} inches"
    )

    panel_axes = {"a": ax_x_mdp, "b": ax_x_width, "c": ax_x_info, "d": ax_y_mdp, "e": ax_y_width, "f": ax_y_info}
    for axis in panel_axes.values():
        if axis.get_title():
            # Use the same title padding for every panel.
            axis.set_title(axis.get_title(), fontsize=fs["title"], pad=TITLE_PAD_PT)
    for label, axis in panel_axes.items():
        panel_letter(fig, axis, label, fs, color=INK, anchor="title", title_gap_in=0.22)
    assert_letters_clear_titles(fig, panel_axes.values())

    # Check panel sizes, label spacing, and rendered bounding boxes.
    assert_pub_axes_same_size_inches(fig, panel_axes.values())
    for axis in geometry_axes:
        assert_pub_axis_tick_labels_clear(fig, axis, min_gap_px=2.0)
    for lower, upper in ((ax_y_mdp, ax_x_mdp), (ax_y_width, ax_x_width), (ax_y_info, ax_x_info)):
        assert_pub_tight_bboxes_separated(fig, lower, upper, axis="y", min_gap_px=18.0)
    for row_axes in ((ax_x_mdp, ax_x_width, ax_x_info), (ax_y_mdp, ax_y_width, ax_y_info)):
        for left, right in zip(row_axes, row_axes[1:]):
            assert_pub_tight_bboxes_separated(fig, left, right, axis="x", min_gap_px=10.0)
    page_text = [*fig.texts]
    for axis in fig.axes:
        page_text.extend(
            [axis.title, axis.xaxis.label, axis.yaxis.label, *axis.get_xticklabels(), *axis.get_yticklabels()]
        )
    assert_pub_artists_inside_figure(fig, [artist for artist in page_text if artist.get_text()], min_margin_px=2.0)

    render_record = {
        "shared_scale": {
            "xlim": x_limits[0].tolist(),
            "ylim": y_limits[0].tolist(),
            "geometry_panel_size_inches": panel_sizes[0].tolist(),
        },
        "mark_encodings": {
            "absorbing_state_and_skill_colors_rgb": [np.asarray(color, float).tolist() for color in colors[:3]],
        },
        "rows": [
            {"winner_field": x_winner_field, "information_panel_occupancy_glyphs": x_glyphs},
            {"winner_field": y_winner_field, "information_panel_occupancy_glyphs": y_glyphs},
        ],
        "layout_inches": {
            "figure": [fig_w, fig_h],
            "panel_side": side,
            "column_lefts": col_lefts,
            "row_bottoms": row_bottoms,
        },
    }
    return fig, render_record
