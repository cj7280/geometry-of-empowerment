"""Render grid policies, occupancy modes, and rollouts for Figures 2, 3, and 11."""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from plotting.figure_palette import teal_alpha_cmap, teal_opaque_cmap
from plotting.figure_sizing import colorbar_labelpad, figure_sizes, panel_letter
from plotting.plot_nodes import centered_unit_path, fontawesome_marker_path
from plotting.plot_style import MUTED_TEAL, muted_rgb01
from plotting.plot_utils import (
    PLOT_THEME,
    action_marker_path,
    draw_action_marker,
    draw_square_grid,
    style_colorbar,
    style_grid_axes,
)

INK = PLOT_THEME["ink"]
ACCENT = PLOT_THEME["secondary"]
ACTION_ORDER = ("R", "D", "L", "U", "S", "P", "O")


@cache
def key_icon_marker():
    """Font Awesome key tilted 38 degrees clockwise, renormalized to the unit box."""
    path = fontawesome_marker_path("key")
    return centered_unit_path(Affine2D().rotate_deg(-38.0).transform(path.vertices), path.codes)


def plot_keyed_grid(ax, env, *, key_marker_size: float = 300, key_linewidth: float = 0.45):
    """Plot the grid with shaded locked rooms and the key."""
    ax.set_facecolor("none")
    draw_square_grid(ax, env.rows, env.cols)
    for r, c in env.locked_cells:
        ax.add_patch(
            plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=ACCENT, alpha=0.14, edgecolor="none", zorder=0.1)
        )
    key_r, key_c = env.key_pos
    ax.scatter(
        [key_c],
        [key_r],
        marker=key_icon_marker(),
        s=key_marker_size,
        c=[ACCENT],
        edgecolor=PLOT_THEME["ink"],
        linewidth=key_linewidth,
        zorder=6.5,
    )
    return style_grid_axes(ax, env.rows, env.cols)


def _arrow_head_center_back_pt(lw, mutation_scale, head_length, head_width):
    """Return the distance in points from the path endpoint to the arrowhead's bounding-box center."""
    style = ArrowStyle("-|>", head_length=head_length, head_width=head_width)
    span = 1000.0
    probe = MplPath(np.array([[0.0, 0.0], [span, 0.0]]), [MplPath.MOVETO, MplPath.LINETO])
    paths, fillables = style(probe, mutation_size=mutation_scale, linewidth=lw)
    for path, fillable in zip(paths, fillables):
        if not fillable:
            continue
        verts = path.vertices
        if path.codes is not None:
            verts = verts[path.codes != MplPath.CLOSEPOLY]
        return span - 0.5 * (verts[:, 0].min() + verts[:, 0].max())
    raise RuntimeError("arrowstyle drew no filled head")


def action_colors() -> dict:
    """Muted color per action name; purple is reserved for skill arrows."""
    palette = muted_rgb01(len(ACTION_ORDER))
    colors = {name: tuple(float(v) for v in palette[i]) for i, name in enumerate(ACTION_ORDER)}
    colors["R"] = to_rgb(MUTED_TEAL)
    colors["S"] = INK
    return colors


def key_icon_legend(markersize: float):
    return Line2D(
        [0],
        [0],
        marker=key_icon_marker(),
        linestyle="None",
        color="w",
        markerfacecolor=ACCENT,
        markeredgecolor=INK,
        markeredgewidth=0.45,
        markersize=markersize,
        label="Key",
    )


def setup_grid_panel(ax, env):
    draw_square_grid(ax, env.rows, env.cols)
    style_grid_axes(ax, env.rows, env.cols)
    return ax


def plot_policy_icons(ax, env, pi_sa, *, size: float):
    """Draw the greedy action at each state."""
    setup_grid_panel(ax, env)
    colors = action_colors()
    for s, action_index in enumerate(np.argmax(np.asarray(pi_sa), axis=1)):
        r, c = env.to_rc(s)
        action = env.actions[int(action_index)]
        draw_action_marker(ax, c, r, action, color=colors[action], size=size)
    return ax


def draw_mi_field_panel(ax, env, I_grid):
    """Draw the MI field with the shared single-hue ramp inside a thin border."""
    image = ax.imshow(I_grid, origin="upper", cmap=teal_alpha_cmap())
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            env.cols,
            env.rows,
            fill=False,
            edgecolor=PLOT_THEME["grid"],
            linewidth=0.7,
            alpha=0.9,
            zorder=3.0,
        )
    )
    style_grid_axes(ax, env.rows, env.cols)
    return image


def skill_endpoint_legend_handles(color, sizes):
    """Return legend handles for skill occupancy modes and the common start state."""
    marker_pt = 0.72 * sizes["legend"]
    return [
        Line2D(
            [0],
            [0],
            marker=action_marker_path("R"),
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=marker_pt,
            label="Skill Endpoint",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=1.2,
            markersize=0.62 * marker_pt,
            label="Start",
        ),
    ]


def draw_skill_direction_panel(ax, env, s0, end_states, sizes):
    """Draw arrows from s0 to distinct supplied occupancy modes, excluding s0 itself."""
    setup_grid_panel(ax, env)
    r0, c0 = env.to_rc(s0)
    cells = []
    for end_state in end_states:
        end_r, end_c = env.to_rc(end_state)
        cell = (float(end_c), float(end_r))
        if cell == (float(c0), float(r0)) or cell in cells:
            continue
        cells.append(cell)

    lw = 1.7
    mutation_scale = 7.5 * lw
    head_length, head_width = 0.4, 0.2
    # Extend the path so the rendered head is centered on the endpoint.
    head_back_pt = _arrow_head_center_back_pt(lw, mutation_scale, head_length, head_width)
    origin_px = ax.transData.transform((0.0, 0.0))
    unit_px = ax.transData.transform((1.0, 0.0))
    data_per_pt = (ax.figure.dpi / 72.0) / abs(unit_px[0] - origin_px[0])
    head_back = head_back_pt * data_per_pt
    for x1, y1 in cells:
        step = np.array([x1 - c0, y1 - r0], dtype=float)
        step /= float(np.hypot(*step))
        ax.add_patch(
            FancyArrowPatch(
                (c0, r0),
                (x1 + step[0] * head_back, y1 + step[1] * head_back),
                arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
                mutation_scale=mutation_scale,
                lw=lw,
                color=ACCENT,
                shrinkA=3.0,
                shrinkB=0.0,
                joinstyle="miter",
                capstyle="butt",
                zorder=6,
            )
        )
    ax.scatter([c0], [r0], marker="o", s=46, facecolor="white", edgecolor=INK, linewidth=1.3, zorder=7)
    ax.legend(
        handles=skill_endpoint_legend_handles(ACCENT, sizes),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=False,
        fontsize=sizes["legend"],
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=0.75,
        borderpad=0.42,
        borderaxespad=0.0,
    )
    return ax


def plot_simple_grid_overview(env, I_grid, s0, end_states, pi_pot):
    """Figure 2: MI field | skill endpoints from ``s0`` | potential policy."""
    panel_size = 2.05
    cbar_w = 0.13
    cbar_h = 0.78 * panel_size
    gap_heat_cbar = 0.07
    gap_cbar_skill = 1.25  # Inches between the colorbar and skill panel; includes label space.
    gap_skill_pot = 0.74  # Inches between panels; includes space for the skill legend.
    margin_l = 0.26
    margin_r = 0.26
    margin_b = 0.62
    margin_t = 0.06
    header_gap = 0.05
    header_h = 0.62
    title_baseline_in = 0.07
    fig_w = (
        margin_l
        + panel_size
        + gap_heat_cbar
        + cbar_w
        + gap_cbar_skill
        + panel_size
        + gap_skill_pot
        + panel_size
        + margin_r
    )
    fig_h = margin_b + panel_size + header_gap + header_h + margin_t
    fs = figure_sizes(fig_w)
    fig = plt.figure(figsize=(fig_w, fig_h))

    panel_y = margin_b
    heat_x = margin_l
    cbar_x = heat_x + panel_size + gap_heat_cbar
    cbar_y = panel_y + 0.5 * (panel_size - cbar_h)
    header_y = panel_y + panel_size + header_gap
    skill_x = cbar_x + cbar_w + gap_cbar_skill
    pot_x = skill_x + panel_size + gap_skill_pot

    ax_heat = fig.add_axes([heat_x / fig_w, panel_y / fig_h, panel_size / fig_w, panel_size / fig_h])
    cax = fig.add_axes([cbar_x / fig_w, cbar_y / fig_h, cbar_w / fig_w, cbar_h / fig_h])
    ax_skill = fig.add_axes([skill_x / fig_w, panel_y / fig_h, panel_size / fig_w, panel_size / fig_h])
    ax_pot = fig.add_axes([pot_x / fig_w, panel_y / fig_h, panel_size / fig_w, panel_size / fig_h])

    for title_x, title_text in [
        (heat_x, r"Empowerment $\mathcal{E}(s)$"),
        (skill_x, "Skill endpoints"),
        (pot_x, r"Policy $\pi_{\mathrm{pot}}$"),
    ]:
        ax_header = fig.add_axes([title_x / fig_w, header_y / fig_h, panel_size / fig_w, header_h / fig_h])
        ax_header.set_axis_off()
        ax_header.text(
            0.5,
            (title_baseline_in - header_gap) / header_h,
            title_text,
            transform=ax_header.transAxes,
            ha="center",
            va="baseline",
            fontsize=fs["title"],
            color=INK,
        )

    im = draw_mi_field_panel(ax_heat, env, I_grid)
    # Precomposite the translucent ramp to prevent PDF seams.
    bar_sm = ScalarMappable(norm=Normalize(*im.get_clim()), cmap=teal_opaque_cmap())
    cbar = fig.colorbar(bar_sm, cax=cax)
    style_colorbar(
        cbar,
        label="nats",
        labelpad=colorbar_labelpad(fs),
        labelsize=fs["label"],
        tick_labelsize=fs["tick"],
        frame_color=PLOT_THEME["grid"],
        frame_lw=0.7,
        frame_alpha=0.9,
    )
    draw_skill_direction_panel(ax_skill, env, s0, end_states, fs)
    plot_policy_icons(ax_pot, env, pi_pot, size=160)
    for lab, panel_ax in zip("abc", [ax_heat, ax_skill, ax_pot]):
        panel_letter(fig, panel_ax, lab, fs, color=INK)
    return fig


def plot_rollout_panel(ax, env, traj_pot, traj_skills, *, mark_scale: float = 1.0):
    """Draw potential and effective-policy rollouts on the keyed grid."""
    area = float(mark_scale) ** 2
    plot_keyed_grid(ax, env, key_marker_size=300 * area)

    for idx in range(len(traj_pot) - 1):
        _, key_cur = env.decode_state(traj_pot[idx])
        r1, c1 = env.to_rc(traj_pot[idx])
        r2, c2 = env.to_rc(traj_pot[idx + 1])
        line_color = ACCENT if key_cur else INK
        ax.plot([c1, c2], [r1, r2], color=line_color, ls="--", alpha=0.78, lw=1.28 * mark_scale, zorder=5)

    for traj_skill in traj_skills:
        for idx in range(len(traj_skill) - 1):
            r1, c1 = env.to_rc(traj_skill[idx])
            r2, c2 = env.to_rc(traj_skill[idx + 1])
            ax.plot([c1, c2], [r1, r2], color=ACCENT, lw=1.48 * mark_scale, alpha=0.76, zorder=6)

    r0, c0 = env.to_rc(traj_pot[0])
    ax.scatter([c0], [r0], marker="o", s=86 * area, facecolor="white", edgecolor=INK, lw=1.5 * mark_scale, zorder=7)
    rs, cs = env.to_rc(traj_pot[-1])
    ax.scatter([cs], [rs], marker="D", s=74 * area, facecolor=INK, edgecolor="white", lw=0.5 * mark_scale, zorder=8)

    for traj_skill in traj_skills:
        _, has_key = env.decode_state(traj_skill[-1])
        rf, cf = env.to_rc(traj_skill[-1])
        ax.scatter(
            [cf],
            [rf],
            marker="*",
            s=150 * area,
            facecolor=ACCENT if has_key else INK,
            edgecolor=INK,
            lw=0.5 * mark_scale,
            alpha=0.82,
            zorder=9,
        )
    return ax


def _key_rollout_handles(*, overview: bool):
    def marker_handle(marker, face, edge, size, label):
        return Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor=face,
            markeredgecolor=edge,
            markersize=size,
            label=label,
            linestyle="None",
        )

    handles = []
    if not overview:
        handles += [
            marker_handle("o", "white", INK, 12.2, "Start"),
            marker_handle("D", INK, "white", 10.0, "Transition"),
        ]
    handles += [
        marker_handle("*", INK, INK, 16.5, "End (no key)"),
        marker_handle("*", ACCENT, INK, 16.5, "End (key)"),
        key_icon_legend(markersize=22.0 if overview else 14.5),
    ]
    if overview:
        handles += [
            Line2D([0], [0], color=INK, ls="--", lw=2.0, label=r"$\pi_{\mathrm{pot}}$ without key"),
            Line2D([0], [0], color=ACCENT, ls="--", lw=2.0, label=r"$\pi_{\mathrm{pot}}$ with key"),
            Line2D([0], [0], color=ACCENT, ls="-", lw=2.0, label=r"$\pi_{\mathrm{eff}}$"),
        ]
    else:
        handles += [
            Line2D([0], [0], color=INK, ls="--", lw=1.9, label=r"$\pi_{\mathrm{pot}}$"),
            Line2D([0], [0], color=ACCENT, ls="-", lw=2.0, label=r"$\pi_{\mathrm{eff}}$"),
        ]
    return handles


def plot_skill_rollouts(env, traj_pot, skill_trajectories):
    """Figure 11: one panel per skill, showing the shared potential-policy prefix and sampled continuations."""
    panels = list(skill_trajectories.values())
    max_per_row = 4
    n_cols = max(1, min(max_per_row, len(panels)))
    n_rows = max(1, (len(panels) + max_per_row - 1) // max_per_row)
    panel_size = 1.55
    panel_gap = 0.08
    title_h = 0.38
    legend_h = 0.76
    legend_gap = 0.13
    margin_l = 0.26
    margin_r = 0.26
    margin_t = 0.08
    margin_b = 0.12
    grid_w = n_cols * panel_size + (n_cols - 1) * panel_gap
    grid_h = n_rows * panel_size + (n_rows - 1) * panel_gap
    fig_w = margin_l + grid_w + margin_r
    fig_h = margin_b + legend_h + legend_gap + grid_h + title_h + margin_t
    fs = figure_sizes(fig_w)
    fig = plt.figure(figsize=(fig_w, fig_h))

    title_y = margin_b + legend_h + legend_gap + grid_h
    ax_title = fig.add_axes([margin_l / fig_w, title_y / fig_h, grid_w / fig_w, title_h / fig_h])
    ax_title.axis("off")
    ax_title.text(
        0.5,
        0.42,
        r"Skill rollouts $\pi_{\mathrm{pot}}$ (dashed) $\rightarrow$ $\pi_{\mathrm{eff}}$ (solid)",
        ha="center",
        va="center",
        fontsize=fs["title"],
        color=INK,
    )
    grid_bottom = margin_b + legend_h + legend_gap
    axes = []
    for row in range(n_rows):
        y = grid_bottom + (n_rows - 1 - row) * (panel_size + panel_gap)
        for col in range(n_cols):
            x = margin_l + col * (panel_size + panel_gap)
            axes.append(fig.add_axes([x / fig_w, y / fig_h, panel_size / fig_w, panel_size / fig_h]))

    ax_leg = fig.add_axes([margin_l / fig_w, margin_b / fig_h, grid_w / fig_w, legend_h / fig_h])
    ax_leg.axis("off")

    for ax, traj_skills in zip(axes, panels):
        plot_rollout_panel(ax, env, traj_pot, traj_skills)
        ax.set_anchor("C")
    for ax in axes[len(panels) :]:
        ax.axis("off")

    ax_leg.legend(
        handles=_key_rollout_handles(overview=False),
        loc="center",
        ncol=4,
        fontsize=fs["legend"],
        handlelength=2.15,
        handletextpad=0.82,
        columnspacing=1.78,
        borderaxespad=0.0,
        frameon=False,
    )
    return fig


def trace_rollout_events(env, traj):
    """Return (pickup_cells, drop_cells), each a list of (row, column) pairs."""
    pickup_positions = []
    drop_positions = []
    for i in range(len(traj) - 1):
        _, key_cur = env.decode_state(traj[i])
        _, key_next = env.decode_state(traj[i + 1])
        r1, c1 = env.to_rc(traj[i])
        if key_next and not key_cur:
            pickup_positions.append((r1, c1))
        if key_cur and not key_next:
            drop_positions.append((r1, c1))
    return pickup_positions, drop_positions


def draw_potential_rollout(ax, env, traj, *, sizes, mark_scale: float = 1.0):
    """Draw the potential-policy trajectory and key pickup/drop events.

    mark_scale scales lengths; marker areas scale by its square.
    """
    area = float(mark_scale) ** 2
    plot_keyed_grid(ax, env, key_marker_size=900 * area, key_linewidth=0.55 * mark_scale)
    pickup_positions, drop_positions = trace_rollout_events(env, traj)

    for i in range(len(traj) - 1):
        _, key_cur = env.decode_state(traj[i])
        r1, c1 = env.to_rc(traj[i])
        r2, c2 = env.to_rc(traj[i + 1])
        line_color = ACCENT if key_cur else INK
        ax.plot([c1, c2], [r1, r2], color=line_color, ls="--", lw=2.4 * mark_scale, alpha=0.86, zorder=5)

    annot_size = sizes["annotation"]
    r0, c0 = env.to_rc(traj[0])
    ax.scatter([c0], [r0], marker="o", s=136 * area, facecolor="white", edgecolor=INK, lw=2.0 * mark_scale, zorder=10)
    rf, cf = env.to_rc(traj[-1])
    ax.scatter([cf], [rf], marker="D", s=116 * area, facecolor=INK, edgecolor="white", lw=1.5 * mark_scale, zorder=10)
    for label, (rm, cm) in (("Start", (r0, c0)), ("Transition", (rf, cf))):
        ax.annotate(label, (cm + 0.42, rm), fontsize=annot_size, color=INK, ha="left", va="center", zorder=12)

    for rp, cp in pickup_positions:
        ax.annotate(
            "Pickup", (cp + 0.42, rp + 0.05), fontsize=annot_size, color=ACCENT, ha="left", va="center", zorder=12
        )
    for rd, cd in drop_positions:
        ax.annotate(
            "Drop",
            (cd + 0.16, rd + 0.28),
            fontsize=annot_size,
            color="#c44e52",
            bbox=dict(boxstyle="square,pad=0.10", facecolor=(1, 1, 1, 0.72), edgecolor="none"),
            zorder=12,
        )
    return ax


def plot_rollout_overview(env, traj_pot, skill_trajectories):
    """Figure 3: the potential-policy rollout beside up to three effective-policy panels."""
    small_panel = 1.15
    panel_gap = 0.07
    main_panel = 3 * small_panel + 2 * panel_gap
    small_mark_scale = small_panel / 1.55
    main_mark_scale = main_panel / 4.79
    x_gap = 1.00
    legend_h = 0.92
    legend_gap = 0.22
    title_h = 0.44
    title_gap = 0.05
    title_baseline_in = 0.07
    base_margin_l = 0.70
    base_margin_r = 0.70
    margin_t = 0.34  # Top margin in inches for panel letters.
    margin_b = 0.12
    panel_group_w = small_panel + x_gap + main_panel
    content_w = base_margin_l + panel_group_w + base_margin_r
    fig_w = max(content_w, 8.20)
    extra_w = fig_w - content_w
    margin_l = base_margin_l + 0.5 * extra_w
    fig_h = margin_b + legend_h + legend_gap + main_panel + title_gap + title_h + margin_t
    fs = figure_sizes(fig_w)
    fig = plt.figure(figsize=(fig_w, fig_h))

    grid_bottom = margin_b + legend_h + legend_gap
    main_x = margin_l
    stack_x = margin_l + main_panel + x_gap
    axes_stack = []
    for row in range(3):
        y = grid_bottom + (2 - row) * (small_panel + panel_gap)
        axes_stack.append(fig.add_axes([stack_x / fig_w, y / fig_h, small_panel / fig_w, small_panel / fig_h]))

    ax_main = fig.add_axes([main_x / fig_w, grid_bottom / fig_h, main_panel / fig_w, main_panel / fig_h])
    title_y = grid_bottom + main_panel + title_gap
    ax_eff_title = fig.add_axes([stack_x / fig_w, title_y / fig_h, small_panel / fig_w, title_h / fig_h])
    ax_pot_title = fig.add_axes([main_x / fig_w, title_y / fig_h, main_panel / fig_w, title_h / fig_h])
    for ax_title, label, color in [
        (ax_pot_title, r"$\pi_{\mathrm{pot}}$", INK),
        (ax_eff_title, r"$\pi_{\mathrm{pot}} \rightarrow \pi_{\mathrm{eff}}$", ACCENT),
    ]:
        ax_title.axis("off")
        ax_title.text(
            0.5,
            (title_baseline_in - title_gap) / title_h,
            label,
            ha="center",
            va="baseline",
            fontsize=fs["title"],
            color=color,
        )
    legend_side_margin = 0.12
    ax_leg = fig.add_axes(
        [
            legend_side_margin / fig_w,
            margin_b / fig_h,
            (fig_w - 2.0 * legend_side_margin) / fig_w,
            legend_h / fig_h,
        ]
    )

    panels = list(skill_trajectories.values())
    for i, ax in enumerate(axes_stack):
        if i >= len(panels):
            ax.axis("off")
            continue
        plot_rollout_panel(ax, env, traj_pot, panels[i], mark_scale=small_mark_scale)
        ax.set_anchor("C")

    draw_potential_rollout(ax_main, env, traj_pot, sizes=fs, mark_scale=main_mark_scale)
    ax_main.set_anchor("C")
    ax_leg.axis("off")
    ax_leg.legend(
        handles=_key_rollout_handles(overview=True),
        loc="center",
        ncol=3,
        fontsize=fs["legend"],
        handlelength=2.2,
        handletextpad=0.84,
        columnspacing=1.0,
        frameon=False,
    )

    for lab, panel_ax in (("i", ax_main), ("ii", axes_stack[0])):
        panel_letter(fig, panel_ax, lab, fs, color=INK)
    return fig
