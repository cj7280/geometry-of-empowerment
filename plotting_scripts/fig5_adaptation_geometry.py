"""Figure 5: higher adaptation (J) can coincide with lower empowerment (I).

Compare two and three absorbing states using the same coordinate scale.
Symmetry makes the skill mixture equal the baseline in both examples.
"""

import numpy as np

from plotting.plot_style import PUB_DENSE_PDF_RASTER_DPI, apply_pub_style, save_pub_figure
from plotting.width_mi_plots_domains import plot_symmetric_adaptation_geometry
from plotting_scripts._common import DEFAULT_OUT, parse_args, prepare_output, write_plot_data
from utils.adaptation_examples import symmetric_absorbing_family
from utils.charts import align2d, chart_kl, kl_level_set, width_chart


def adaptation_geometry(family, first_deg, second_deg=None):
    """Compute chart data and check the width and information identities.

    The angles orient the chart to match the MDP cartoon.
    """
    q = family.mixture  # q(s)=sum_z p(z)P_z(s)=b(s), shape (K,) in this symmetric family.
    skills = family.skill_distributions  # P_z(s), shape (K,K), over absorbing states.
    assert np.allclose(family.baseline, q, atol=1e-12), "Figure 5 requires the baseline to equal the skill mixture q"
    coords, basis, mean = width_chart(skills, q)  # u_z=(P_z-q)/sqrt(q) in a (K,2) chart.
    rotation = align2d(coords, first_deg, second_deg)
    coords, basis = coords @ rotation.T, rotation @ basis
    assert np.allclose(mean, q, atol=1e-12), "Figure 5's uniformly weighted skills must have mean q"
    reconstructed_skills = mean[None, :] + np.sqrt(q)[None, :] * (coords @ basis)
    assert np.allclose(reconstructed_skills, skills, atol=1e-10), (
        "Rotating the chart must preserve the skill probabilities represented by its points"
    )

    # In 2-D, J = Per(conv U) / (2 sqrt(2 pi)); a segment's perimeter counts its length twice.
    if len(coords) == 2:
        hull_order = np.argsort(coords[:, 0])
        perimeter_multiplier = 2
        hull_perimeter = perimeter_multiplier * float(np.linalg.norm(coords[hull_order[1]] - coords[hull_order[0]]))
    else:
        centroid = coords.mean(axis=0)
        hull_order = np.argsort(np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0]))
        hull = coords[hull_order]
        edge_lengths = np.linalg.norm(np.roll(hull, -1, axis=0) - hull, axis=1)
        if not np.allclose(edge_lengths, edge_lengths[0], atol=1e-10):
            raise ValueError("the canonical three-state width panel expects an equilateral hull")
        perimeter_multiplier = len(hull)
        hull_perimeter = float(edge_lengths.sum())
    assert np.isclose(family.gaussian_width, hull_perimeter / (2.0 * np.sqrt(2.0 * np.pi)), atol=1e-10), (
        "Figure 5's plotted hull perimeter must give the same Gaussian width as the skill distributions"
    )

    # Symmetry gives D_KL(P_z || q) = I and equal ||(P_z-q)/sqrt(q)||_2 for all z.
    skill_kl = chart_kl(coords, q, basis)  # D_KL(P_z || q), shape (K,), in nats.
    assert np.allclose(skill_kl, family.information_nats, atol=1e-10), (
        "In Figure 5's symmetric examples, every skill must lie on the KL = I contour"
    )
    skill_radii = np.linalg.norm(coords, axis=1)
    assert np.allclose(skill_radii, skill_radii[0], atol=1e-10), (
        "Figure 5's symmetric skills must be equally distant from the chart origin"
    )
    rank = int(np.linalg.matrix_rank(coords - coords.mean(axis=0), tol=1e-10))
    kl_contour, level_deviation = None, 0.0
    if rank != 1:
        kl_contour = kl_level_set(q, basis, family.information_nats)
        level_deviation = float(np.nanmax(np.abs(chart_kl(kl_contour, q, basis) - family.information_nats)))
        assert level_deviation < 1e-10, "Figure 5's computed KL contour differs from I by at least 1e-10 nats"

    return {
        "coords": coords,
        "basis": basis,
        "hull_order": hull_order,
        "hull_perimeter": hull_perimeter,
        "perimeter_multiplier": perimeter_multiplier,
        "rank": rank,
        "kl_contour": kl_contour,
    }


def family_record(family, geometry, drawn):
    perimeter = geometry["hull_perimeter"]
    return {
        "n_absorbing_states": family.n_absorbing,
        "commitment": family.commitment,
        "skill_distributions": family.skill_distributions.tolist(),
        "mixture_q": family.mixture.tolist(),
        "information_nats": family.information_nats,
        "gaussian_width": family.gaussian_width,
        "u_coordinates": geometry["coords"].tolist(),
        "u_basis": geometry["basis"].tolist(),
        "intrinsic_chart_rank": geometry["rank"],
        "information_panel_occupancy_glyphs": drawn["information_panel_occupancy_glyphs"],
        "hull_perimeter": perimeter,
        "convex_hull_vertex_order": [int(index) for index in geometry["hull_order"]],
        "winner_field": drawn["winner_field"],
    }


def build_figure(x_family, y_family):
    """Draw Figure 5 for the two families and return it with its plot-data record."""
    if not (
        y_family.gaussian_width > x_family.gaussian_width
        and y_family.information_nats < x_family.information_nats
        and y_family.whitened_radius < x_family.whitened_radius
    ):
        raise ValueError("families must show larger J but smaller I/radius for the 3-state row")
    x_geometry = adaptation_geometry(x_family, 180.0)
    y_geometry = adaptation_geometry(y_family, 90.0, 210.0)
    fig, render = plot_symmetric_adaptation_geometry((x_family, x_geometry), (y_family, y_geometry))
    x_drawn, y_drawn = render.pop("rows")
    manifest = {
        "figure": "adaptation_geometry",
        **render,
        "two_state": family_record(x_family, x_geometry, x_drawn),
        "three_state": family_record(y_family, y_geometry, y_drawn),
    }
    return fig, manifest


def main(out_dir=DEFAULT_OUT):
    out_dir = prepare_output(out_dir)
    apply_pub_style()
    x_family = symmetric_absorbing_family(2, commitment=1.00)
    y_family = symmetric_absorbing_family(3, commitment=0.65)
    fig, manifest = build_figure(x_family, y_family)
    save_pub_figure(fig, out_dir / "05_adaptation_geometry", dpi=PUB_DENSE_PDF_RASTER_DPI)
    write_plot_data(out_dir, "05_adaptation_geometry", manifest)


if __name__ == "__main__":
    args = parse_args(__doc__.splitlines()[0])
    main(args.out)
