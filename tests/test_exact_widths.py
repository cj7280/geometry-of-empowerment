"""Check Gaussian width formulas, the final adaptation bound, and Figure 5 geometry."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import utils.adaptation_bound as ab
from plotting_scripts.fig5_adaptation_geometry import build_figure
from utils.adaptation_examples import block_family, symmetric_absorbing_family, symmetric_absorbing_information
from utils.charts import chart_kl, width_chart
from utils.information import mutual_information


def blocks_family(n_targets, K, t):
    return block_family(n_targets, K, t)[0]


def halves_family(n_targets):
    return block_family(n_targets, 2, 1.0)[0]


def test_gaussian_max_mean_anchors():
    assert abs(ab.gaussian_max_mean(2) - 1.0 / math.sqrt(math.pi)) < 1e-10
    assert abs(ab.gaussian_max_mean(3) - 3.0 / (2.0 * math.sqrt(math.pi))) < 1e-10


def test_simplex_frame_anchors():
    assert abs(ab.simplex_frame_J(2) - math.sqrt(2.0 / math.pi)) < 1e-10
    assert abs(ab.simplex_frame_J(3) - 3.0 * math.sqrt(3.0) / (2.0 * math.sqrt(2.0 * math.pi))) < 1e-10


def test_exact_widths_match_mc():
    n = 96
    B = np.ones(n) / n
    for K in (2, 3, 4, 8, 16):
        S = halves_family(n) if K == 2 else blocks_family(n, K, 1.0 / math.sqrt(K - 1))
        prior = np.ones(K) / K
        j_mc = ab.compute_J(S, prior, B, n_samples=400_000, seed=1)
        assert abs(ab.compute_J_exact(S, prior, B) - j_mc) < 0.01, K


def test_exact_width_nonsolvable_fallback():
    n = 24
    B = np.ones(n) / n
    rng = np.random.default_rng(3)
    S = rng.dirichlet(np.ones(n), size=5)
    prior = np.ones(5) / 5
    # The high-rank fallback must match compute_J with its default seed and sample count.
    j = ab.compute_J_exact(S, prior, B)
    assert abs(j - ab.compute_J(S, prior, B)) < 1e-12


def test_final_information_to_adaptation_bound():
    n = 48
    rng = np.random.default_rng(7)
    ensembles = [
        (halves_family(n), np.ones(2) / 2),
        (blocks_family(n, 4, 0.5), np.ones(4) / 4),
        (rng.dirichlet(np.ones(n) * 0.3, size=6), np.ones(6) / 6),
        ((1 - 1e-3) * np.eye(n) + 1e-3 / n, np.ones(n) / n),  # Nearly deterministic skills.
        (blocks_family(n, 3, 0.6), np.array([0.7, 0.2, 0.1])),  # nonuniform prior
    ]
    for S, prior in ensembles:
        information = mutual_information(S, prior)
        adaptation = ab.compute_J_exact(S, prior, np.ones(n) / n)
        bound = information / (ab.KL_SQRT_CHI2_CONSTANT * math.sqrt(2.0 * math.pi))
        assert adaptation >= bound


def test_sharpened_information_to_J_constant():
    """The final bound uses the optimal log(1+x)/sqrt(x) coefficient."""
    c0 = ab.KL_SQRT_CHI2_CONSTANT
    x = np.linspace(1e-6, 20.0, 400_001)
    assert abs(np.max(np.log1p(x) / np.sqrt(x)) - c0) < 1e-9


def test_symmetric_absorbing_adaptation_flip_has_one_shared_chart():
    """Check higher J but lower I for three states, with identical chart scales across panels."""
    two = symmetric_absorbing_family(2, 1.00)
    three = symmetric_absorbing_family(3, 0.65)

    for family in (two, three):
        assert np.allclose(family.mixture, family.baseline, atol=1e-12)
        assert math.isclose(
            family.information_nats,
            symmetric_absorbing_information(family.n_absorbing, family.commitment),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            family.whitened_radius,
            family.commitment * math.sqrt(family.n_absorbing - 1.0),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        coords, basis, mean = width_chart(family.skill_distributions, family.mixture)
        assert np.allclose(mean, family.mixture, atol=1e-12)
        assert np.allclose(
            mean[None, :] + np.sqrt(family.mixture)[None, :] * (coords @ basis),
            family.skill_distributions,
            atol=1e-10,
        )
        # Independently integrate directional support to check the exact width.
        theta = 2.0 * np.pi * (np.arange(16_384) + 0.5) / 16_384
        directional_returns = np.column_stack([np.cos(theta), np.sin(theta)]) @ coords.T
        support = directional_returns.max(axis=1)
        winner = directional_returns.argmax(axis=1)
        assert support.min() >= -1e-12
        assert set(winner) == set(range(family.n_absorbing))
        assert math.isclose(
            math.sqrt(math.pi / 2.0) * float(np.mean(support)),
            family.gaussian_width,
            rel_tol=0.0,
            abs_tol=1e-8,
        )

    # The two-state simplex has dimension one, so the second chart axis is zero.
    two_coords, two_basis, _ = width_chart(two.skill_distributions, two.mixture)
    assert np.allclose(two_coords[:, 1], 0.0, atol=1e-12)
    assert np.allclose(two_basis[1], 0.0, atol=1e-12)

    # The triangle has a smaller radius and MI but a larger Gaussian width than the segment.
    assert three.whitened_radius < two.whitened_radius
    assert three.gaussian_width > two.gaussian_width
    assert three.information_nats < two.information_nats
    # Check the two-decimal I and J values for Figure 5.
    assert round(two.gaussian_width, 2) == 0.80
    assert round(three.gaussian_width, 2) == 0.95
    assert round(two.information_nats, 2) == 0.69
    assert round(three.information_nats, 2) == 0.39
    # For q=Unif(3), log(3/2) is the smallest KL value on the simplex
    # boundary. Staying below it keeps panel (f)'s exact KL level set closed.
    assert three.information_nats < math.log(3.0 / 2.0)

    fig, manifest = build_figure(two, three)
    try:
        scale = manifest["shared_scale"]
        assert np.allclose(scale["xlim"], [-1.28, 1.28], atol=1e-12)
        assert np.allclose(scale["ylim"], [-1.28, 1.28], atol=1e-12)
        assert np.allclose(scale["geometry_panel_size_inches"], [3.35, 3.35])
        for record, info_axis in zip((manifest["two_state"], manifest["three_state"]), fig.axes[4:6]):
            field = record["winner_field"]
            assert field["grid_size"] == 500
            assert math.isclose(field["constant_color_mix"], 0.34, abs_tol=1e-12)
            u_radii = np.linalg.norm(record["u_coordinates"], axis=1)
            expected_radius = record["commitment"] * math.sqrt(record["n_absorbing_states"] - 1)
            assert np.allclose(u_radii, expected_radius, atol=1e-12)
            skill_kl = chart_kl(
                np.asarray(record["u_coordinates"]), np.asarray(record["mixture_q"]), np.asarray(record["u_basis"])
            )
            assert np.allclose(skill_kl, record["information_nats"], atol=1e-12)
            glyphs = record["information_panel_occupancy_glyphs"]
            assert glyphs["shared_probability_vmax"] == 1.0
            assert len(glyphs["centers"]) == record["n_absorbing_states"] + 1
            distributions = np.vstack([record["skill_distributions"], record["mixture_q"]])
            bars = [patch for patch in info_axis.patches if isinstance(patch, Rectangle)]
            assert np.allclose(
                [bar.get_height() for bar in bars],
                distributions.ravel() * glyphs["glyph_height"] * 0.74,
                atol=1e-12,
            )

        for mdp_axis in fig.axes[:2]:
            labels = [artist.get_text() for artist in mdp_axis.texts]
            assert "$s$" in labels
            assert "$s_0$" not in labels
        assert {artist.get_text() for artist in fig.axes[0].texts} >= {"$s_{1}$", "$s_{2}$"}
        assert {artist.get_text() for artist in fig.axes[1].texts} >= {"$s_{1}$", "$s_{2}$", "$s_{3}$"}
        two_arrows = [patch for patch in fig.axes[0].patches if isinstance(patch, FancyArrowPatch)]
        three_arrows = [patch for patch in fig.axes[1].patches if isinstance(patch, FancyArrowPatch)]
        assert len(two_arrows) == 2
        assert len(three_arrows) == 9
        assert all(math.isclose(arrow.get_alpha(), 1.0, abs_tol=1e-12) for arrow in two_arrows)
        assert any(arrow.get_alpha() < 0.5 for arrow in three_arrows)
        assert any(math.isclose(arrow.get_alpha(), 1.0, abs_tol=1e-12) for arrow in three_arrows)
    finally:
        plt.close(fig)

