"""Figure 6: increasing mutual information as skill distributions collapse.

Shrink a ring of Gaussian skills while adding skills and narrowing their
spread. MI rises, but differences in expected reward for any 1-Lipschitz
reward shrink. Gray lines show equal values of a linear reward; panel labels
give the width (diameter) of the skill means, bounding those reward differences.
"""

from __future__ import annotations

import numpy as np

from plotting.mi_collapse_plots import plot_mi_information_collapse
from plotting.plot_style import save_pub_figure
from plotting.plot_utils import init_style
from plotting_scripts._common import DEFAULT_OUT, parse_args, prepare_output, write_plot_data
from utils.information import gaussian_mixture_information_mc

CENTER = np.array([0.0, 0.0])
RING_RADIUS = 1.5
SEPARATION_OVER_SIGMA = 4.5
PANELS = [(1.0, 6), (1e-1, 24), (1e-3, 96)]
REWARD_DIRECTION_DEGREES = 25.0
REWARD_DIRECTION = np.array(
    [
        np.cos(np.radians(REWARD_DIRECTION_DEGREES)),
        np.sin(np.radians(REWARD_DIRECTION_DEGREES)),
    ]
)
MI_SAMPLES_PER_SKILL = 20_000


def skill_means(scale, n_skills):
    angles = np.pi / 2 + 2 * np.pi * np.arange(n_skills) / n_skills
    return CENTER + scale * RING_RADIUS * np.stack(
        [np.cos(angles), np.sin(angles)],
        axis=1,
    )


def skill_sigma(scale, n_skills):
    neighbor_distance = 2 * RING_RADIUS * scale * np.sin(np.pi / n_skills)
    return neighbor_distance / SEPARATION_OVER_SIGMA


def collapse_panels():
    """Compute three Gaussian mixtures and check increasing MI with decreasing reward-range bounds."""
    results = []
    panel_means = []
    for index, (scale, n_skills) in enumerate(PANELS):
        means = skill_means(scale, n_skills)
        sigma = skill_sigma(scale, n_skills)
        information = gaussian_mixture_information_mc(means, sigma, MI_SAMPLES_PER_SKILL, seed=10 + index)
        # Width is the maximum distance between skill means (opposite ring points).
        # NOT to be confused with Gaussian Width, which is not directly relevant here
        # since rewards are now 1-Lipschitz.
        # It bounds expected-reward differences for any 1-Lipschitz reward,
        # since the skills are translations of the same Gaussian.
        width = 2 * RING_RADIUS * scale
        reward_values = means @ REWARD_DIRECTION  # Exact E_{P_z}[r(X)] for r(x)=direction @ x.
        results.append(
            {
                "s": scale,
                "K": n_skills,
                "sigma": sigma,
                "I": information,
                "width": width,
                "logK": float(np.log(n_skills)),
                "linear_expected_reward_range": float(reward_values.max() - reward_values.min()),
                "lipschitz_expected_reward_range_bound": float(width),
            }
        )
        panel_means.append(means)
        assert abs(information - np.log(n_skills)) < 0.15, (
            "Figure 6 requires estimated MI to be within 0.15 nats of log(K), indicating distinguishable skills"
        )

    assert all(left["I"] < right["I"] for left, right in zip(results, results[1:])), (
        "Figure 6 must show increasing mutual information from left to right"
    )
    assert all(
        left["lipschitz_expected_reward_range_bound"] > 8 * right["lipschitz_expected_reward_range_bound"]
        for left, right in zip(results, results[1:])
    ), "Figure 6 must show the expected-reward range bound shrinking by more than eightfold per panel"
    return results, panel_means


def main(out_dir=DEFAULT_OUT):
    out_dir = prepare_output(out_dir)
    init_style()
    results, panel_means = collapse_panels()
    fig, render_record = plot_mi_information_collapse(
        results,
        panel_means,
        center=CENTER,
        ring_radius=RING_RADIUS,
        reward_direction=REWARD_DIRECTION,
    )
    save_pub_figure(fig, out_dir / "06_mi_information_collapse")

    write_plot_data(out_dir, "06_mi_information_collapse", {
        "schema_version": 2,
        "panels": [{key: panel[key] for key in ("s", "K", "sigma", "I", "width")} for panel in results],
        "center": CENTER.tolist(),
        "ring_radius": RING_RADIUS,
        "reward": {
            "grad_dir_deg": REWARD_DIRECTION_DEGREES,
            "iso_spacing": 1.0,
        },
        "mi_mc": {
            "n_per_skill": MI_SAMPLES_PER_SKILL,
            "seeds": list(range(10, 10 + len(PANELS))),
        },
        **render_record,
    })


if __name__ == "__main__":
    args = parse_args(__doc__.splitlines()[0])
    main(args.out)
