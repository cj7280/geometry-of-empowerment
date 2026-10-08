"""Figures 9 and 10: the bound J >= I / (c * sqrt(2*pi)).

Figure 9 varies commitment; Figure 10 varies the number of absorbing targets.
Here c is the smallest constant satisfying log(1+x) <= c * sqrt(x).
"""

import numpy as np

from plotting.kl_width_plots import plot_kl_width_quality, plot_kl_width_vs_bound
from plotting.plot_style import save_pub_figure
from plotting.plot_utils import init_style
from plotting_scripts._common import DEFAULT_OUT, parse_args, prepare_output, write_plot_data
from utils.adaptation_bound import KL_SQRT_CHI2_CONSTANT, compute_J_exact, simplex_frame_J
from utils.adaptation_examples import block_family
from utils.information import mutual_information

SQ2PI = float(np.sqrt(2.0 * np.pi))
BOUND_SLOPE = KL_SQRT_CHI2_CONSTANT * SQ2PI  # J >= I / BOUND_SLOPE.
INLAY_TS = [0.05, 0.2, 0.9]


def exact_I_and_J(n_states, n_blocks, commitment):
    """Compute mutual information and adapted return for the plotted block family."""
    skills, prior = block_family(n_states, n_blocks, commitment)
    information = mutual_information(skills, prior)
    adaptation = compute_J_exact(skills, prior, np.ones(n_states) / n_states)
    return information, adaptation


def commitment_sweep():
    """Figure 9: 4 absorbing states, with t = 0.02, 0.05, 0.1, ..., 0.9, 1."""
    ts = np.concatenate([[0.02, 0.05], np.linspace(0.1, 0.9, 9), [1.0]])
    I_t, J_t = np.array([exact_I_and_J(4, 4, float(t)) for t in ts]).T
    bound_t = I_t / BOUND_SLOPE  # The theorem's lower bound on J at each commitment t.
    assert np.all(J_t >= bound_t), "Figure 9's Gaussian widths must lie above their information-based lower bounds"
    return ts, I_t, J_t, bound_t


def committed_absorbing_states():
    """Figure 10: K skills, each selecting a distinct absorbing target with certainty."""
    state_counts = np.array([2, 4, 8, 16, 32, 64])
    # Distributions are normalized over targets, excluding the common start state.
    # One state per block gives P_z(g) = 1[g=z], with uniform skill prior and baseline.
    I_K, J_K = np.array([exact_I_and_J(int(K), int(K), 1.0) for K in state_counts]).T
    J_closed = np.array([simplex_frame_J(int(K), np.sqrt(K - 1.0)) if K >= 3 else 2.0 / SQ2PI for K in state_counts])
    assert np.allclose(I_K, np.log(state_counts)), "Disjoint, uniformly weighted skills must have MI = log(K)"
    assert np.allclose(J_K, J_closed, rtol=1e-9), "Figure 10's widths must match the regular-simplex formula"
    assert np.all(I_K <= BOUND_SLOPE * J_K), "Every Figure 10 point must satisfy the information-to-width bound"
    return state_counts, I_K, J_K


def main(out_dir=DEFAULT_OUT):
    out_dir = prepare_output(out_dir)
    init_style()

    ts, I_t, J_t, bound_t = commitment_sweep()
    fig = plot_kl_width_quality(ts, J_t, bound_t, inlay_ts=INLAY_TS)
    save_pub_figure(fig, out_dir / "09_kl_width_J_quality")
    write_plot_data(
        out_dir,
        "09_kl_width_J_quality",
        {
            "family": "blocks_t",
            "n_states": 4,
            "n_blocks": 4,
            "t": ts.tolist(),
            "J": J_t.tolist(),
            "kl_sqrt_chi2_constant": KL_SQRT_CHI2_CONSTANT,
            "I_over_c0_sqrt_2pi": bound_t.tolist(),
            "inlay_t": INLAY_TS,
        },
    )

    state_counts, I_K, J_K = committed_absorbing_states()
    fig, axis_record = plot_kl_width_vs_bound(I_K, J_K, state_counts, BOUND_SLOPE)
    save_pub_figure(fig, out_dir / "10_kl_width_J_vs_bound")
    write_plot_data(
        out_dir,
        "10_kl_width_J_vs_bound",
        {
            "family": "absorbing_states_committed",
            "n_absorbing_states": state_counts.tolist(),
            "occupancy_normalization": "targets only; common starting state excluded",
            "commitment_t": 1.0,
            "K": state_counts.tolist(),
            "x_I_nats": I_K.tolist(),
            "y_J": J_K.tolist(),
            "reference_slope_c0_sqrt_2pi": BOUND_SLOPE,
            **axis_record,
        },
    )


if __name__ == "__main__":
    args = parse_args(__doc__.splitlines()[0])
    main(args.out)
