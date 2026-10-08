"""Figures 2, 3, and 11: skills and potential policies on the open and key grids.

LPs: Clarabel, Goulart and Chen (2024).
"""

import numpy as np

from environments import GridEnv, KeyGridEnv
from plotting.plot_utils import init_style, save_fig
from plotting.plot_utils_diff import plot_rollout_overview, plot_simple_grid_overview, plot_skill_rollouts
from plotting_scripts._common import DEFAULT_OUT, parse_args, prepare_output, write_plot_data
from utils.blahut_arimoto import build_skill_data
from utils.occupancy_vertices import occupancy_to_policy
from utils.potential_policy import greedy_rollout, qlearn_potential_policy, sample_rollout

SEARCH_SEED = 0
GRID_DIRECTIONS, KEY_DIRECTIONS = 500, 1000  # Random LP directions per start state.
SKILL_THRESH = 0.01  # Minimum BA prior weight for displaying a key-grid rollout.
QLEARN = dict(n_iters=200_000, alpha=0.1, eps_start=1.0, eps_end=0.01, seed=0)
SKILL_STEPS = 15  # Number of transitions in each sampled skill rollout.
# Figure number -> (maximum skills shown, rollouts per skill, rollout seed).
ROLLOUT_PANELS = {11: (12, 3, 0), 3: (3, 2, 1)}


def active_skills(weights):
    """Return skills with prior >= SKILL_THRESH, sorted by decreasing prior."""
    weights = np.asarray(weights)
    active = np.where(weights >= SKILL_THRESH)[0]
    return active[np.argsort(weights[active])[::-1]]


def solve_skills_and_potential_policy(env, gamma, p0, start_states=None, *, n_directions):
    """Find skills at each selected start, then train a policy on their statewise MI values."""
    # data.I_s0[s] is BA's mutual information over the sampled skills, used as E_hat(s);
    # both the skill search and BA's stopping rule are approximate.
    data = build_skill_data(env, gamma, start_states=start_states, seed=SEARCH_SEED, n_directions=n_directions)
    pi_pot = qlearn_potential_policy(env, data.I_s0, gamma, p0, **QLEARN)  # pi_pot(a | s), shape (nS,nA).
    return data, pi_pot


def simple_grid_experiment():
    """Figure 2: skills on the open 5x5 grid, their MI field, and the potential policy."""
    env = GridEnv(rows=5, cols=5)
    gamma = 0.95  # Discount; the occupancy stopping time has Pr(T=t)=(1-gamma) gamma^t.
    p0 = np.ones(env.nS) / env.nS  # Distribution over start states S_0.
    data, pi_pot = solve_skills_and_potential_policy(env, gamma, p0, n_directions=GRID_DIRECTIONS)

    center = env.to_s(env.rows // 2, env.cols // 2)
    # Show every distinct occupancy mode found, without a probability cutoff.
    end_states = np.unique(np.argmax(data.pz_s0[center], axis=1))
    return {
        "env": env,
        "gamma": gamma,
        "I_grid": data.I_s0.reshape(env.rows, env.cols),
        "pi_pot": pi_pot,
        "center": center,
        "end_states": end_states,
    }


def key_grid_experiment():
    """Figures 3 and 11: sample skills from the final state of the potential-policy rollout."""
    env = KeyGridEnv(rows=5, cols=5)
    gamma = 0.99
    starts = env.valid_starts()
    p0 = np.zeros(env.nS)
    p0[starts] = 1.0 / len(starts)
    start_state = env.to_s(4, 2)  # Bottom hallway cell, without the key.
    data, pi_pot = solve_skills_and_potential_policy(env, gamma, p0, starts, n_directions=KEY_DIRECTIONS)

    # Follow modal actions and successors for int(1/(1-gamma)) steps, then switch to skills.
    traj_pot = greedy_rollout(env, pi_pot, start_state, int(1 / (1 - gamma)))
    dropoff = traj_pot[-1]
    # policies[z][s,a] = pi_z(a | s), recovered from rho_z(s,a | dropoff).
    policies = [occupancy_to_policy(rho, env.nS, env.nA) for rho in data.vertices_by_s0[dropoff]]
    skills = active_skills(data.p_prior[dropoff])

    def sample_rollouts(n_skills, n_rollouts, seed):
        rng = np.random.default_rng(seed)
        return {
            int(z): [sample_rollout(env, policies[z], dropoff, SKILL_STEPS, rng) for _ in range(n_rollouts)]
            for z in skills[:n_skills]
        }

    print(f"dropoff state {dropoff}; {len(skills)} skills >= {SKILL_THRESH}")
    print(f"rollout: state {start_state} -> {dropoff}, key at end: {'yes' if env.decode_state(dropoff)[1] else 'no'}")
    return {
        "env": env,
        "gamma": gamma,
        "traj_pot": traj_pot,
        "rollouts": {number: sample_rollouts(*panels) for number, panels in ROLLOUT_PANELS.items()},
    }


def main(out_dir=DEFAULT_OUT):
    out_dir = prepare_output(out_dir)
    init_style()

    grid = simple_grid_experiment()
    env = grid["env"]
    stem = "02_skills_as_yardsticks_of_empowerment"
    fig = plot_simple_grid_overview(env, grid["I_grid"], grid["center"], grid["end_states"], grid["pi_pot"])
    save_fig(stem, fig, plot_dir=out_dir)
    write_plot_data(
        out_dir,
        stem,
        {
            "schema_version": 1,
            "environment": {
                "type": "GridEnv",
                "rows": env.rows,
                "cols": env.cols,
                "slip": env.slip,
                "actions": list(env.actions),
            },
            "gamma": grid["gamma"],
            "search": "random",
            "n_directions": GRID_DIRECTIONS,
            "seeds": {"random_directions": SEARCH_SEED, "q_learning": QLEARN["seed"]},
            "estimated_capacity_nats_by_state": grid["I_grid"].tolist(),
            "center_skill_endpoints": {
                "start_state": int(grid["center"]),
                "end_states": [int(s) for s in grid["end_states"]],
            },
            "potential_policy_actions": np.argmax(grid["pi_pot"], axis=1).astype(int).tolist(),
        },
    )

    key = key_grid_experiment()
    env = key["env"]
    figures = {
        11: ("11_key_pickup_skill_rollouts_3x4", plot_skill_rollouts),
        3: ("03_key_potential_policy_centralized_states", plot_rollout_overview),
    }
    for number, (stem, plot) in figures.items():
        trajectories = key["rollouts"][number]
        save_fig(stem, plot(env, key["traj_pot"], trajectories), plot_dir=out_dir)
        write_plot_data(
            out_dir,
            stem,
            {
                "schema_version": 1,
                "environment": {
                    "type": "KeyGridEnv",
                    "rows": env.rows,
                    "cols": env.cols,
                    "slip": env.slip,
                    "actions": list(env.actions),
                    "hallway_col": env.hallway_col,
                    "key_position": list(env.key_pos),
                },
                "gamma": key["gamma"],
                "search": "random",
                "n_directions": KEY_DIRECTIONS,
                "seeds": {"random_directions": SEARCH_SEED, "q_learning": QLEARN["seed"], "rollouts": ROLLOUT_PANELS[number][2]},
                "potential_trajectory": [int(s) for s in key["traj_pot"]],
                "skill_trajectories": {str(z): rollouts for z, rollouts in trajectories.items()},
            },
        )


if __name__ == "__main__":
    args = parse_args(__doc__.splitlines()[0])
    main(args.out)
