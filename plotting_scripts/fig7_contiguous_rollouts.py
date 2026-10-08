"""Figure 7: contiguous-rollout empowerment on the two bottleneck grids."""

import numpy as np

from plotting.bottleneck_plots import plot_contiguous_fields
from plotting.plot_utils import init_style, save_fig
from plotting_scripts._common import DEFAULT_OUT, parse_args, prepare_output, write_plot_data
from plotting_scripts.fig4_bottleneck_states import bottleneck_configs
from utils.blahut_arimoto import build_skill_data
from utils.temporal_empowerment import solve_reactive_sources

GAMMA, BETA = 0.95, 0.3
N_DIRECTIONS, SEED = 500, 0
SOURCE_OPTIONS = dict(max_iterations=10_000, anneal_iterations=8000,
                      uniform_weight=0.1, learning_rate=0.05)


def main(out_dir=DEFAULT_OUT):
    out_dir = prepare_output(out_dir)
    init_style()
    configs = bottleneck_configs()
    fields = []
    for name, env in configs:
        print(f"{name.replace(chr(10), ' ')}: sampling {N_DIRECTIONS} directions per start", flush=True)
        # Sample skill vertices at every state ONCE for this grid. Their
        # one-rollout occupancies stay fixed throughout the source optimizations.
        data = build_skill_data(env, GAMMA, seed=SEED, n_directions=N_DIRECTIONS)
        values = []
        for start in range(env.nS):
            # Score this starting state by optimizing p(z | s) EVERYWHERE.
            # Each start gets a separate optimization and may prefer a different
            # source across the whole grid. The plotted field keeps only scores.
            value, _ = solve_reactive_sources(data.pz_s0, data.p_prior, BETA, start_state=start, **SOURCE_OPTIONS)
            values.append(value)
            print(f"  start {start + 1}/{env.nS}: MI = {value:.4f} nats", flush=True)
        values = np.asarray(values)
        fields.append(values)
    stem = "07_bottleneck_temporal_empowerment"
    write_plot_data(out_dir, stem, {
        "gamma": GAMMA, "beta": BETA, "search": "random",
        "n_directions": N_DIRECTIONS, "seed": SEED,
        "source_solver": {"objective": "MI at each start, optimized separately",
                          "method": "Adam", "initialization": "uniform",
                          "uniform_mixing": "linear decay to zero", "restarts": 0,
                          **SOURCE_OPTIONS,
                          "global_optimum_certified": False},
        "panels": [{"name": name.replace("\n", " "), "rows": env.rows,
                    "cols": env.cols, "walls": [sorted(edge) for edge in sorted(env.walls, key=sorted)],
                    "contiguous_mi_nats": np.asarray(values).tolist()}
                   for (name, env), values in zip(configs, fields)],
    })

    save_fig(stem, plot_contiguous_fields(configs, fields), plot_dir=out_dir)


if __name__ == "__main__":
    main(parse_args(__doc__).out)
