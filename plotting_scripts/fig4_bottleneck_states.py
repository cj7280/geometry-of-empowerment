"""Figure 4: estimated empowerment in two bottleneck grids.

Sample random LP directions, then optimize skill probabilities with Blahut-Arimoto.
LPs: Clarabel, Goulart and Chen (2024).
"""

import hashlib
import json
import time

from environments import GridEnv, two_room_walls
from plotting.bottleneck_plots import plot_mi_heatmap_grid
from plotting.plot_utils import init_style, save_fig
from plotting_scripts._common import DEFAULT_OUT, ROOT, parse_args, prepare_output, write_plot_data
from utils.blahut_arimoto import BA_MAX_ITERS, BA_TOL, build_skill_data
from utils.figure_cache import FIGURE_CACHE_SCHEMA_VERSION, load_matching_cache, write_matching_cache
from utils.occupancy_vertices import LP_SOLVER

ROWS, COLS = 8, 8
GAMMAS = (0.5, 0.95)
SLIP = 0.0
SEARCH_SEED = 0
N_DIRECTIONS = 500  # Random LP directions per start state.
CACHE_PATH = ROOT / "data" / "mi_field_cache.pkl"


def bottleneck_configs():
    """Return (panel label, grid) pairs with a central or corner doorway."""
    return [
        (
            label,
            GridEnv(rows=ROWS, cols=COLS, slip=SLIP, walls=two_room_walls(rows=ROWS, wall_col=3, hole_row=hole_row)),
        )
        for label, hole_row in (("Two room\n(center door)", 4), ("Two room\n(corner door)", 0))
    ]


def cache_key(configs, gamma):
    """Hash the cache schema, grid settings, seed, random-search settings, and solver name."""
    inputs = {
        "cache_schema_version": FIGURE_CACHE_SCHEMA_VERSION,
        "names": [name for name, _ in configs],
        "walls": [sorted(map(tuple, map(sorted, env.walls))) for _, env in configs],
        "rows": ROWS,
        "cols": COLS,
        "gamma": gamma,
        "slip": SLIP,
        "seed": SEARCH_SEED,
        "lp_solver": LP_SOLVER,
    }
    inputs.update(search="random", n_directions=N_DIRECTIONS, ba_tol=BA_TOL, ba_max_iters=BA_MAX_ITERS)
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()


def compute_capacity_fields(configs, gamma, *, recompute=False):
    """Return (results, cache_key, recomputed), where results maps panel labels to MI fields."""
    key = cache_key(configs, gamma)
    # Each discount has its own random-direction cache.
    cache_path = CACHE_PATH if gamma == 0.95 else CACHE_PATH.with_stem(f"mi_field_cache_gamma_{gamma:g}")
    cache_path = cache_path.with_stem(f"{cache_path.stem}_random")
    cached = load_matching_cache(cache_path, key, recompute=recompute)
    if cached is not None:
        print(f"MI field loaded from cache ({cache_path})")
        return cached, key, False

    results = {}
    for index, (name, env) in enumerate(configs):
        print(f"[{index + 1}/{len(configs)}] {name.replace(chr(10), ' ')} ...", end=" ", flush=True)
        start = time.time()
        data = build_skill_data(env, gamma, seed=SEARCH_SEED, n_directions=N_DIRECTIONS)
        # I_s0[s] estimates E(s)=sup_{skills, p(z|s)} I(Z;S_+ | S_0=s), in nats;
        # mean_I averages this field uniformly over the grid's start states.
        results[name] = {"I_s0": data.I_s0, "mean_I": float(data.I_s0.mean())}
        print(
            f"mean estimated I = {results[name]['mean_I']:.3f}, "
            f"({time.time() - start:.1f}s)"
        )
    write_matching_cache(cache_path, key, results)
    print(f"MI field cached to {cache_path}")
    return results, key, True


def save_capacity_fields(out_dir, configs, results, gamma):
    stem = f"04_bottleneck_states_gamma_{str(gamma).replace('.', 'p')}"
    fig = plot_mi_heatmap_grid(configs, {gamma: results})
    save_fig(stem, fig, plot_dir=out_dir)
    write_plot_data(
        out_dir,
        stem,
        {
            "schema_version": 2,
            "units": "nats",
            "within_skill_gamma": gamma,
            "search": "random",
            "n_directions": N_DIRECTIONS,
            "configurations": [
                {
                    "name": name.replace("\n", " "),
                    "rows": env.rows,
                    "cols": env.cols,
                    "slip": env.slip,
                    "walls": [sorted(edge) for edge in sorted(env.walls, key=sorted)],
                    "estimated_one_rollout_capacity_nats": results[name]["I_s0"].tolist(),
                    "mean_estimated_one_rollout_capacity_nats": float(results[name]["mean_I"]),
                }
                for name, env in configs
            ],
        },
    )


def main(out_dir=DEFAULT_OUT, *, recompute=False):
    out_dir = prepare_output(out_dir)
    init_style()
    configs = bottleneck_configs()
    for gamma in GAMMAS:
        print(f"{len(configs)} configurations, gamma={gamma}, slip={SLIP}, search=random")
        results, _, _ = compute_capacity_fields(configs, gamma, recompute=recompute)
        save_capacity_fields(out_dir, configs, results, gamma)


if __name__ == "__main__":
    args = parse_args(__doc__.splitlines()[0], recompute=True)
    main(args.out, recompute=args.recompute_fig4)
