"""Regenerate the paper figures in figures/paper-figures/.

    uv run python make_all_figures.py          # all figures
    uv run python make_all_figures.py 7 9 10    # selected figures

Run generators in separate processes and check that their outputs are fresh.
Use --out to change the output folder; --recompute-fig4 refreshes its cache.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Generator module -> expected output files; each selected module runs once.
GENERATORS = {
    "plotting_scripts.fig4_bottleneck_states": [
        "04_bottleneck_states_gamma_0p5.pdf",
        "04_bottleneck_states_gamma_0p5.png",
        "04_bottleneck_states_gamma_0p5_plot_data.json",
        "04_bottleneck_states_gamma_0p95.pdf",
        "04_bottleneck_states_gamma_0p95.png",
        "04_bottleneck_states_gamma_0p95_plot_data.json",
    ],
    "plotting_scripts.fig2_3_11_grid_skills": [
        "03_key_potential_policy_centralized_states.pdf",
        "03_key_potential_policy_centralized_states.png",
        "03_key_potential_policy_centralized_states_plot_data.json",
        "02_skills_as_yardsticks_of_empowerment.pdf",
        "02_skills_as_yardsticks_of_empowerment.png",
        "02_skills_as_yardsticks_of_empowerment_plot_data.json",
        "11_key_pickup_skill_rollouts_3x4.pdf",
        "11_key_pickup_skill_rollouts_3x4.png",
        "11_key_pickup_skill_rollouts_3x4_plot_data.json",
    ],
    "plotting_scripts.fig5_adaptation_geometry": [
        "05_adaptation_geometry.pdf",
        "05_adaptation_geometry.png",
        "05_adaptation_geometry_plot_data.json",
    ],
    "plotting_scripts.fig6_mi_information_collapse": [
        "06_mi_information_collapse.pdf",
        "06_mi_information_collapse.png",
        "06_mi_information_collapse_plot_data.json",
    ],
    "plotting_scripts.fig7_contiguous_rollouts": [
        "07_bottleneck_temporal_empowerment.pdf",
        "07_bottleneck_temporal_empowerment.png",
        "07_bottleneck_temporal_empowerment_plot_data.json",
    ],
    "plotting_scripts.fig9_10_kl_width_bounds": [
        "09_kl_width_J_quality.pdf",
        "09_kl_width_J_quality.png",
        "09_kl_width_J_quality_plot_data.json",
        "10_kl_width_J_vs_bound.pdf",
        "10_kl_width_J_vs_bound.png",
        "10_kl_width_J_vs_bound_plot_data.json",
    ],
}

FIGURE_NUMBERS = {paper[:2]: script for script, papers in GENERATORS.items() for paper in papers}


def run_generator(script: str, fig_dir: Path, *, recompute_fig4: bool = False) -> None:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    # Generators import the local plotting, math, and environment packages.
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    command = [sys.executable, "-m", script, "--out", str(fig_dir)]
    if recompute_fig4 and script == "plotting_scripts.fig4_bottleneck_states":
        command.append("--recompute-fig4")
    print(f"-- running {script}", flush=True)
    t0 = time.time()
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f"-- {script} done in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figures", nargs="*", type=int, help="figure numbers to rebuild (default: all canonical figures)")
    parser.add_argument(
        "--out",
        default="figures/paper-figures",
        help="output directory (relative to the repository root)",
    )
    parser.add_argument(
        "--recompute-fig4",
        action="store_true",
        help="ignore and replace Figure 4's cached capacity fields",
    )
    args = parser.parse_args()
    if args.figures:
        wanted = {f"{n:02d}" for n in args.figures}
        unknown = wanted - set(FIGURE_NUMBERS)
        if unknown:
            parser.error(f"unknown figure number(s): {sorted(unknown)}")
        scripts = {FIGURE_NUMBERS[n] for n in wanted}
    else:
        wanted = set(FIGURE_NUMBERS)
        scripts = set(GENERATORS)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for script in GENERATORS:
        if script in scripts:
            run_generator(script, out_dir, recompute_fig4=args.recompute_fig4)

    stale = [
        paper
        for script, papers in GENERATORS.items()
        if script in scripts
        for paper in papers
        if not (out_dir / paper).exists() or (out_dir / paper).stat().st_mtime < start
    ]
    if stale:
        print(f"ERROR: not regenerated this run: {stale}", file=sys.stderr)
        return 1
    print(f"all requested paper figures refreshed in {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
