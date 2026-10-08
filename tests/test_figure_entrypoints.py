"""Exercise the Python-only release workflow and scientific helper functions."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import make_all_figures as runner

ROOT = Path(__file__).resolve().parents[1]


def subprocess_env(tmp_path):
    return {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "mplconfig"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }


def test_imports_do_not_run_experiments_or_change_cwd(tmp_path):
    work = tmp_path / "unrelated-project"
    work.mkdir()
    (work / "pyproject.toml").write_text("# A different project's root\n")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
from pathlib import Path
from unittest.mock import patch
import matplotlib as mpl
import matplotlib.pyplot as plt
from make_all_figures import GENERATORS

cwd = Path.cwd()
before = mpl.rcParams.copy()
with patch('utils.blahut_arimoto.build_skill_data', side_effect=AssertionError('solve on import')), \\
     patch('utils.figure_cache.load_matching_cache', side_effect=AssertionError('cache on import')), \\
     patch.object(plt, 'figure', side_effect=AssertionError('figure on import')):
    for name in GENERATORS:
        assert callable(importlib.import_module(name).main)
assert Path.cwd() == cwd
assert dict(mpl.rcParams) == dict(before)
assert sorted(p.name for p in cwd.iterdir()) == ['pyproject.toml']
""",
        ],
        cwd=work,
        env=subprocess_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("mode", ["runner", "module"])
def test_figure_five_cli_from_unrelated_directory(tmp_path, mode):
    out = tmp_path / "output with spaces"
    command = (
        [sys.executable, str(ROOT / "make_all_figures.py"), "5"]
        if mode == "runner"
        else [sys.executable, "-m", "plotting_scripts.fig5_adaptation_geometry"]
    )
    result = subprocess.run(
        [*command, "--out", str(out)],
        cwd=tmp_path,
        env=subprocess_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    stem = "05_adaptation_geometry"
    assert (out / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
    with Image.open(out / f"{stem}.png") as image:
        assert min(image.size) > 100
        assert np.asarray(image.convert("RGB")).std() > 10
    manifest = json.loads((out / f"{stem}_plot_data.json").read_text())
    assert manifest


@pytest.mark.parametrize("number", ["oops", "99"])
def test_bad_figure_number_is_a_cli_error_without_output(tmp_path, number):
    out = tmp_path / "unused"
    result = subprocess.run(
        [sys.executable, str(ROOT / "make_all_figures.py"), number, "--out", str(out)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_runner_checks_every_output_of_a_shared_generator(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["make_all_figures.py", "9", "--out", str(tmp_path)]
    )

    def incomplete_generator(script, out_dir, **kwargs):
        for name in runner.GENERATORS[script]:
            if name.startswith("09_"):
                (out_dir / name).write_text("fresh")

    monkeypatch.setattr(runner, "run_generator", incomplete_generator)
    assert runner.main() == 1
    assert "10_kl_width_J_vs_bound" in capsys.readouterr().err


def test_runner_forwards_recompute_only_to_figure_four(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        runner.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    for name in runner.GENERATORS:
        runner.run_generator(name, tmp_path, recompute_fig4=True)
    for command in commands:
        assert ("--recompute-fig4" in command) == (
            command[2] == "plotting_scripts.fig4_bottleneck_states"
        )
        assert command[3:5] == ["--out", str(tmp_path)]


def test_gaussian_mixture_information_is_scale_invariant():
    from plotting_scripts.fig6_mi_information_collapse import skill_means, skill_sigma
    from utils.information import gaussian_mixture_information_mc

    means = skill_means(1.0, 6)
    sigma = skill_sigma(1.0, 6)
    information = gaussian_mixture_information_mc(means, sigma, 1000, seed=7)
    scaled = gaussian_mixture_information_mc(1e-3 * means, 1e-3 * sigma, 1000, seed=7)
    assert 0 < information < np.log(6)
    assert scaled == pytest.approx(information, abs=1e-12)
    # Indistinguishable conditional distributions carry no skill information.
    assert gaussian_mixture_information_mc(np.zeros((6, 2)), sigma, 1000) == pytest.approx(
        0, abs=1e-12
    )


def test_bottleneck_discounts_use_separate_fields_and_caches(tmp_path, monkeypatch):
    from environments import GridEnv
    from plotting_scripts import fig4_bottleneck_states as figure

    configs = [("first", GridEnv(rows=1, cols=2)), ("second", GridEnv(rows=1, cols=2))]
    monkeypatch.setattr(figure, "CACHE_PATH", tmp_path / "field.pkl")
    monkeypatch.setattr(figure, "bottleneck_configs", lambda: configs)
    fields = {}
    for gamma in figure.GAMMAS:
        result, _, recomputed = figure.compute_capacity_fields(configs, gamma)
        assert recomputed
        fields[gamma] = result["first"]["I_s0"]
    assert np.all(fields[0.95] > fields[0.5])
    assert figure.cache_key(configs, 0.5) != figure.cache_key(configs, 0.95)

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("Matching caches should avoid another solve")

    monkeypatch.setattr(figure, "build_skill_data", unexpected_solve)
    out = tmp_path / "figures"
    figure.main(out)
    for gamma in figure.GAMMAS:
        stem = f"04_bottleneck_states_gamma_{str(gamma).replace('.', 'p')}"
        record = json.loads((out / f"{stem}_plot_data.json").read_text())
        assert record["within_skill_gamma"] == gamma
        np.testing.assert_allclose(
            record["configurations"][0]["estimated_one_rollout_capacity_nats"], fields[gamma]
        )
        assert (out / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
        with Image.open(out / f"{stem}.png") as image:
            assert np.asarray(image.convert("RGB")).std() > 10


def test_contiguous_figure_uses_random_skills_and_writes_outputs(tmp_path, monkeypatch):
    from environments import GridEnv
    from plotting_scripts import fig7_contiguous_rollouts as figure

    configs = [("center", GridEnv(rows=1, cols=2)), ("corner", GridEnv(rows=1, cols=2))]
    monkeypatch.setattr(figure, "bottleneck_configs", lambda: configs)
    monkeypatch.setattr(figure, "N_DIRECTIONS", 12)
    assert runner.FIGURE_NUMBERS["07"] == "plotting_scripts.fig7_contiguous_rollouts"
    figure.main(tmp_path)
    stem = "07_bottleneck_temporal_empowerment"
    record = json.loads((tmp_path / f"{stem}_plot_data.json").read_text())
    assert record["search"] == "random"
    assert record["n_directions"] == 12
    assert record["source_solver"]["objective"] == "MI at each start, optimized separately"
    assert record["source_solver"]["method"] == "Adam"
    assert record["source_solver"]["restarts"] == 0
    assert len(record["panels"]) == 2
    assert all(np.all(np.array(panel["contiguous_mi_nats"]) > 0) for panel in record["panels"])
    assert (tmp_path / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
