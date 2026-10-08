"""Release-path tests for solver diagnostics and the Figure 4 cache."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from utils.blahut_arimoto import blahut_arimoto
from utils.figure_cache import (
    FIGURE_CACHE_SCHEMA_VERSION,
    load_matching_cache,
    write_matching_cache,
)


def test_ba_gap_certifies_the_restricted_finite_channel():
    crossover = 0.13
    channel = np.array(
        [
            [1.0 - crossover, crossover],
            [crossover, 1.0 - crossover],
        ]
    )
    result = blahut_arimoto(channel, tol=1e-13)
    entropy = -crossover * math.log(crossover) - (1.0 - crossover) * math.log(
        1.0 - crossover
    )
    exact_capacity = math.log(2.0) - entropy

    assert result.mi <= exact_capacity + 1e-12
    assert exact_capacity <= result.upper_bound + 1e-12
    assert result.gap <= 1e-12


def test_figure_cache_fails_closed_and_recompute_bypasses_stale_data(tmp_path):
    path = tmp_path / "field.pkl"
    write_matching_cache(path, "current", {"value": 3})
    assert load_matching_cache(path, "current") == {"value": 3}

    with pytest.raises(RuntimeError, match="--recompute-fig4"):
        load_matching_cache(path, "changed")
    assert load_matching_cache(path, "changed", recompute=True) is None

    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": FIGURE_CACHE_SCHEMA_VERSION - 1,
                "key": "current",
                "results": {},
            },
            handle,
        )
    with pytest.raises(RuntimeError, match="stale figure cache"):
        load_matching_cache(path, "current")


def test_fixed_random_search_uses_requested_directions_then_ba_once(monkeypatch):
    from environments import GridEnv
    from utils import blahut_arimoto as solver

    env = GridEnv(rows=1, cols=2)
    objectives, ba_calls = [], []
    solve_lp, ba = solver.solve_lp, solver.blahut_arimoto

    def record_lp(Aeq, beq, c):
        objectives.append(c.reshape(env.nS, env.nA).copy())
        return solve_lp(Aeq, beq, c)

    def record_ba(channel, **kwargs):
        result = ba(channel, **kwargs)
        ba_calls.append(result)
        return result

    monkeypatch.setattr(solver, "solve_lp", record_lp)
    monkeypatch.setattr(solver, "blahut_arimoto", record_ba)
    data = solver.build_skill_data(
        env, 0.8, start_states=[1], seed=7, n_directions=12
    )
    assert len(objectives) == 12
    assert len(ba_calls) == 1
    expected = np.random.default_rng(8).dirichlet(np.ones(env.nS), size=12)
    np.testing.assert_allclose(np.array(objectives)[:, :, 0], expected)
    for coefficients in objectives:
        np.testing.assert_allclose(coefficients, np.repeat(coefficients[:, :1], env.nA, axis=1))
    np.testing.assert_allclose(data.pz_s0[1].sum(axis=1), 1.0)
    assert data.I_s0[1] > 0
    assert ba_calls[0].gap <= 1e-6
