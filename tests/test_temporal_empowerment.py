"""Contiguous-rollout calculations: analytic values and the distance identity."""

import numpy as np
import pytest

from utils.temporal_empowerment import successor_occupancy, temporal_empowerment, solve_reactive_sources


@pytest.mark.parametrize("beta", [0.2, 0.5, 0.9])
def test_two_state_closed_form_and_consistent_mixture(beta):
    libraries = [np.eye(2), np.eye(2)]
    sources = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    values, occupancy, conditioned = temporal_empowerment(libraries, sources, beta)
    first = np.array([1 - beta**2 / 2, beta**2 / 2])
    second = np.array([1 - beta + beta**2 / 2, beta - beta**2 / 2])
    marginal = (first + second) / 2
    exact = sum(np.sum(row * np.log(row / marginal)) for row in (first, second)) / 2
    np.testing.assert_allclose(values, exact, atol=1e-12)
    for s, rows in enumerate(conditioned):
        np.testing.assert_allclose(sources[s] @ rows, occupancy[s], atol=1e-12)
        distance = -np.log(occupancy[s] / np.diag(occupancy))
        conditioned_distance = -np.log(rows / np.diag(occupancy))
        expected_reduction = sources[s] @ np.sum(rows * (distance - conditioned_distance), axis=1)
        assert expected_reduction == pytest.approx(values[s], abs=1e-12)


def test_successor_matches_rollout_series():
    kernel = np.array([[0.7, 0.3], [0.4, 0.6]])
    beta = 0.8
    series = (1-beta) * sum(beta**k * np.linalg.matrix_power(kernel, k) for k in range(200))
    np.testing.assert_allclose(successor_occupancy(kernel, beta), series, atol=1e-12)


@pytest.mark.parametrize("start", [0, 1])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_source_gradient_includes_later_rollouts(start, epsilon):
    from scipy.optimize._numdiff import approx_derivative
    from scipy.special import softmax
    from utils.temporal_empowerment import _start_information_gradient

    libraries = [np.array([[0.9, 0.1], [0.2, 0.8]]), np.eye(2)]
    logits = np.array([0.2, -0.4, 0.7, -0.1])
    base = [softmax(row) for row in logits.reshape(2, 2)]
    sources = [(1-epsilon) * p + epsilon / len(p) for p in base]
    value, gradients = _start_information_gradient(libraries, sources, 0.3, start)
    analytic = (1-epsilon) * np.concatenate([p * (g - p @ g) for p, g in zip(base, gradients)])

    def evaluate(x):
        sources = [(1-epsilon) * softmax(row) + epsilon / len(row) for row in x.reshape(2, 2)]
        return temporal_empowerment(libraries, sources, 0.3)[0][start]

    assert value == pytest.approx(evaluate(logits), abs=1e-12)
    np.testing.assert_allclose(analytic, approx_derivative(evaluate, logits).ravel(), atol=1e-9)
    assert np.linalg.norm(analytic.reshape(2, 2)[1-start]) > 1e-5


def test_each_start_gets_its_own_maximizing_source():
    from scipy.optimize import differential_evolution

    libraries = [np.array([[0.9, 0.1], [0.2, 0.8]]), np.eye(2)]
    sources = [np.array([0.8, 0.2]), np.array([0.3, 0.7])]
    initial, _, _ = temporal_empowerment(libraries, sources, 0.3)
    optimized_sources = []
    for start in range(2):
        value, optimized = solve_reactive_sources(libraries, sources, start_state=start)
        measured = temporal_empowerment(libraries, optimized, 0.3)[0][start]
        assert value == pytest.approx(measured, abs=1e-12)
        assert value >= initial[start]
        # Independently optimize the two probabilities, including simplex boundaries.
        def objective(x):
            p = [np.array([v, 1-v]) for v in x]
            return -temporal_empowerment(libraries, p, 0.3)[0][start]
        reference = differential_evolution(objective, [(0, 1)] * 2, seed=17, tol=1e-10)
        assert value == pytest.approx(-reference.fun, abs=2e-7)
        optimized_sources.append(optimized)
    assert not np.allclose(optimized_sources[0], optimized_sources[1])
