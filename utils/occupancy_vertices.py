"""Discounted occupancy polytope: flow constraints, LP solutions, and skill channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cvxpy as cp
import numpy as np

from .information import EPS

LP_SOLVER = "CLARABEL"  # Goulart and Chen (2024).
LP_TOLERANCES = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}


@dataclass
class ConvexSkillData:
    # Discounted state-action occupancies, grouped by start state and skill.
    # Include the initial step; flatten each occupancy as rho[s * nA + a].
    vertices_by_s0: list[list[np.ndarray]]
    # State distributions after summing occupancies over actions.
    # Each start has an array of shape (skills, states), or None if unsolved.
    pz_s0: list[np.ndarray | None]
    # Optimized skill probabilities for each start state.
    p_prior: list[np.ndarray]
    # Estimated empowerment in nats per start state; zero for unsolved starts.
    I_s0: np.ndarray


def build_occupancy_constraints(P: np.ndarray, gamma: float, p_init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Aeq, beq) for discounted occupancy flow with ``beq = (1-gamma) p_init``."""
    # At each state, occupancy equals discounted incoming flow plus the
    # initial-state contribution. P contains the transition probabilities.
    nS, nA, _ = P.shape
    Aeq = np.zeros((nS, nS * nA), dtype=np.float64)
    for state in range(nS):
        Aeq[state, state * nA : (state + 1) * nA] = 1.0
        Aeq[state] -= gamma * P[:, :, state].reshape(-1)
    return Aeq, (1.0 - gamma) * p_init


def solve_lp(Aeq: np.ndarray, beq: np.ndarray, c: np.ndarray) -> np.ndarray | None:
    """Maximize c @ rho with flow constraints and nonnegative occupancy.

    Return None if the solver fails. A successful result may be approximate
    and need not be a vertex.
    """
    # c[s*nA+a] is the reward at (s,a); c @ rho is its normalized discounted return.
    rho = cp.Variable(c.shape[0])
    problem = cp.Problem(cp.Maximize(c @ rho), [Aeq @ rho == beq, rho >= 0])
    problem.solve(solver=LP_SOLVER, verbose=False, **LP_TOLERANCES)
    if problem.status not in ("optimal", "optimal_inaccurate") or rho.value is None:
        return None
    return np.asarray(rho.value, dtype=np.float64)


def normalize_rho(rho: np.ndarray) -> np.ndarray:
    total = float(np.sum(rho))
    if total <= EPS:
        return rho
    return rho / total


def state_occupancy(rho: np.ndarray, nS: int, nA: int) -> np.ndarray:
    """Marginalize ``rho(s, a)`` over actions."""
    return rho.reshape(nS, nA).sum(axis=1)


def extract_state_distributions(vertices: Sequence[np.ndarray], nS: int, nA: int) -> np.ndarray:
    """Sum each occupancy over actions and normalize it to a skill distribution P_z(s)."""
    if not vertices:
        return np.zeros((0, nS), dtype=np.float64)
    return np.stack([normalize_rho(state_occupancy(rho, nS, nA)) for rho in vertices], axis=0)


def occupancy_to_policy(rho: np.ndarray, nS: int, nA: int) -> np.ndarray:
    """Recover ``pi(a | s)`` from ``rho(s, a)``."""
    # Output shape (nS, nA): pi[s,a] = rho(s,a) / sum_{a'} rho(s,a').
    # Unvisited states keep all-zero rows; the occupancy does not specify their policy.
    rho_sa = rho.reshape(nS, nA)
    rho_s = rho_sa.sum(axis=1, keepdims=True)
    return rho_sa / np.where(rho_s > 0, rho_s, 1.0)
