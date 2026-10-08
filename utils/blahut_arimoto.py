"""Random occupancy search and Blahut-Arimoto optimization of skill probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .information import EPS, kl_divergence
from .occupancy_vertices import (
    ConvexSkillData,
    build_occupancy_constraints,
    extract_state_distributions,
    normalize_rho,
    solve_lp,
    state_occupancy,
)


BA_TOL = 1e-6  # Stop once max_z D_KL(P_z || q) - I(Z; X) <= BA_TOL ...
BA_MAX_ITERS = 50_000  # ... or after this many updates; convergence is not checked.


@dataclass
class BAResult:
    # For the supplied finite channel P_z(x), let q(x) = sum_z prior[z] P_z(x).
    prior: np.ndarray  # Shape (Z,): current p(z), optimized by BA.
    mi: float  # I(Z; X) = sum_z prior[z] D_KL(P_z || q), in nats.
    upper_bound: float  # max_z D_KL(P_z || q): upper bound on this channel's capacity.
    gap: float  # upper_bound - mi, in nats: bounds this finite channel's capacity error.


def blahut_arimoto(channel: np.ndarray, *, tol: float = BA_TOL, max_iters: int = BA_MAX_ITERS) -> BAResult:
    """Find the capacity-achieving skill probabilities of a fixed (Z, X) channel.

    Iterate until the capacity gap reaches tol or after max_iters updates.

    Return the probabilities, mutual information, capacity upper bound, and gap.
    """
    # channel[z,x] = P_z(x) = p(X=x | Z=z), shape (Z, X).
    # For the grid experiments, X=S_+ and the entire solve is conditioned on one s0.
    channel = np.asarray(channel, dtype=np.float64)
    if channel.ndim != 2:
        raise ValueError("channel must have shape (Z, X)")
    if channel.shape[0] == 0:
        return BAResult(np.zeros(0), 0.0, 0.0, 0.0)
    channel = np.clip(channel, 0.0, None)
    channel = channel / np.maximum(channel.sum(axis=1, keepdims=True), EPS)

    prior = np.full(channel.shape[0], 1.0 / channel.shape[0])
    iteration = 0
    while True:
        # prior @ channel is q(x), the skill-marginalized output distribution (X,).
        # D_z = D_KL(P_z || mixture) are the KKT scores; max_z D_z - I bounds the gap.
        scores = kl_divergence(channel, prior @ channel)
        mi, upper = float(prior @ scores), float(scores.max())
        if upper - mi <= tol or iteration >= max_iters:
            return BAResult(prior, mi, upper, upper - mi)
        # BA update: p_new(z) is proportional to p(z) exp(D_z); subtract upper for stability.
        prior = prior * np.exp(scores - upper)
        prior /= prior.sum()
        iteration += 1


def find_vertices_random(P, gamma, p_init, *, n_directions, seed=0):
    """Sample reward directions uniformly on the state simplex, then run BA once.

    For a generic state reward, the occupancy LP's maximizer is a vertex of the
    state-occupancy polytope, so each direction gives a candidate skill; distinct
    state distributions are kept. The search can miss vertices, so it is approximate.
    """
    if n_directions < 1:
        raise ValueError("n_directions must be positive")
    nS, nA = P.shape[:2]
    Aeq, beq = build_occupancy_constraints(P, gamma, p_init)
    rng = np.random.default_rng(seed)
    vertices, keys = [], set()
    for _ in range(n_directions):
        direction = rng.dirichlet(np.ones(nS))
        rho = solve_lp(Aeq, beq, np.repeat(direction, nA))
        if rho is None:
            continue
        rho = np.clip(rho, 0.0, None)
        key = np.round(normalize_rho(state_occupancy(rho, nS, nA)), decimals=6).tobytes()
        if key not in keys:
            keys.add(key)
            vertices.append(rho)
    if not vertices:
        raise RuntimeError("No feasible occupancy found by fixed random search")
    ba = blahut_arimoto(extract_state_distributions(vertices, nS, nA))
    return vertices, ba


def build_skill_data(
    env,
    gamma: float,
    *,
    start_states: Sequence[int] | None = None,
    seed: int = 0,
    n_directions: int = 500,
) -> ConvexSkillData:
    """Find skills and optimize their source separately for each start state."""
    # Solve each start with p_init=delta_{s0}.
    nS, nA = int(env.nS), int(env.nA)
    starts = list(range(nS)) if start_states is None else [int(s) for s in start_states]

    vertices_by_s0: list[list[np.ndarray]] = [[] for _ in range(nS)]  # rho_z(s,a | s0).
    pz_s0: list[np.ndarray | None] = [None] * nS  # P_z(s | s0), shape (Z_s0, nS) per solved start.
    p_prior = [np.zeros(0, dtype=np.float64) for _ in range(nS)]  # p(z | s0), shape (Z_s0,).
    I_s0 = np.zeros(nS, dtype=np.float64)  # Estimated empowerment E_hat(s0), nats; unsolved entries stay 0.
    for s0 in starts:
        vertices, ba = find_vertices_random(
            np.asarray(env.P), gamma, np.eye(nS)[s0], seed=seed + s0, n_directions=n_directions
        )
        vertices_by_s0[s0] = vertices
        pz_s0[s0] = extract_state_distributions(vertices, nS, nA)
        p_prior[s0] = ba.prior
        I_s0[s0] = ba.mi

    return ConvexSkillData(vertices_by_s0, pz_s0, p_prior, I_s0)
