"""Q-learning on statewise mutual information and tabular rollouts."""

from __future__ import annotations

import numpy as np

from .information import EPS

# Tabular Q-learning; exact value iteration would also do (for Figure 2 it gives the same greedy policy).

def qlearn_potential_policy(
    env,
    I_s0: np.ndarray,
    gamma: float,
    p0: np.ndarray,
    n_iters: int,
    alpha: float,
    eps_start: float,
    eps_end: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return the greedy policy of tabular Q-learning on the reward ``r(s) = (1-gamma) I_s0[s]``."""
    # Train a policy to maximize discounted empowerment along its trajectory.
    nS, nA = int(env.nS), int(env.nA)
    rng = np.random.default_rng(seed)
    # Q[s,a] estimates the optimal discounted return after taking action a at s.
    Q = np.zeros((nS, nA), dtype=np.float32)
    # The (1-gamma) factor makes return equal an occupancy-weighted mean of I(s).
    r = (1.0 - gamma) * np.asarray(I_s0, dtype=np.float32)

    p0 = np.asarray(p0, dtype=np.float32).reshape(-1)  # Pr(S_0=s), shape (nS,).
    p0 = p0 / (float(np.sum(p0)) + EPS)
    s = int(rng.choice(nS, p=p0))

    for t in range(n_iters):
        frac = min(1.0, t / max(1, n_iters - 1))
        eps_t = eps_start + frac * (eps_end - eps_start) # epsilon-greedy
        if rng.random() < eps_t:
            a = int(rng.integers(nA))
        else:
            a = int(np.argmax(Q[s]))

        p_next = np.asarray(env.P[s, a], dtype=np.float64)
        p_next = p_next / (float(np.sum(p_next)) + EPS)
        s_next = int(rng.choice(nS, p=p_next))

        td_target = r[s] + gamma * np.max(Q[s_next])
        Q[s, a] += alpha * (td_target - Q[s, a])
        s = s_next

    pi = np.zeros((nS, nA), dtype=np.float32)  # pi_pot(a | s), one-hot greedy action per state.
    pi[np.arange(nS), np.argmax(Q, axis=1)] = 1.0
    return pi


def greedy_rollout(env, pi: np.ndarray, s0: int, steps: int) -> list[int]:
    """Follow each state's argmax action and most likely successor."""
    traj = [int(s0)]
    P = np.asarray(env.P)
    pi = np.asarray(pi)
    for _ in range(int(steps)):
        a = int(np.argmax(pi[traj[-1]]))
        traj.append(int(np.argmax(P[traj[-1], a])))
    return traj


def sample_rollout(env, pi: np.ndarray, s0: int, steps: int, rng: np.random.Generator) -> list[int]:
    """Sample actions from ``pi`` and successors from ``env.P``."""
    # Return steps + 1 states, including the initial state; the horizon is fixed.
    traj = [int(s0)]
    P = np.asarray(env.P)
    pi = np.asarray(pi)
    for _ in range(int(steps)):
        probs = np.asarray(pi[traj[-1]], dtype=np.float64)
        probs = probs / (float(np.sum(probs)) + EPS)
        a = int(rng.choice(len(probs), p=probs))
        next_probs = np.asarray(P[traj[-1], a], dtype=np.float64)
        next_probs = next_probs / (float(np.sum(next_probs)) + EPS)
        traj.append(int(rng.choice(len(next_probs), p=next_probs)))
    return traj
