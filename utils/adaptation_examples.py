"""Skill distributions for Figures 5, 9, and 10.

Each skill puts mass t on its own block and spreads the remaining mass
uniformly. One state per block gives Figure 5's absorbing-state example.
Equal blocks and a uniform skill prior make the mixture uniform too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .adaptation_bound import compute_J_exact, simplex_frame_J
from .information import kl_divergence, mutual_information


def block_family(n_states: int, n_blocks: int, commitment: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(K, n)`` skill rows and the uniform prior of the ``K``-block family."""
    # P_z(s) = (1-t)/n + (t/|B_z|) 1{s in B_z}; p(z)=1/K.
    # Equal-size disjoint blocks B_z make the mixture q(s)=1/n for every t.
    if n_states % n_blocks:
        raise ValueError("equal-size blocks need n_blocks to divide n_states")
    block = n_states // n_blocks
    skills = np.zeros((n_blocks, n_states))
    for z in range(n_blocks):
        skills[z, :] = (1 - commitment) / n_states
        skills[z, z * block : (z + 1) * block] += commitment / block
    return skills, np.ones(n_blocks) / n_blocks


@dataclass(frozen=True)
class SymmetricAbsorbingFamily:
    """Symmetric absorbing-state channel and I and J formulas."""

    n_absorbing: int  # K: number of absorbing states, also the number of skills.
    commitment: float  # t in P_z = (1-t)b + t delta_z.
    skill_distributions: np.ndarray  # P_z(s_+), shape (K,K), over absorbing states only.
    skill_prior: np.ndarray  # p(z)=1/K, shape (K,); capacity-achieving here by symmetry, not in general.
    baseline: np.ndarray  # b(s)=1/K, shape (K,); defines the Gaussian reward metric.
    mixture: np.ndarray  # q=sum_z p(z)P_z=b, shape (K,), for this symmetric family.
    information_nats: float  # I(Z;S_+)=capacity; computed from the specified channel.
    gaussian_width: float  # J=E_g max_z <P_z,g>; exact formula, evaluated numerically.
    whitened_radius: float  # ||(P_z-q)/sqrt(q)||_2 = t sqrt(K-1), identical for every z.


def symmetric_absorbing_information(n_absorbing: int, commitment: float) -> float:
    """Return the channel capacity in nats for K absorbing states.

    Each skill favors one state; symmetry makes the uniform prior optimal.
    """
    K = int(n_absorbing)
    t = float(commitment)
    if K < 2:
        raise ValueError("n_absorbing must be at least 2")
    if not 0.0 <= t <= 1.0:
        raise ValueError("commitment must lie in [0, 1]")
    p_hit = (1.0 + (K - 1.0) * t) / K
    p_miss = (1.0 - t) / K
    hit_term = p_hit * math.log(K * p_hit) if p_hit > 0.0 else 0.0
    miss_term = (K - 1.0) * p_miss * math.log(K * p_miss) if p_miss > 0.0 else 0.0
    return float(hit_term + miss_term)


def symmetric_absorbing_family(n_absorbing: int, commitment: float) -> SymmetricAbsorbingFamily:
    """Build the skill channel of a symmetric absorbing MDP and evaluate I and J."""
    K = int(n_absorbing)
    t = float(commitment)
    expected_information = symmetric_absorbing_information(K, t)
    skills, prior = block_family(K, K, t)
    baseline = prior.copy()

    mixture = prior @ skills
    information = mutual_information(skills, prior)
    gaussian_width = compute_J_exact(skills, prior, baseline)
    whitened_radius = t * math.sqrt(K - 1.0)
    expected_width = t * math.sqrt(2.0 / math.pi) if K == 2 else simplex_frame_J(K, whitened_radius)
    # Check normalization, q=b, and agreement with the symmetric-family I and J formulas.
    assert np.allclose(skills.sum(axis=1), 1.0, atol=1e-12), "Each skill's state probabilities must sum to one"
    assert np.allclose(mixture, baseline, atol=1e-12), "This symmetric family's skill mixture must equal its baseline"
    assert abs(kl_divergence(mixture, baseline)) < 1e-12, "Equal mixture and baseline distributions must have zero KL"
    assert math.isclose(information, expected_information, rel_tol=0.0, abs_tol=1e-12), (
        "Computed mutual information does not match this family's closed-form formula"
    )
    assert math.isclose(gaussian_width, expected_width, rel_tol=1e-10, abs_tol=1e-12), (
        "Computed Gaussian width does not match this family's closed-form formula"
    )

    return SymmetricAbsorbingFamily(
        n_absorbing=K,
        commitment=t,
        skill_distributions=skills,
        skill_prior=prior,
        baseline=baseline,
        mixture=mixture,
        information_nats=information,
        gaussian_width=gaussian_width,
        whitened_radius=whitened_radius,
    )
