"""Gaussian width formulas, Monte Carlo estimates, and the KL-to-width bound."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate
from scipy.special import ndtr

from .information import EPS

# Each row of Pz gives one skill's probabilities over states.
# prior gives the skill probabilities; skills with prior <= EPS are ignored.
# baseline is a positive reference distribution that sets the reward scale
# at each state.
# J is the average reward of the best skill across random Gaussian rewards.

# Smallest c with log(1 + x) <= c * sqrt(x) for all x >= 0; the maximizer solves 2x = (1 + x) log(1 + x).
KL_SQRT_CHI2_CONSTANT = 0.8047423425494119


def _sample_gaussian_l2b(reference, count, rng):
    """Sample Gaussian rewards, scaled by a positive reference distribution.

    Each reward has zero reference-weighted mean. Return one row per sample.
    """
    # Scale independent normal draws using the reference probabilities.
    raw = rng.standard_normal((count, len(reference))) / np.sqrt(reference)
    # Remove each reward's reference-weighted mean.
    raw -= np.sum(reference * raw, axis=1, keepdims=True)
    return raw


def gaussian_max_mean(skill_count: int) -> float:
    """Return the closed form expected maximum of K independent N(0, 1) numbers, K=skill_count.

    These are auxiliary Gaussian coordinates for the regular-simplex formula.
    simplex_frame_J converts this expectation into the expected best skill reward.
    """
    # One draw has mean zero. Empty input also returns zero by convention.
    if skill_count <= 1:
        return 0.0

    def integrand(value):
        normal_pdf = math.exp(-(value**2) / 2.0) / math.sqrt(2.0 * math.pi)
        normal_cdf = ndtr(value)  # Probability that one draw is at most value.
        # All K draws are at most value with probability normal_cdf**K.
        # Differentiate that probability to get the maximum's density.
        maximum_pdf = skill_count * normal_pdf * normal_cdf ** (skill_count - 1)
        return value * maximum_pdf  # Expectation = integral of value times density.

    # The full integral runs from -infinity to infinity. We truncate at twelve
    # standard deviations; the omitted tails are negligible for our skill counts.
    # quad returns (estimate, error); limit=200 caps adaptive subdivisions.
    mean, _ = integrate.quad(integrand, -12, 12, limit=200)
    return float(mean)


def _simplex_frame_params(gram, rank_tol=1e-10):
    """Return (K, radius) for a centered regular simplex with K >= 3 within tolerance, else None."""
    skill_count = gram.shape[0]
    if skill_count < 3:
        return None
    diagonal = np.diag(gram)
    scale = max(float(diagonal.max()), EPS)
    off_diagonal = gram[~np.eye(skill_count, dtype=bool)]
    if (
        np.ptp(diagonal) > rank_tol * scale * skill_count
        or np.ptp(off_diagonal) > rank_tol * scale * skill_count
        or np.max(np.abs(gram.sum(axis=1))) > rank_tol * scale * skill_count
    ):
        return None
    return skill_count, float(math.sqrt(max(diagonal[0], 0.0)))


def simplex_frame_J(skill_count: int, radius: float = 1.0) -> float:
    """Return J for K >= 2 centered regular-simplex skills; radius is in whitened coordinates."""
    # In general, maximization is over a continuous skill space. For our symmetric channel
    # MDP examples, considering the K vertex skills with discrete uniform prior loses no generality.
    # Let g contain K independent N(0, 1) numbers, one per skill.
    # The correlated skill returns have the same distribution as
    # radius * sqrt(K/(K-1)) * (g_i - mean(g)).
    # Subtracting mean(g) shifts every return equally and has expectation zero,
    # so J is the scale factor times the expected maximum computed above.
    return float(radius * math.sqrt(skill_count / (skill_count - 1.0)) * gaussian_max_mean(skill_count))


def compute_J(Pz, prior, baseline, n_samples=200_000, seed=42):
    """Estimate ``J = E_g max_z <P_z, g>`` by Gaussian Monte Carlo."""
    # Score every retained skill on the same sampled rewards.
    samples = _sample_gaussian_l2b(baseline, n_samples, np.random.default_rng(seed))
    values = Pz[prior > EPS] @ samples.T
    return float(np.mean(np.max(values, axis=0)))


def compute_J_exact(Pz, prior, baseline, rank_tol=1e-10):
    """Compute J using formulas for a line, a plane, or a regular simplex.

    Measure rank after scaling skill offsets by sqrt(baseline), using rank_tol.
    Other cases use compute_J with its default sample count and fixed seed.
    """
    active = Pz[prior > EPS]
    if len(active) == 0:
        return 0.0
    # u_z = (P_z-b)/sqrt(b); <u_z,u_w> = <P_z/b-1, P_w/b-1>_b.
    whitened = (active / baseline - 1.0) * np.sqrt(baseline)
    # gram[z,w] = <u_z,u_w>; keep only eigenvalues above the relative tolerance.
    gram = whitened @ whitened.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(eigenvalues[-1], EPS)
    keep = eigenvalues > rank_tol * scale
    rank = int(np.sum(keep))
    if rank == 0:
        return 0.0
    coordinates = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])  # u_z in the retained eigenbasis: (Z_active,rank).
    if rank == 1:
        # A segment of length L has Gaussian width L/sqrt(2 pi).
        line_coordinates = coordinates[:, 0]
        return float((np.max(line_coordinates) - np.min(line_coordinates)) / math.sqrt(2.0 * math.pi))
    if rank > 2:
        frame = _simplex_frame_params(gram, rank_tol)
        if frame is not None:
            return simplex_frame_J(*frame)
        return compute_J(Pz, prior, baseline)

    radii = np.linalg.norm(coordinates, axis=1)
    angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    boundaries = set()

    # Find reward directions where two skills tie and the winning skill might change.
    for first in range(len(coordinates)):
        for second in range(first + 1, len(coordinates)):
            difference = coordinates[first] - coordinates[second]
            if np.linalg.norm(difference) < rank_tol * np.sqrt(scale):
                continue
            angle = math.atan2(difference[1], difference[0])
            boundaries.add((angle + math.pi / 2.0) % (2.0 * math.pi))
            boundaries.add((angle - math.pi / 2.0) % (2.0 * math.pi))
    ordered = sorted(boundaries) or [0.0]
    ordered.append(ordered[0] + 2.0 * math.pi)
    support_integral = 0.0
    for low, high in zip(ordered[:-1], ordered[1:]):
        # One point wins the whole arc; its support radius * cos(t - angle) integrates to a sine difference.
        midpoint = (low + high) / 2.0
        winner = int(np.argmax(coordinates @ np.array([math.cos(midpoint), math.sin(midpoint)])))
        support_integral += radii[winner] * (math.sin(high - angles[winner]) - math.sin(low - angles[winner]))
    return float(math.sqrt(math.pi / 2.0) * support_integral / (2.0 * math.pi))
