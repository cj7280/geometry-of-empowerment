"""Whitened occupancy charts and exact KL level sets for Figure 5."""

import numpy as np


def width_chart(skill_dists, baseline_dist):
    """Center and scale skill distributions, then project onto two axes.

    Require at least two distinct skill rows and two states, with a positive
    normalized baseline. Return coordinates (K, 2), basis (2, n), and the
    mean distribution (n,). The projection is exact for rank at most two;
    with two states, the second axis is zero.
    """
    # Scale each skill's deviation from the baseline by sqrt(baseline).
    whitened_skill_offsets = (skill_dists / baseline_dist - 1.0) * np.sqrt(baseline_dist)
    # Centering preserves Gaussian width because the rewards have zero mean.
    centered = whitened_skill_offsets - whitened_skill_offsets.mean(0, keepdims=True)
    # Right singular vectors with nonzero singular values are orthogonal to sqrt(b).
    _, singular_values, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    basis = right_singular_vectors[:2].copy()
    if singular_values.shape[0] < 2 or singular_values[1] < 1e-8 * singular_values[0]:
        # Choose the second axis orthogonal to sqrt(b) and basis[0], or zero if none exists.
        root_b = np.sqrt(baseline_dist)
        candidates = []
        for unit in np.eye(centered.shape[1]):
            candidate = unit - (unit @ root_b) * root_b / (root_b @ root_b)
            candidates.append(candidate - (candidate @ basis[0]) * basis[0])
        norms = np.array([np.linalg.norm(candidate) for candidate in candidates])
        best = int(np.argmax(norms))
        basis[1] = candidates[best] / norms[best] if norms[best] > 1e-10 else 0.0
    coords = centered @ basis.T  # (K,2): coordinates of (P_z-mean_dist)/sqrt(b).
    mean_dist = skill_dists.mean(0)  # Arithmetic mean of skill rows; equals q for a uniform prior.
    exact = singular_values.shape[0] < 3 or singular_values[2] < 1e-8 * singular_values[0]
    if exact:
        # Verify reconstruction when the centered rank is at most two within tolerance.
        reconstructed_skills = mean_dist[None, :] + np.sqrt(baseline_dist) * (coords @ basis)
        assert np.allclose(reconstructed_skills, skill_dists, atol=1e-10), (
            "Converting chart coordinates back to probabilities did not recover the input skills"
        )
    return coords, basis, mean_dist


def align2d(E, first_deg, second_deg=None):
    """Return a rotation placing skill 0 at first_deg.

    If needed, reflect to place skill 1 within five degrees of second_deg.
    Apply the result as E @ M.T; distances stay unchanged.
    """
    t = np.radians(first_deg) - np.arctan2(E[0, 1], E[0, 0])
    M = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    if second_deg is not None:
        a1 = np.degrees(np.arctan2(*(E[1] @ M.T)[::-1])) % 360
        if min(abs(a1 - second_deg), 360 - abs(a1 - second_deg)) > 5.0:
            phi = np.radians(first_deg)
            M = np.array([[np.cos(2 * phi), np.sin(2 * phi)], [np.sin(2 * phi), -np.cos(2 * phi)]]) @ M
    return M


def chart_kl(points, q, basis):
    """Convert chart points back to probabilities and compute KL divergence to q.

    Require a positive normalized q and a basis orthogonal to sqrt(q).
    Return NaN for probabilities below -1e-9; clip smaller roundoff errors.
    """
    # Undo the projection and scaling to recover one distribution per point.
    distributions = q[None, :] + np.sqrt(q)[None, :] * (np.atleast_2d(points) @ basis)
    invalid = distributions.min(axis=1) < -1e-9
    clipped = np.clip(distributions, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(clipped > 0.0, clipped * np.log(clipped / q[None, :]), 0.0)
    values = terms.sum(axis=1)
    values[invalid] = np.nan
    return values


def kl_level_set(q, basis, level, n_angles=2048):
    """Trace a constant-KL contour by searching outward from q.

    Use chart_kl's input requirements and a positive level. Raise if the contour
    reaches the probability simplex boundary. Return n_angles points in 2-D.
    """
    theta = 2.0 * np.pi * np.arange(n_angles) / n_angles
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    # A ray reaches the simplex boundary when any state probability first reaches zero.
    probability_delta_per_radius = np.sqrt(q)[None, :] * (directions @ basis)
    boundary_candidates = np.full_like(probability_delta_per_radius, np.inf)
    np.divide(
        -q[None, :],
        probability_delta_per_radius,
        out=boundary_candidates,
        where=probability_delta_per_radius < -1e-14,
    )
    feasible_radii = np.min(boundary_candidates, axis=1)
    if not np.all(np.isfinite(feasible_radii)):
        raise ValueError("every chart direction must meet the probability-simplex boundary")
    if float(np.nanmin(chart_kl(feasible_radii[:, None] * directions, q, basis))) <= level:
        raise ValueError("the KL level reaches the simplex boundary; no closed contour exists")

    lower = np.zeros(n_angles)
    upper = feasible_radii.copy()
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        above = chart_kl(middle[:, None] * directions, q, basis) >= level
        upper[above] = middle[above]
        lower[~above] = middle[~above]
    return 0.5 * (lower + upper)[:, None] * directions
