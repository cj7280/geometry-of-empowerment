"""KL divergence and mutual information in nats: exact for finite channels, Monte Carlo for Gaussian skills."""

import numpy as np
from scipy.special import logsumexp

# Numerical floor for normalizations and for ignoring skills with negligible prior.
EPS = 1e-12


def kl_divergence(p, q):
    """Return KL in nats along the last axis; p=0 contributes 0, and p>0 with q=0 gives inf."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * (np.log(p) - np.log(q)), 0.0)
    return terms.sum(axis=-1)


def mutual_information(channel, prior):
    """Return ``I(Z; X)`` for skill rows ``channel[z] = p(x | z)`` under ``prior``."""
    # Inputs are normalized probabilities:
    # channel:(Z,X), prior=p(z):(Z,).
    channel = np.asarray(channel, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    active = prior > 0.0
    mixture = prior @ channel  # q(x) = sum_z p(z) p(x | z), shape (X,).
    return float(prior[active] @ kl_divergence(channel[active], mixture)) # I(Z;X) = sum_z p(z) D_KL(P_z || q)


def gaussian_mixture_information_mc(means, sigma, n_samples, seed=0):
    """Estimate I(Z; X) for uniform Z and X | Z=z ~ N(means[z], sigma^2 I) by Monte Carlo."""
    # I = (1/K) sum_z E_{x~P_z}[log P_z(x) - log q(x)] with q = (1/K) sum_z P_z.
    rng = np.random.default_rng(seed)
    n_skills, dim = means.shape
    terms = []
    for skill in range(n_skills):
        samples = means[skill] + sigma * rng.standard_normal((n_samples, dim))
        squared_distance = ((samples[:, None, :] - means[None, :, :]) ** 2).sum(axis=-1)
        # Shape (n_samples, K): log P_z(x) up to the shared Gaussian normalizer, which cancels.
        log_likelihood = -squared_distance / (2 * sigma**2)
        terms.append(
            (
                log_likelihood[:, skill]
                - (logsumexp(log_likelihood, axis=1) - np.log(n_skills))
            ).mean()
        )
    return float(np.mean(terms))
