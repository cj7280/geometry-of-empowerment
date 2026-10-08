"""Information about the current skill after a sequence of skill rollouts."""

import numpy as np

from .information import mutual_information


def successor_occupancy(kernel, beta):
    """Distribution after a geometrically sampled number of whole rollouts.

    beta is the probability of continuing to another rollout. It is distinct
    from gamma, which determines the length of each individual rollout.
    """
    if not 0 < beta < 1:
        raise ValueError("beta must lie strictly inside (0, 1)")
    identity = np.eye(len(kernel))
    occupancy = (1 - beta) * np.linalg.solve(identity - beta * kernel, identity)
    occupancy = np.clip(occupancy, 0, None)
    return occupancy / occupancy.sum(axis=1, keepdims=True)


def skill_conditioned_successor(channel, occupancy, state, beta):
    """Fix the first skill, then follow the current skill probabilities thereafter."""
    rows = beta * (channel @ occupancy)
    # With probability 1-beta, the sampled sequence ends before any rollout.
    rows[:, state] += 1 - beta
    return rows


def temporal_empowerment(libraries, sources, beta):
    """Measure effective contiguous empowerment I(Z; G | X_0=s) at each start s.

    Z is the first skill.
    G is the state observed after a geometrically sampled
    number of skill rollouts. Evaluate given p(z | s).

    Return (values, occupancy, conditioned), defined beside their calculations.
    """
    # libraries[s][z, g] = p^pi(g | s,z): where one skill rollout terminates.
    # sources[s][z] = p(z | s): probability of selecting that skill at state s.
    if len(libraries) != len(sources) or not libraries:
        raise ValueError("provide one skill library and source per state")
    for channel, source in zip(libraries, sources):
        if (channel.ndim != 2 or channel.shape[1] != len(libraries)
                or source.shape != (len(channel),) or np.any(channel < 0)
                or np.any(source < 0) or not np.allclose(channel.sum(axis=1), 1)
                or not np.isclose(source.sum(), 1)):
            raise ValueError("skill rows and sources must be probability distributions")

    # kernel[s, g] = sum_z p(z | s) p^pi(g | s,z).
    # Probability of terminating at g after ONE rollout starting at s.
    kernel = np.stack([p @ channel for p, channel in zip(sources, libraries)])

    # occupancy[s, g] = M_beta^pi(g | s): probability of observing g from start s.
    # Observe after N rollouts, with P(N=k) = (1-beta) * beta**k; N=0 observes s.
    # Each new skill is drawn from p(z | the state where the previous one ended).
    # gamma sets each rollout's duration and is already included in libraries.
    occupancy = successor_occupancy(kernel, beta)

    # conditioned[s][z, g] = M_beta^pi(g | s,z): the same observation, given Z=z.
    # Fix only the first skill. Later skills still use p(z | current state),
    # so different first skills can lead to different later skill choices.
    conditioned = [skill_conditioned_successor(c, occupancy, s, beta)
                   for s, c in enumerate(libraries)]

    # values[s] = sum_z p(z | s) KL[M_beta^pi(. | s,z) || M_beta^pi(. | s)].
    # This is I(Z; G | X_0=s), in nats: how much G tells us about the first skill.
    # Identical conditional distributions give zero; distinguishable ones give MI.
    values = np.array([mutual_information(c, p) for c, p in zip(conditioned, sources)])
    return values, occupancy, conditioned


def _start_information_gradient(libraries, sources, beta, start_state):
    """Differentiate I(Z; G | X_0=start_state) with respect to p(z | s) everywhere."""
    # libraries[s][z, g] = probability that skill z terminates at g, starting at s.
    # These skill policies stay fixed, and found via vertex search.
    # sources[s][z] = p(z | s) is learned.
    # kernel[s, g] averages those termination probabilities over the chosen skill.

    kernel = np.stack([p @ c for p, c in zip(sources, libraries)])

    # occupancy[s, g] = M_beta^pi(g | s): probability of observing g from start s.
    # Observe after N rollouts, with P(N=k) = (1-beta) * beta**k, including N=0.
    # At each rollout, select a skill using p(z | the state reached so far).
    occupancy = successor_occupancy(kernel, beta)
    channel = libraries[start_state]  # One-rollout termination probabilities from s_0.
    prior = sources[start_state]  # Current initial-skill probabilities p(z | s_0).

    # rows[z, g] = M_beta^pi(g | s_0, z): probability of observing g given first skill z.
    # Use the same observation time N; choose later skills using p(z | current state).
    rows = skill_conditioned_successor(channel, occupancy, start_state, beta)
    mixture = prior @ rows  # M_beta^pi(g | s_0), averaging over initial skills.

    # Objective: E_z KL[M_beta^pi(. | s_0,z) || M_beta^pi(. | s_0)].
    # Equivalently, the expected reduction in temporal distance from choosing z (see paper)
    log_ratio = np.log(np.maximum(rows, 1e-300)) - np.log(np.maximum(mixture, 1e-300))
    scores = np.sum(rows * log_ratio, axis=1)  # One KL per initial skill.

    # Calculating exact gradients:

    # FUTURE SELECTION: how does changing later skill probabilities affect MI?
    # E.g., staying in the room chosen by Z preserves information; leaving may erase it.
    occupancy_gradient = beta * channel.T @ (prior[:, None] * log_ratio)
    # This includes all later rollouts: dM = beta/(1-beta) M dK M.
    kernel_gradient = beta / (1-beta) * occupancy.T @ occupancy_gradient @ occupancy.T
    gradients = [c @ row for c, row in zip(libraries, kernel_gradient)]

    # CURRENT SELECTION: also differentiate the weights p(z | s_0) in the MI average.
    gradients[start_state] += scores - 1
    return float(prior @ scores), gradients  # I(Z; G | X_0=s_0), dI/dp(z | s).


def solve_reactive_sources(libraries, initial_sources, beta=0.3, *, start_state,
                           max_iterations=10_000, anneal_iterations=8000,
                           uniform_weight=0.1, learning_rate=0.05):
    """Maximize contiguous empowerment at s_0 = start_state:

        max_{p(z | s), all s} I(Z; G | X_0=s_0).

    Equivalently, maximize expected temporal-distance reduction from choosing Z.
    Local search over sources for fixed skills; no global-optimum guarantee.
    """

    if not 0 <= start_state < len(libraries):
        raise ValueError("start_state must index the skill libraries")
    if not 0 <= anneal_iterations < max_iterations:
        raise ValueError("anneal_iterations must be nonnegative and below max_iterations")
    if not 0 <= uniform_weight < 1 or learning_rate <= 0:
        raise ValueError("require 0 <= uniform_weight < 1 and learning_rate > 0")

    # Figure 7 run supplies single-rollout BA probabilities. Score their CONTIGUOUS MI.
    initial_values, _, _ = temporal_empowerment(libraries, initial_sources, beta)
    best_value = float(initial_values[start_state])  # Best contiguous MI at s_0 so far.
    best_probabilities = np.concatenate(initial_sources)

    sizes = np.array([len(p) for p in initial_sources])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    state_ids = np.repeat(np.arange(len(sizes)), sizes)
    uniform = 1.0 / sizes[state_ids]

    # Learn p(z | s) at every state by gradient ascent on the start state's MI.
    logits = np.zeros(sizes.sum())  # Learn p_theta(z | s), starting uniformly.
    first_moment = np.zeros_like(logits)
    second_moment = np.zeros_like(logits)

    for step in range(max_iterations + 1):
        # Each state's softmax gives p(z | s).
        weights = np.exp(logits - np.maximum.reduceat(logits, starts)[state_ids])
        base = weights / np.add.reduceat(weights, starts)[state_ids]

        # Uniform mixing keeps every first skill in play early; it decays to zero by anneal_iterations.
        epsilon = (uniform_weight * max(0.0, 1 - step / anneal_iterations)
                   if anneal_iterations else 0.0)
        # Every skill gets at least epsilon / (number of skills at its state).
        probabilities = (1 - epsilon) * base + epsilon * uniform
        # This same p(z | s) selects the initial AND subsequent skills.
        sources = np.split(probabilities, starts[1:])

        # Changing p(z | s) changes both the initial-skill weights and
        # M_beta^pi(g | s_0,z). Differentiate BOTH effects, so this is no longer BA.
        value, gradients = _start_information_gradient(libraries, sources, beta, start_state)
        if value > best_value:  # Save the probabilities with the highest MI at s_0.
            best_value, best_probabilities = value, probabilities.copy()
        if step == max_iterations:
            break

        # Differentiate this same objective with respect to the learnable logits.
        g = np.concatenate(gradients)
        mean_gradient = np.add.reduceat(base * g, starts)[state_ids]
        gradient = (1 - epsilon) * base * (g - mean_gradient)
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_mean = first_moment / (1 - 0.9**(step + 1))
        corrected_variance = second_moment / (1 - 0.999**(step + 1))
        progress = max(0.0, (step - anneal_iterations) / (max_iterations - anneal_iterations))
        rate = learning_rate * (1 - 0.99 * progress)

        # Update logits.
        logits += rate * corrected_mean / (np.sqrt(corrected_variance) + 1e-8)

    # Return MI at particular start state
    return best_value, np.split(best_probabilities, starts[1:])
