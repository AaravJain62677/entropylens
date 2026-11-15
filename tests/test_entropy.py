import torch
import pytest

from metrics.entropy import (
    token_entropies,
    mean_entropy,
    token_probability_margins,
    mean_probability_margin,
    token_top_k_mass,
    mean_top_k_mass,
)


def test_uniform_distribution_has_max_entropy():
    vocab_size = 100

    uniform_logits = torch.zeros(
        1,
        vocab_size,
    )

    scores = (uniform_logits,) * 5

    entropies = token_entropies(scores)

    assert len(entropies) == 5

    # Entropy of a uniform distribution over 100 values:
    #
    # H = log(100) ≈ 4.605
    #
    for entropy in entropies:
        assert abs(entropy - 4.605) < 0.01


def test_deterministic_distribution_has_zero_entropy():
    vocab_size = 100

    peaked_logits = torch.full(
        (1, vocab_size),
        -1e9,
    )

    peaked_logits[0][42] = 0.0

    scores = (peaked_logits,)

    entropies = token_entropies(scores)

    assert entropies[0] < 0.01


def test_mean_entropy():
    assert mean_entropy(
        [2.0, 4.0, 6.0]
    ) == 4.0

    assert mean_entropy([]) == 0.0


def test_probability_margin_is_non_negative():
    logits = torch.tensor(
        [[5.0, 1.0, 0.0]]
    )

    scores = (logits,)

    margins = token_probability_margins(
        scores
    )

    assert len(margins) == 1
    assert margins[0] >= 0.0


def test_probability_margin_for_uniform_distribution():
    logits = torch.zeros(
        1,
        100,
    )

    scores = (logits,)

    margins = token_probability_margins(
        scores
    )

    # All probabilities are equal.
    # Therefore top-1 probability == top-2 probability.
    assert abs(margins[0]) < 1e-6


def test_mean_probability_margin():
    margins = [
        0.2,
        0.4,
        0.6,
    ]

    assert mean_probability_margin(
        margins
    ) == pytest.approx(0.4)


def test_top_k_mass_is_between_zero_and_one():
    logits = torch.randn(
        1,
        100,
    )

    scores = (logits,)

    masses = token_top_k_mass(
        scores,
        k=5,
    )

    assert len(masses) == 1
    assert 0.0 <= masses[0] <= 1.0


def test_top_k_mass_for_uniform_distribution():
    vocab_size = 100

    logits = torch.zeros(
        1,
        vocab_size,
    )

    scores = (logits,)

    masses = token_top_k_mass(
        scores,
        k=5,
    )

    # Uniform distribution:
    #
    # 5 * (1 / 100) = 0.05
    #
    assert abs(masses[0] - 0.05) < 1e-6


def test_mean_top_k_mass():
    masses = [
        0.5,
        0.7,
        0.9,
    ]

    assert mean_top_k_mass(
        masses
    ) == pytest.approx(0.7)


def test_empty_signal_lists():
    assert mean_probability_margin([]) == 0.0
    assert mean_top_k_mass([]) == 0.0