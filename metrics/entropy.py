import torch
import torch.nn.functional as F


def token_entropies(scores: tuple) -> list[float]:
    """
    Calculate Shannon entropy for each generated token.

    Each element of `scores` is expected to contain logits with shape:

        [batch_size, vocabulary_size]

    Returns:
        One entropy value per generated token.
    """

    entropies = []

    for logits in scores:
        # Remove batch dimension.
        logits = logits[0]

        probs = F.softmax(logits, dim=-1)

        # Avoid log(0).
        probs = torch.clamp(probs, min=1e-10)

        entropy = -torch.sum(
            probs * torch.log(probs)
        ).item()

        entropies.append(entropy)

    return entropies


def mean_entropy(entropies: list[float]) -> float:
    """
    Calculate mean token entropy.
    """

    return (
        sum(entropies) / len(entropies)
        if entropies
        else 0.0
    )


def token_probability_margins(
    scores: tuple,
) -> list[float]:
    """
    Calculate the probability margin between the
    most likely and second-most-likely token.

    margin = P(top1) - P(top2)

    Larger values indicate a more concentrated
    next-token distribution.
    """

    margins = []

    for logits in scores:
        logits = logits[0]

        probs = F.softmax(logits, dim=-1)

        top_probs = torch.topk(
            probs,
            k=2,
            dim=-1,
        ).values

        margin = (
            top_probs[0] - top_probs[1]
        ).item()

        margins.append(margin)

    return margins


def mean_probability_margin(
    margins: list[float],
) -> float:
    """
    Calculate the mean probability margin.
    """

    return (
        sum(margins) / len(margins)
        if margins
        else 0.0
    )


def token_top_k_mass(
    scores: tuple,
    k: int = 5,
) -> list[float]:
    """
    Calculate the probability mass contained
    in the top-k next-token candidates.

    For example:

        top_5_mass = P(token_1) + ... + P(token_5)

    Values closer to 1 indicate that probability
    is concentrated among a small number of candidates.
    """

    masses = []

    for logits in scores:
        logits = logits[0]

        probs = F.softmax(logits, dim=-1)

        effective_k = min(
            k,
            probs.shape[-1],
        )

        mass = torch.topk(
            probs,
            k=effective_k,
            dim=-1,
        ).values.sum().item()

        masses.append(mass)

    return masses


def mean_top_k_mass(
    masses: list[float],
) -> float:
    """
    Calculate the mean top-k probability mass.
    """

    return (
        sum(masses) / len(masses)
        if masses
        else 0.0
    )