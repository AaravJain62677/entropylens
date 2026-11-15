import re


def extract_answer(text: str) -> str | None:
    """
    Extract an answer from text using the expected format:

        ANSWER: <answer>

    Returns None if no answer marker is found.
    """

    match = re.search(
        r"ANSWER\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    answer = match.group(1).strip()

    return answer if answer else None


def normalize_answer(
    answer: str | None,
) -> str | None:
    """
    Normalize an answer for deterministic comparison.

    Normalization:
    - strips whitespace
    - lowercases
    - removes common surrounding punctuation
    - normalizes degree symbols
    - normalizes common textual variations
    - collapses whitespace
    """

    if answer is None:
        return None

    normalized = answer.strip().lower()

    normalized = normalized.replace(
        "°c",
        " degrees celsius",
    )

    normalized = normalized.replace(
        "ºc",
        " degrees celsius",
    )

    normalized = normalized.replace(
        "° c",
        " degrees celsius",
    )

    normalized = normalized.replace(
        "co₂",
        "co2",
    )

    normalized = re.sub(
        r"[^\w\s\.\+\-]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    normalized = re.sub(
        r"^[\s\.,;:!?()\[\]{}]+|[\s\.,;:!?()\[\]{}]+$",
        "",
        normalized,
    )

    return normalized if normalized else None


def answers_match(
    first: str | None,
    second: str | None,
) -> bool:
    """
    Determine whether two answers are equivalent
    under normalization.

    Missing answers never count as matching.
    """

    normalized_first = normalize_answer(first)
    normalized_second = normalize_answer(second)

    if (
        normalized_first is None
        or normalized_second is None
    ):
        return False

    return normalized_first == normalized_second


def answer_is_correct(
    answer: str | None,
    ground_truth: str,
    acceptable_answers: list[str] | None = None,
) -> bool | None:
    """
    Determine whether a model answer is correct.

    The evaluation is deterministic and benchmark-specific.

    If acceptable_answers are provided, the model answer
    is considered correct if it matches any acceptable answer
    after normalization.

    Returns None when the model answer cannot be evaluated.
    """

    if answer is None:
        return None

    candidates = (
        acceptable_answers
        if acceptable_answers
        else [ground_truth]
    )

    normalized_answer = normalize_answer(
        answer
    )

    if normalized_answer is None:
        return None

    normalized_candidates = {
        normalize_answer(candidate)
        for candidate in candidates
    }

    normalized_candidates.discard(None)

    return (
        normalized_answer
        in normalized_candidates
    )