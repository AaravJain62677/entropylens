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


def normalize_answer(answer: str | None) -> str | None:
    """
    Normalize an extracted answer for comparison.

    Normalization currently:
    - handles missing answers
    - strips leading/trailing whitespace
    - converts to lowercase
    - removes surrounding punctuation
    - collapses repeated whitespace
    """

    if answer is None:
        return None

    normalized = answer.strip().lower()

    normalized = re.sub(
        r"^[\s\.,;:!?()\[\]{}]+|[\s\.,;:!?()\[\]{}]+$",
        "",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized if normalized else None


def answers_match(
    first: str | None,
    second: str | None,
) -> bool:
    """
    Determine whether two answers are equivalent
    under the current normalization rules.

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