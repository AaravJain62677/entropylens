from enum import Enum


class EpisodeOutcome(str, Enum):
    STABLE = "stable"
    CHANGED = "changed"
    UNKNOWN = "unknown"


def classify_answer_change(
    answer_changed: bool | None,
) -> EpisodeOutcome:
    """
    Classify the behavioral outcome of an episode
    based only on whether the extracted answer changed.

    This classification does not determine whether
    the change was epistemically good or bad.
    """

    if answer_changed is None:
        return EpisodeOutcome.UNKNOWN

    if answer_changed:
        return EpisodeOutcome.CHANGED

    return EpisodeOutcome.STABLE