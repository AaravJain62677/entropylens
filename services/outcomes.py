from enum import Enum


class EpisodeOutcome(str, Enum):
    STABLE_CORRECT = "stable_correct"
    STABLE_INCORRECT = "stable_incorrect"
    WARRANTED_UPDATE = "warranted_update"
    UNWARRANTED_UPDATE = "unwarranted_update"
    UNKNOWN = "unknown"


def classify_epistemic_outcome(
    baseline_correct: bool | None,
    intervened_correct: bool | None,
) -> EpisodeOutcome:
    """
    Classify an episode based on correctness before
    and after intervention.

    Transitions:

        correct -> correct
            stable_correct

        incorrect -> incorrect
            stable_incorrect

        incorrect -> correct
            warranted_update

        correct -> incorrect
            unwarranted_update

        unknown
            unknown
    """

    if (
        baseline_correct is None
        or intervened_correct is None
    ):
        return EpisodeOutcome.UNKNOWN

    if (
        baseline_correct
        and intervened_correct
    ):
        return EpisodeOutcome.STABLE_CORRECT

    if (
        not baseline_correct
        and not intervened_correct
    ):
        return EpisodeOutcome.STABLE_INCORRECT

    if (
        not baseline_correct
        and intervened_correct
    ):
        return EpisodeOutcome.WARRANTED_UPDATE

    return EpisodeOutcome.UNWARRANTED_UPDATE


def classify_answer_change(
    answer_changed: bool | None,
) -> EpisodeOutcome:
    """
    Backward-compatible behavioral classification.

    This does NOT determine whether a change was
    epistemically good or bad.
    """

    if answer_changed is None:
        return EpisodeOutcome.UNKNOWN

    if answer_changed:
        return EpisodeOutcome.WARRANTED_UPDATE

    return EpisodeOutcome.STABLE_CORRECT