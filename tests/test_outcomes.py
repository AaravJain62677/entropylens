from services.outcomes import (
    EpisodeOutcome,
    classify_answer_change,
)


def test_true_is_changed():
    assert (
        classify_answer_change(True)
        == EpisodeOutcome.CHANGED
    )


def test_false_is_stable():
    assert (
        classify_answer_change(False)
        == EpisodeOutcome.STABLE
    )


def test_none_is_unknown():
    assert (
        classify_answer_change(None)
        == EpisodeOutcome.UNKNOWN
    )


def test_outcome_values():
    assert EpisodeOutcome.STABLE.value == "stable"
    assert EpisodeOutcome.CHANGED.value == "changed"
    assert EpisodeOutcome.UNKNOWN.value == "unknown"