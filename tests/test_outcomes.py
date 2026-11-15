from services.outcomes import (
    EpisodeOutcome,
    classify_epistemic_outcome,
)


def test_stable_correct():
    assert (
        classify_epistemic_outcome(
            True,
            True,
        )
        == EpisodeOutcome.STABLE_CORRECT
    )


def test_stable_incorrect():
    assert (
        classify_epistemic_outcome(
            False,
            False,
        )
        == EpisodeOutcome.STABLE_INCORRECT
    )


def test_warranted_update():
    assert (
        classify_epistemic_outcome(
            False,
            True,
        )
        == EpisodeOutcome.WARRANTED_UPDATE
    )


def test_unwarranted_update():
    assert (
        classify_epistemic_outcome(
            True,
            False,
        )
        == EpisodeOutcome.UNWARRANTED_UPDATE
    )


def test_unknown_baseline():
    assert (
        classify_epistemic_outcome(
            None,
            True,
        )
        == EpisodeOutcome.UNKNOWN
    )


def test_unknown_intervened():
    assert (
        classify_epistemic_outcome(
            True,
            None,
        )
        == EpisodeOutcome.UNKNOWN
    )


def test_both_unknown():
    assert (
        classify_epistemic_outcome(
            None,
            None,
        )
        == EpisodeOutcome.UNKNOWN
    )