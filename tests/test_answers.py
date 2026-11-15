from utils.answers import (
    extract_answer,
    normalize_answer,
    answers_match,
)


def test_extract_answer():
    text = "The capital is Canberra.\nANSWER: Canberra"

    assert extract_answer(text) == "Canberra"


def test_extract_answer_is_case_insensitive():
    text = "answer: Canberra"

    assert extract_answer(text) == "Canberra"


def test_extract_answer_handles_extra_spaces():
    text = "ANSWER:    Canberra   "

    assert extract_answer(text) == "Canberra"


def test_extract_answer_returns_none_when_missing():
    text = "The capital is Canberra."

    assert extract_answer(text) is None


def test_extract_answer_returns_none_for_empty_answer():
    text = "ANSWER:"

    assert extract_answer(text) is None


def test_normalize_answer_lowercases():
    assert normalize_answer("Canberra") == "canberra"


def test_normalize_answer_strips_whitespace():
    assert normalize_answer(
        "   Canberra   "
    ) == "canberra"


def test_normalize_answer_removes_punctuation():
    assert normalize_answer(
        "Canberra."
    ) == "canberra"


def test_normalize_answer_collapses_whitespace():
    assert normalize_answer(
        "New    Delhi"
    ) == "new delhi"


def test_normalize_answer_handles_none():
    assert normalize_answer(None) is None


def test_answers_match_equivalent_answers():
    assert answers_match(
        "Canberra",
        "canberra.",
    ) is True


def test_answers_match_different_answers():
    assert answers_match(
        "Canberra",
        "Sydney",
    ) is False


def test_answers_match_missing_answers():
    assert answers_match(
        None,
        "Canberra",
    ) is False


def test_answers_match_both_missing():
    assert answers_match(
        None,
        None,
    ) is False