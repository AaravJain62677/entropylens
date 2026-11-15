from utils.answers import (
    extract_answer,
    normalize_answer,
    answers_match,
)


def test_extract_answer():
    text = "ANSWER: Canberra"

    assert extract_answer(text) == "Canberra"


def test_extract_answer_case_insensitive():
    text = "answer: Canberra"

    assert extract_answer(text) == "Canberra"


def test_extract_answer_with_whitespace():
    text = "ANSWER:    Canberra   "

    assert extract_answer(text) == "Canberra"


def test_extract_answer_with_colon_spacing():
    text = "ANSWER : Canberra"

    assert extract_answer(text) == "Canberra"


def test_extract_answer_multiline():
    text = (
        "Some reasoning.\n"
        "ANSWER: Canberra\n"
        "Additional text."
    )

    assert extract_answer(text) == "Canberra"


def test_extract_answer_returns_none_without_marker():
    text = (
        "The capital of Australia is Canberra."
    )

    assert extract_answer(text) is None


def test_extract_answer_returns_none_for_empty_answer():
    text = "ANSWER:"

    assert extract_answer(text) is None


def test_extract_answer_returns_none_for_empty_text():
    assert extract_answer("") is None


def test_extract_answer_returns_none_for_whitespace():
    assert extract_answer("   ") is None


def test_extract_answer_falls_back_to_plain_response():
    text = (
        "The Australian Capital Territory, "
        "which includes Canberra, is the federal district."
    )

    assert extract_answer(text) is None


def test_extract_answer_fallback_handles_multiline_response():
    text = (
        "Canberra is the capital of Australia.\n"
        "It is located in the Australian Capital Territory."
    )

    assert extract_answer(text) is None


def test_normalize_answer_lowercases():
    assert normalize_answer(
        "Canberra"
    ) == "canberra"


def test_normalize_answer_strips_whitespace():
    assert normalize_answer(
        "  Canberra  "
    ) == "canberra"


def test_normalize_answer_removes_surrounding_punctuation():
    assert normalize_answer(
        "Canberra."
    ) == "canberra"


def test_normalize_answer_collapses_whitespace():
    assert normalize_answer(
        "New   Delhi"
    ) == "new delhi"


def test_normalize_answer_handles_none():
    assert normalize_answer(None) is None


def test_answers_match_equivalent_answers():
    assert answers_match(
        "Canberra",
        "canberra.",
    )


def test_answers_match_different_answers():
    assert not answers_match(
        "Canberra",
        "Sydney",
    )


def test_answers_match_missing_answer():
    assert not answers_match(
        None,
        "Canberra",
    )


def test_answers_match_both_missing():
    assert not answers_match(
        None,
        None,
    )