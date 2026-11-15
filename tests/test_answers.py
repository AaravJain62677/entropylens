from utils.answers import (
    extract_answer,
    normalize_answer,
    answers_match,
    answer_is_correct,
)


def test_extract_answer():
    text = "ANSWER: Canberra"

    assert extract_answer(text) == "Canberra"


def test_extract_answer_case_insensitive():
    text = "answer: Jupiter"

    assert extract_answer(text) == "Jupiter"


def test_extract_answer_missing():
    text = "The answer is Canberra."

    assert extract_answer(text) is None


def test_extract_answer_empty():
    text = "ANSWER:"

    assert extract_answer(text) is None


def test_normalize_answer():
    assert (
        normalize_answer("  Canberra!  ")
        == "canberra"
    )


def test_normalize_degree_answer():
    assert (
        normalize_answer("100°C")
        == "100 degrees celsius"
    )


def test_normalize_co2():
    assert (
        normalize_answer("CO₂")
        == "co2"
    )


def test_answers_match():
    assert answers_match(
        "Canberra",
        "canberra",
    )


def test_answers_do_not_match():
    assert not answers_match(
        "Canberra",
        "Sydney",
    )


def test_missing_answers_do_not_match():
    assert not answers_match(
        None,
        "Canberra",
    )


def test_semantic_acceptable_answer():
    assert answer_is_correct(
        "There are seven continents on Earth.",
        "7",
        [
            "7",
            "seven",
            "there are seven continents on earth",
            "there are seven continents on Earth",
        ],
    )


def test_semantic_equivalent_answer():
    assert answer_is_correct(
        "Carbon dioxide (CO2)",
        "Carbon dioxide",
        [
            "Carbon dioxide",
            "carbon dioxide (CO2)",
            "CO2",
        ],
    )


def test_numeric_answer_with_units():
    assert answer_is_correct(
        "100°C",
        "100",
        [
            "100",
            "100°C",
            "100 degrees Celsius",
        ],
    )


def test_sentence_answer():
    assert answer_is_correct(
        "The basic unit of life is the cell.",
        "Cell",
        [
            "Cell",
            "the cell",
            "cells",
            "the basic unit of life is the cell",
            "the basic unit of life is typically considered to be the cell",
        ],
    )


def test_sentence_paraphrase_is_correct():
    assert answer_is_correct(
        "The basic unit of life is typically considered to be the cell.",
        "Cell",
        [
            "Cell",
            "the cell",
            "cells",
            "the basic unit of life is the cell",
            "the basic unit of life is typically considered to be the cell",
        ],
    )


def test_extended_fact_answer_is_correct():
    assert answer_is_correct(
        "George Orwell wrote the novel 1984.",
        "George Orwell",
        [
            "George Orwell",
            "George Orwell wrote the novel 1984",
            "the novel 1984 was written by George Orwell",
        ],
    )


def test_numeric_unit_variant_is_correct():
    assert answer_is_correct(
        "100 degrees Celsius",
        "100",
        [
            "100",
            "100°C",
            "100 degrees Celsius",
        ],
    )


def test_gravity_sentence_answer_is_correct():
    assert answer_is_correct(
        "Gravity pulls objects toward the center of the Earth.",
        "Gravity",
        [
            "Gravity",
            "gravity pulls objects toward the center of the Earth",
            "the force is gravity",
        ],
    )


def test_heart_sentence_answer_is_correct():
    assert answer_is_correct(
        "The heart pumps blood throughout the human body.",
        "Heart",
        [
            "Heart",
            "the heart",
            "the heart pumps blood throughout the human body",
        ],
    )


def test_train_duration_variant_is_correct():
    assert answer_is_correct(
        "2 hours and 30 minutes",
        "2 hours 30 minutes",
        [
            "2 hours 30 minutes",
            "2 hours and 30 minutes",
            "2.5 hours",
            "two hours and thirty minutes",
        ],
    )


def test_incorrect_answer():
    assert not answer_is_correct(
        "Sydney",
        "Canberra",
        [
            "Canberra",
        ],
    )


def test_missing_answer():
    assert (
        answer_is_correct(
            None,
            "Canberra",
            ["Canberra"],
        )
        is None
    )


def test_wrong_alternative_is_not_accepted():
    assert not answer_is_correct(
        "Sydney",
        "Canberra",
        [
            "Canberra",
            "the capital of Australia is Canberra",
        ],
    )


def test_oxygen_is_not_carbon_dioxide():
    assert not answer_is_correct(
        "Oxygen",
        "Carbon dioxide",
        [
            "Carbon dioxide",
            "carbon dioxide (CO2)",
            "CO2",
        ],
    )


def test_atom_is_not_cell():
    assert not answer_is_correct(
        "Atom",
        "Cell",
        [
            "Cell",
            "the cell",
            "cells",
            "the basic unit of life is the cell",
        ],
    )