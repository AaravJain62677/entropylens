import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    ROOT
    / "experiments"
    / "questions.json"
)


def load_questions():
    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def test_question_dataset_exists():
    assert QUESTIONS_PATH.exists()


def test_question_dataset_is_list():
    questions = load_questions()

    assert isinstance(
        questions,
        list,
    )


def test_question_dataset_has_40_questions():
    questions = load_questions()

    assert len(questions) == 40


def test_question_ids_are_unique():
    questions = load_questions()

    ids = [
        question["id"]
        for question in questions
    ]

    assert len(ids) == len(set(ids))


def test_question_categories_are_balanced():
    questions = load_questions()

    categories = [
        question["category"]
        for question in questions
    ]

    assert categories.count("factual") == 10
    assert categories.count("science") == 10
    assert categories.count("mathematics") == 10
    assert categories.count("logic") == 10


def test_question_ids_are_sequential():
    questions = load_questions()

    expected_ids = [
        f"q{i:03d}"
        for i in range(1, 41)
    ]

    actual_ids = [
        question["id"]
        for question in questions
    ]

    assert actual_ids == expected_ids


def test_required_fields_exist():
    questions = load_questions()

    required_fields = {
        "id",
        "category",
        "question",
        "answer",
        "acceptable_answers",
        "alternative_answer",
    }

    for question in questions:
        assert required_fields.issubset(
            question.keys()
        )


def test_categories_are_valid():
    questions = load_questions()

    valid_categories = {
        "factual",
        "science",
        "mathematics",
        "logic",
    }

    for question in questions:
        assert question["category"] in valid_categories


def test_acceptable_answers_are_lists():
    questions = load_questions()

    for question in questions:
        assert isinstance(
            question["acceptable_answers"],
            list,
        )


def test_acceptable_answers_are_non_empty():
    questions = load_questions()

    for question in questions:
        assert question["acceptable_answers"]


def test_answers_are_non_empty():
    questions = load_questions()

    for question in questions:
        assert question["answer"].strip()


def test_alternative_answers_are_non_empty():
    questions = load_questions()

    for question in questions:
        assert question["alternative_answer"].strip()