import json
from pathlib import Path


QUESTIONS_PATH = (
    Path(__file__).parent.parent
    / "experiments"
    / "questions.json"
)


def load_questions():
    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def test_question_dataset_has_20_questions():
    questions = load_questions()

    assert len(questions) == 20


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

    assert categories.count("factual") == 5
    assert categories.count("science") == 5
    assert categories.count("mathematics") == 5
    assert categories.count("logic") == 5


def test_questions_have_required_fields():
    questions = load_questions()

    for question in questions:
        assert "id" in question
        assert "category" in question
        assert "question" in question
        assert "answer" in question

        assert question["id"]
        assert question["question"]
        assert question["answer"]