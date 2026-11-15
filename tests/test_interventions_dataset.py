import json
from pathlib import Path


INTERVENTIONS_PATH = (
    Path(__file__).parent.parent
    / "experiments"
    / "interventions.json"
)


def load_interventions():
    with INTERVENTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def test_intervention_dataset_has_five_interventions():
    interventions = load_interventions()

    assert len(interventions) == 5


def test_intervention_ids_are_unique():
    interventions = load_interventions()

    ids = [
        intervention["id"]
        for intervention in interventions
    ]

    assert len(ids) == len(set(ids))


def test_intervention_types_are_unique():
    interventions = load_interventions()

    types = [
        intervention["type"]
        for intervention in interventions
    ]

    assert len(types) == len(set(types))


def test_interventions_have_required_fields():
    interventions = load_interventions()

    for intervention in interventions:
        assert "id" in intervention
        assert "type" in intervention
        assert "content" in intervention

        assert intervention["id"]
        assert intervention["type"]
        assert intervention["content"]


def test_expected_intervention_types_exist():
    interventions = load_interventions()

    types = {
        intervention["type"]
        for intervention in interventions
    }

    expected = {
        "strong_evidence",
        "weak_evidence",
        "social_pressure",
        "misleading_evidence",
        "ambiguity",
    }

    assert types == expected