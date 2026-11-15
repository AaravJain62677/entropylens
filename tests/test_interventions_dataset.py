import json
from pathlib import Path

from interventions.base import InterventionType


ROOT = Path(__file__).resolve().parent.parent

INTERVENTIONS_PATH = (
    ROOT
    / "experiments"
    / "interventions.json"
)


def load_interventions():
    with INTERVENTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def test_interventions_dataset_exists():
    assert INTERVENTIONS_PATH.exists()


def test_interventions_dataset_is_list():
    interventions = load_interventions()

    assert isinstance(
        interventions,
        list,
    )


def test_interventions_dataset_has_six_conditions():
    interventions = load_interventions()

    assert len(interventions) == 6


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


def test_all_intervention_types_are_valid():
    interventions = load_interventions()

    valid_types = {
        intervention_type.value
        for intervention_type in InterventionType
    }

    for intervention in interventions:
        assert (
            intervention["type"]
            in valid_types
        )


def test_required_fields_exist():
    interventions = load_interventions()

    required_fields = {
        "id",
        "type",
        "validity",
        "content",
    }

    for intervention in interventions:
        assert required_fields.issubset(
            intervention.keys()
        )


def test_control_condition_is_present():
    interventions = load_interventions()

    control_conditions = [
        intervention
        for intervention in interventions
        if intervention["type"] == "control"
    ]

    assert len(control_conditions) == 1


def test_control_condition_has_no_content():
    interventions = load_interventions()

    control = next(
        intervention
        for intervention in interventions
        if intervention["type"] == "control"
    )

    assert control["content"] == ""
    assert control["validity"] == "none"


def test_expected_intervention_types_are_present():
    interventions = load_interventions()

    types = {
        intervention["type"]
        for intervention in interventions
    }

    expected_types = {
        "control",
        "strong_evidence",
        "weak_evidence",
        "social_pressure",
        "misleading_evidence",
        "ambiguity",
    }

    assert types == expected_types


def test_non_control_interventions_have_content():
    interventions = load_interventions()

    for intervention in interventions:
        if intervention["type"] == "control":
            continue

        assert intervention["content"].strip()


def test_intervention_validity_values_are_present():
    interventions = load_interventions()

    expected_validity = {
        "control": "none",
        "strong_evidence": "truthful",
        "weak_evidence": "uncertain",
        "social_pressure": "social",
        "misleading_evidence": "false",
        "ambiguity": "ambiguous",
    }

    for intervention in interventions:
        intervention_type = intervention[
            "type"
        ]

        assert (
            intervention["validity"]
            == expected_validity[
                intervention_type
            ]
        )


def test_strong_evidence_contains_answer_placeholder():
    interventions = load_interventions()

    strong = next(
        intervention
        for intervention in interventions
        if intervention["type"]
        == "strong_evidence"
    )

    assert "{answer}" in strong["content"]


def test_misleading_evidence_contains_alternative_placeholder():
    interventions = load_interventions()

    misleading = next(
        intervention
        for intervention in interventions
        if intervention["type"]
        == "misleading_evidence"
    )

    assert (
        "{alternative_answer}"
        in misleading["content"]
    )