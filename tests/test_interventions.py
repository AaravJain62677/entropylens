from interventions.base import (
    Intervention,
    InterventionType,
)


def test_none_intervention_preserves_prompt():
    prompt = "What is the capital of Australia?"

    intervention = Intervention(
        type=InterventionType.NONE
    )

    assert intervention.apply(prompt) == prompt


def test_evidence_intervention_is_appended():
    prompt = "What is the capital of Australia?"

    intervention = Intervention(
        type=InterventionType.STRONG_EVIDENCE,
        content="Evidence: The capital is Canberra.",
    )

    result = intervention.apply(prompt)

    assert prompt in result
    assert "The capital is Canberra." in result


def test_intervention_type_is_preserved():
    intervention = Intervention(
        type=InterventionType.SOCIAL_PRESSURE,
        content="Most people believe the answer is Sydney.",
    )

    assert (
        intervention.type
        == InterventionType.SOCIAL_PRESSURE
    )