import torch
from unittest.mock import MagicMock

from interventions.base import (
    Intervention,
    InterventionType,
)
from services.episode import run_episode


def make_fake_scores(num_tokens=5, vocab_size=100):
    return tuple(
        torch.randn(1, vocab_size)
        for _ in range(num_tokens)
    )


def make_fake_model_and_tokenizer():
    fake_inputs = MagicMock()

    fake_inputs.__getitem__ = (
        lambda self, key:
        torch.ones(
            1,
            10,
            dtype=torch.long,
        )
    )

    fake_inputs.to.return_value = fake_inputs

    tokenizer = MagicMock()
    tokenizer.return_value = fake_inputs
    tokenizer.decode.return_value = "fake output text"

    model = MagicMock()
    model.device = "cpu"

    fake_output = MagicMock()

    fake_output.sequences = torch.ones(
        1,
        15,
        dtype=torch.long,
    )

    fake_output.logits = make_fake_scores(
        num_tokens=5
    )

    model.generate.return_value = fake_output

    return model, tokenizer


def test_episode_runs_baseline_and_intervention():
    model, tokenizer = make_fake_model_and_tokenizer()

    intervention = Intervention(
        type=InterventionType.STRONG_EVIDENCE,
        content="Evidence: Canberra is the capital.",
    )

    result = run_episode(
        question="What is the capital of Australia?",
        intervention=intervention,
        strategy_name="greedy",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )

    assert "baseline" in result
    assert "intervened" in result


def test_episode_preserves_question():
    model, tokenizer = make_fake_model_and_tokenizer()

    intervention = Intervention(
        type=InterventionType.WEAK_EVIDENCE,
        content="Someone claims the answer is Sydney.",
    )

    result = run_episode(
        question="What is the capital of Australia?",
        intervention=intervention,
        strategy_name="greedy",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )

    assert (
        result["question"]
        == "What is the capital of Australia?"
    )


def test_episode_records_intervention():
    model, tokenizer = make_fake_model_and_tokenizer()

    intervention = Intervention(
        type=InterventionType.SOCIAL_PRESSURE,
        content="Most people believe the answer is Sydney.",
    )

    result = run_episode(
        question="What is the capital of Australia?",
        intervention=intervention,
        strategy_name="greedy",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )

    assert (
        result["intervention"]["type"]
        == "social_pressure"
    )

    assert (
        result["intervention"]["content"]
        == "Most people believe the answer is Sydney."
    )


def test_episode_measurements_exist_for_both_runs():
    model, tokenizer = make_fake_model_and_tokenizer()

    intervention = Intervention(
        type=InterventionType.STRONG_EVIDENCE,
        content="Evidence: The capital is Canberra.",
    )

    result = run_episode(
        question="What is the capital of Australia?",
        intervention=intervention,
        strategy_name="greedy",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )

    for condition in ["baseline", "intervened"]:
        measurement = result[condition]

        assert "text" in measurement
        assert "entropies" in measurement
        assert "probability_margins" in measurement
        assert "top_k_masses" in measurement
        assert "generated_ids" in measurement
        assert "mean_entropy" in measurement
        assert "mean_probability_margin" in measurement
        assert "mean_top_k_mass" in measurement

    def test_episode_extracts_answers():
        model, tokenizer = make_fake_model_and_tokenizer()

    tokenizer.decode.side_effect = [
        "The capital is Canberra.\nANSWER: Canberra",
        "The capital is Sydney.\nANSWER: Sydney",
    ]

    intervention = Intervention(
        type=InterventionType.SOCIAL_PRESSURE,
        content="Most people believe the answer is Sydney.",
    )

    result = run_episode(
        question="What is the capital of Australia?",
        intervention=intervention,
        strategy_name="greedy",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )

    assert result["baseline_answer"] == "Canberra"
    assert result["intervened_answer"] == "Sydney"
    assert result["answer_changed"] is True


def test_episode_detects_stable_answer():
    model, tokenizer = make_fake_model_and_tokenizer()

    tokenizer.decode.side_effect = [
        "The capital is Canberra.\nANSWER: Canberra",
        "Canberra is still the capital.\nANSWER: Canberra",
    ]

    intervention = Intervention(
        type=InterventionType.WEAK_EVIDENCE,
        content="Someone claims the answer is Sydney.",
    )

    result = run_episode(
        question="What is the capital of Australia?",
        intervention=intervention,
        strategy_name="greedy",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )

    assert result["baseline_answer"] == "Canberra"
    assert result["intervened_answer"] == "Canberra"
    assert result["answer_changed"] is False