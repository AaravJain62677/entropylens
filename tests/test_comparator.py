import torch
import pytest
from unittest.mock import MagicMock
from services.comparator import run_comparison


def make_fake_scores(num_tokens=5, vocab_size=100):
    return tuple(torch.randn(1, vocab_size) for _ in range(num_tokens))


def make_fake_model_and_tokenizer():
    fake_inputs = MagicMock()
    fake_inputs.__getitem__ = lambda self, key: torch.ones(1, 10, dtype=torch.long)
    fake_inputs.to.return_value = fake_inputs  # .to() returns itself

    tokenizer = MagicMock()
    tokenizer.return_value = fake_inputs
    tokenizer.decode.return_value = "fake output text"

    model = MagicMock()
    model.device = "cpu"

    fake_output = MagicMock()
    # sequences shape: (1, input_len + output_len) = (1, 15)
    fake_output.sequences = torch.ones(1, 15, dtype=torch.long)
    fake_output.logits = make_fake_scores(num_tokens=5)
    model.generate.return_value = fake_output

    return model, tokenizer


def test_comparator_returns_all_strategies():
    model, tokenizer = make_fake_model_and_tokenizer()
    results = run_comparison(
        prompt="Test prompt",
        strategies_to_run=["greedy", "top_k"],
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )
    assert "greedy" in results
    assert "top_k" in results


def test_comparator_result_has_required_keys():
    model, tokenizer = make_fake_model_and_tokenizer()
    results = run_comparison(
        prompt="Test prompt",
        strategies_to_run=["greedy"],
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )
    result = results["greedy"]
    for key in ["text", "entropies", "mean_entropy", "latency", "repetition_rate",
                "num_input_tokens", "num_output_tokens"]:
        assert key in result, f"Missing key: {key}"


def test_comparator_mean_entropy_is_float():
    model, tokenizer = make_fake_model_and_tokenizer()
    results = run_comparison(
        prompt="Test prompt",
        strategies_to_run=["greedy"],
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )
    assert isinstance(results["greedy"]["mean_entropy"], float)


def test_comparator_entropies_length_matches_output_tokens():
    model, tokenizer = make_fake_model_and_tokenizer()
    results = run_comparison(
        prompt="Test prompt",
        strategies_to_run=["greedy"],
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,
    )
    result = results["greedy"]
    assert len(result["entropies"]) == result["num_output_tokens"]