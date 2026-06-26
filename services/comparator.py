from strategies.registry import get_strategy
from inference.runner import run_inference
from metrics.entropy import token_entropies, mean_entropy
from metrics.latency import summarize_latency
from metrics.quality import repetition_rate

def run_comparison(
    prompt: str,
    strategies_to_run: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 200,
) -> dict:
    results = {}

    for strategy_name in strategies_to_run:
        strategy_kwargs = get_strategy(strategy_name)
        output = run_inference(model, tokenizer, prompt, strategy_kwargs, max_new_tokens)

        entropies = token_entropies(output["scores"])

        results[strategy_name] = {
            "text": output["text"],
            "entropies": entropies,
            "mean_entropy": mean_entropy(entropies),
            "latency": summarize_latency(output["latency_ms"], output["num_output_tokens"]),
            "repetition_rate": repetition_rate(output["text"]),
            "num_input_tokens": output["num_input_tokens"],
            "num_output_tokens": output["num_output_tokens"],
        }

    return results