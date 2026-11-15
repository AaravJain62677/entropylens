from strategies.registry import get_strategy
from inference.runner import run_inference

from metrics.entropy import (
    token_entropies,
    mean_entropy,
    token_probability_margins,
    mean_probability_margin,
    token_top_k_mass,
    mean_top_k_mass,
)

from metrics.latency import summarize_latency
from metrics.quality import repetition_rate


def run_comparison(
    prompt: str,
    strategies_to_run: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 200,
) -> dict:
    """
    Run the same prompt using multiple generation strategies
    and calculate EntropyLens internal uncertainty signals.

    Existing metrics are preserved.
    """

    results = {}

    for strategy_name in strategies_to_run:
        strategy_kwargs = get_strategy(strategy_name)

        output = run_inference(
            model,
            tokenizer,
            prompt,
            strategy_kwargs,
            max_new_tokens,
        )


        entropies = token_entropies(
            output["scores"]
        )

        probability_margins = token_probability_margins(
            output["scores"]
        )

        top_k_masses = token_top_k_mass(
            output["scores"],
            k=5,
        )

        # ---------------------------------------------------------
        # Preserve existing result structure while extending it
        # ---------------------------------------------------------

        results[strategy_name] = {
            "text": output["text"],

            # Existing entropy measurements
            "entropies": entropies,
            "mean_entropy": mean_entropy(entropies),

            # New Phase 1 internal signals
            "probability_margins": probability_margins,
            "mean_probability_margin": mean_probability_margin(
                probability_margins
            ),

            "top_k_masses": top_k_masses,
            "mean_top_k_mass": mean_top_k_mass(
                top_k_masses
            ),

            # Preserve token information
            "generated_ids": output["generated_ids"],

            # Existing performance/quality metrics
            "latency": summarize_latency(
                output["latency_ms"],
                output["num_output_tokens"],
            ),

            "repetition_rate": repetition_rate(
                output["text"]
            ),

            "num_input_tokens": output[
                "num_input_tokens"
            ],

            "num_output_tokens": output[
                "num_output_tokens"
            ],
        }

    return results