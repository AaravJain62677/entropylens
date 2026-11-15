from interventions.base import Intervention
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
from strategies.registry import get_strategy
from utils.answers import (
    extract_answer,
    answers_match,
)
from services.outcomes import classify_answer_change


def format_prompt(
    prompt: str,
    tokenizer,
) -> str:
    """
    Format a raw user prompt for an instruction/chat model.

    If the tokenizer provides a chat template, use it.
    Otherwise, return the prompt unchanged.
    """

    chat_template = getattr(
        tokenizer,
        "chat_template",
        None,
    )

    if chat_template:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def measure_inference(
    prompt: str,
    strategy_name: str,
    model,
    tokenizer,
    max_new_tokens: int = 200,
) -> dict:
    """
    Run one inference and calculate all EntropyLens
    measurement signals.

    The prompt is formatted for the target model before
    generation.
    """

    formatted_prompt = format_prompt(
        prompt,
        tokenizer,
    )

    strategy_kwargs = get_strategy(
        strategy_name
    )

    output = run_inference(
        model,
        tokenizer,
        formatted_prompt,
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

    return {
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "text": output["text"],
        "generated_ids": output["generated_ids"],
        "entropies": entropies,
        "mean_entropy": mean_entropy(
            entropies
        ),
        "probability_margins": (
            probability_margins
        ),
        "mean_probability_margin": (
            mean_probability_margin(
                probability_margins
            )
        ),
        "top_k_masses": top_k_masses,
        "mean_top_k_mass": (
            mean_top_k_mass(
                top_k_masses
            )
        ),
        "latency": summarize_latency(
            output["latency_ms"],
            output["num_output_tokens"],
        ),
        "repetition_rate": repetition_rate(
            output["text"]
        ),
        "num_input_tokens": (
            output["num_input_tokens"]
        ),
        "num_output_tokens": (
            output["num_output_tokens"]
        ),
    }


def run_episode(
    question: str,
    intervention: Intervention,
    strategy_name: str,
    model,
    tokenizer,
    max_new_tokens: int = 200,
) -> dict:
    """
    Run a baseline inference followed by an
    intervention inference.

    The intervention is applied to the raw question
    BEFORE model-specific chat formatting.

    Both runs use the same model, tokenizer,
    generation strategy, and measurement pipeline.
    """

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline = measure_inference(
        prompt=question,
        strategy_name=strategy_name,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    # ---------------------------------------------------------
    # Intervention
    #
    # Apply the intervention to the raw prompt.
    # measure_inference() then handles chat formatting.
    # ---------------------------------------------------------

    intervention_prompt = intervention.apply(
        question
    )

    intervened = measure_inference(
        prompt=intervention_prompt,
        strategy_name=strategy_name,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    # ---------------------------------------------------------
    # Extract answers
    # ---------------------------------------------------------

    baseline_answer = extract_answer(
        baseline["text"]
    )

    intervened_answer = extract_answer(
        intervened["text"]
    )

    # ---------------------------------------------------------
    # Compare answers
    # ---------------------------------------------------------

    answer_changed = (
        None
        if (
            baseline_answer is None
            or intervened_answer is None
        )
        else not answers_match(
            baseline_answer,
            intervened_answer,
        )
    )

    # ---------------------------------------------------------
    # Classify behavioral outcome
    # ---------------------------------------------------------

    outcome = classify_answer_change(
        answer_changed
    )

    return {
        "question": question,
        "intervention": {
            "type": intervention.type.value,
            "content": intervention.content,
        },
        "baseline": baseline,
        "intervened": intervened,
        "baseline_answer": baseline_answer,
        "intervened_answer": intervened_answer,
        "answer_changed": answer_changed,
        "outcome": outcome.value,
    }