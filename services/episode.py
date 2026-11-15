from interventions.base import (
    Intervention,
    InterventionType,
)

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
    answer_is_correct,
)

from services.outcomes import (
    classify_epistemic_outcome,
)


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

    probability_margins = (
        token_probability_margins(
            output["scores"]
        )
    )

    top_k_masses = token_top_k_mass(
        output["scores"],
        k=5,
    )

    return {
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "text": output["text"],
        "generated_ids": output[
            "generated_ids"
        ],

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
    ground_truth: str | None = None,
    acceptable_answers: list[str] | None = None,
) -> dict:
    """
    Run a baseline inference followed by either:

    - a repeat-control inference, or
    - an intervention inference.

    Both baseline and comparison prompts use the same
    strict ANSWER format so that answer extraction
    remains deterministic.

    Control:
        question + answer-format instruction

    Intervention:
        question + intervention + answer-format instruction
    """

    answer_format_instruction = (
        "\n\n"
        "Answer in exactly this format and nothing else:\n"
        "ANSWER: <your answer>"
    )

    baseline_prompt = (
        question
        + answer_format_instruction
    )

    baseline = measure_inference(
        prompt=baseline_prompt,
        strategy_name=strategy_name,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    if (
        intervention.type
        == InterventionType.CONTROL
    ):
        comparison_prompt = baseline_prompt
        comparison_label = "control"
    else:
        comparison_prompt = (
            intervention.apply(question)
            + answer_format_instruction
        )
        comparison_label = "intervened"

    comparison = measure_inference(
        prompt=comparison_prompt,
        strategy_name=strategy_name,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    baseline_answer = extract_answer(
        baseline["text"]
    )

    comparison_answer = extract_answer(
        comparison["text"]
    )

    if ground_truth is not None:
        baseline_correct = answer_is_correct(
            baseline_answer,
            ground_truth,
            acceptable_answers,
        )

        comparison_correct = answer_is_correct(
            comparison_answer,
            ground_truth,
            acceptable_answers,
        )
    else:
        baseline_correct = None
        comparison_correct = None

    answer_changed = (
        None
        if (
            baseline_answer is None
            or comparison_answer is None
        )
        else not answers_match(
            baseline_answer,
            comparison_answer,
        )
    )

    outcome = classify_epistemic_outcome(
        baseline_correct,
        comparison_correct,
    )

    result = {
        "question": question,
        "intervention": {
            "type": intervention.type.value,
            "content": intervention.content,
        },
        "baseline": baseline,
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_correct,
        "answer_changed": answer_changed,
        "ground_truth": ground_truth,
        "acceptable_answers": acceptable_answers,
        "outcome": outcome.value,
    }

    if comparison_label == "control":
        result["control"] = comparison
        result["control_answer"] = comparison_answer
        result["control_correct"] = comparison_correct

        result["control_entropy_delta"] = (
            comparison["mean_entropy"]
            - baseline["mean_entropy"]
        )

        result[
            "control_probability_margin_delta"
        ] = (
            comparison[
                "mean_probability_margin"
            ]
            - baseline[
                "mean_probability_margin"
            ]
        )

        result["control_top_k_mass_delta"] = (
            comparison["mean_top_k_mass"]
            - baseline["mean_top_k_mass"]
        )

    else:
        result["intervened"] = comparison
        result["intervened_answer"] = comparison_answer
        result["intervened_correct"] = comparison_correct

        result["intervened_entropy_delta"] = (
            comparison["mean_entropy"]
            - baseline["mean_entropy"]
        )

        result[
            "intervened_probability_margin_delta"
        ] = (
            comparison[
                "mean_probability_margin"
            ]
            - baseline[
                "mean_probability_margin"
            ]
        )

        result[
            "intervened_top_k_mass_delta"
        ] = (
            comparison["mean_top_k_mass"]
            - baseline["mean_top_k_mass"]
        )

    return result