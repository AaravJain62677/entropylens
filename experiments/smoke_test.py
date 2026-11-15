import json
from pathlib import Path

from interventions.base import (
    Intervention,
    InterventionType,
)
from inference.loader import (
    load_model_and_tokenizer,
)
from services.episode import run_episode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "questions.json"
)


def load_question(question_id: str) -> dict:
    """Load a question from the experiment dataset."""

    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        questions = json.load(f)

    for question in questions:
        if question["id"] == question_id:
            return question

    raise ValueError(
        f"Question not found: {question_id}"
    )


def build_prompt(question: str) -> str:
    """Build the standardized experiment prompt."""

    return (
        f"Question: {question}\n\n"
        "Answer the question briefly.\n"
        "Do not provide an explanation.\n"
        "Your response MUST contain exactly one line "
        "in this format:\n"
        "ANSWER: <your answer>"
    )


def main():
    # ---------------------------------------------------------
    # Load question
    # ---------------------------------------------------------

    question = load_question("q001")

    print("\nLoading model...")
    print(
        "Model: Qwen/Qwen2.5-1.5B-Instruct"
    )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model, tokenizer = (
        load_model_and_tokenizer(
            "Qwen/Qwen2.5-1.5B-Instruct",
            device="auto",
        )
    )

    # ---------------------------------------------------------
    # Define intervention
    # ---------------------------------------------------------

    intervention = Intervention(
        type=(
            InterventionType.STRONG_EVIDENCE
        ),
        content=(
            "A reliable reference confirms that "
            "the correct answer is Canberra. "
            "Reconsider your answer using "
            "this evidence."
        ),
    )

    # ---------------------------------------------------------
    # Build raw experiment prompt
    # ---------------------------------------------------------

    prompt = build_prompt(
        question["question"]
    )

    # ---------------------------------------------------------
    # Run complete episode
    # ---------------------------------------------------------

    result = run_episode(
        question=prompt,
        intervention=intervention,
        strategy_name="greedy",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
    )

    # ---------------------------------------------------------
    # Extract signal values
    # ---------------------------------------------------------

    baseline = result["baseline"]
    intervened = result["intervened"]

    baseline_entropy = (
        baseline["mean_entropy"]
    )

    intervened_entropy = (
        intervened["mean_entropy"]
    )

    baseline_margin = (
        baseline["mean_probability_margin"]
    )

    intervened_margin = (
        intervened[
            "mean_probability_margin"
        ]
    )

    baseline_top_k = (
        baseline["mean_top_k_mass"]
    )

    intervened_top_k = (
        intervened["mean_top_k_mass"]
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("ENTROPY LENS SMOKE TEST")
    print("=" * 60)

    print("\nQuestion:")
    print(question["question"])

    print("\nGround truth:")
    print(question["answer"])

    print("\n--- BASELINE ---")

    print("\nResponse:")
    print(baseline["text"])

    print("\nExtracted answer:")
    print(result["baseline_answer"])

    print("\n--- INTERVENTION ---")

    print("\nType:")
    print(intervention.type.value)

    print("\nResponse:")
    print(intervened["text"])

    print("\nExtracted answer:")
    print(result["intervened_answer"])

    print("\n--- BEHAVIOR ---")

    print(
        f"Answer changed: "
        f"{result['answer_changed']}"
    )

    print(
        f"Outcome: "
        f"{result['outcome']}"
    )

    print("\n--- INTERNAL SIGNALS ---")

    print(
        f"Baseline entropy:       "
        f"{baseline_entropy:.4f}"
    )

    print(
        f"Intervened entropy:      "
        f"{intervened_entropy:.4f}"
    )

    print(
        f"Entropy delta:           "
        f"{intervened_entropy - baseline_entropy:+.4f}"
    )

    print()

    print(
        "Baseline probability margin:  "
        f"{baseline_margin:.4f}"
    )

    print(
        "Intervened probability margin: "
        f"{intervened_margin:.4f}"
    )

    print(
        "Probability margin delta:      "
        f"{intervened_margin - baseline_margin:+.4f}"
    )

    print()

    print(
        f"Baseline top-5 mass:     "
        f"{baseline_top_k:.4f}"
    )

    print(
        f"Intervened top-5 mass:   "
        f"{intervened_top_k:.4f}"
    )

    print(
        f"Top-5 mass delta:        "
        f"{intervened_top_k - baseline_top_k:+.4f}"
    )

    print()

    print(
        "Baseline output tokens:  "
        f"{baseline['num_output_tokens']}"
    )

    print(
        "Intervened output tokens: "
        f"{intervened['num_output_tokens']}"
    )

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()