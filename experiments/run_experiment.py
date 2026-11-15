"""Run the minimum EntropyLens epistemic-stability experiment."""

import json
from pathlib import Path

from inference.loader import load_model_and_tokenizer
from interventions.base import (
    Intervention,
    InterventionType,
)
from services.episode import run_episode


ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = (
    ROOT
    / "experiments"
    / "questions.json"
)

INTERVENTIONS_PATH = (
    ROOT
    / "experiments"
    / "interventions.json"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "results.json"
)

MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

STRATEGY_NAME = "greedy"

MAX_NEW_TOKENS = 20


def load_json(path: Path):
    """Load a JSON file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def resolve_content(
    content: str,
    question: dict,
) -> str:
    """
    Resolve intervention placeholders using
    question-specific answers.
    """

    return content.format(
        answer=question["answer"],
        alternative_answer=question[
            "alternative_answer"
        ],
    )


def flatten_result(
    result: dict,
    question_id: str,
    category: str,
    intervention_id: str,
) -> dict:
    """
    Flatten an episode result into one
    experiment row.
    """

    intervention_type = result[
        "intervention"
    ]["type"]

    comparison_key = (
        "control"
        if intervention_type == "control"
        else "intervened"
    )

    row = {
        "question_id": question_id,
        "category": category,
        "intervention_id": intervention_id,
        "intervention_type": (
            intervention_type
        ),

        "question": result[
            "question"
        ],

        "ground_truth": result[
            "ground_truth"
        ],

        "baseline_answer": result[
            "baseline_answer"
        ],

        "baseline_correct": result[
            "baseline_correct"
        ],

        "answer_changed": result[
            "answer_changed"
        ],

        "outcome": result[
            "outcome"
        ],

        "baseline_mean_entropy": result[
            "baseline"
        ]["mean_entropy"],

        "baseline_mean_probability_margin": (
            result["baseline"][
                "mean_probability_margin"
            ]
        ),

        "baseline_mean_top_k_mass": (
            result["baseline"][
                "mean_top_k_mass"
            ]
        ),

        f"{comparison_key}_answer": (
            result[
                f"{comparison_key}_answer"
            ]
        ),

        f"{comparison_key}_correct": (
            result[
                f"{comparison_key}_correct"
            ]
        ),

        f"{comparison_key}_mean_entropy": (
            result[
                comparison_key
            ]["mean_entropy"]
        ),

        f"{comparison_key}_mean_probability_margin": (
            result[
                comparison_key
            ]["mean_probability_margin"]
        ),

        f"{comparison_key}_mean_top_k_mass": (
            result[
                comparison_key
            ]["mean_top_k_mass"]
        ),

        f"{comparison_key}_entropy_delta": (
            result[
                f"{comparison_key}_entropy_delta"
            ]
        ),

        f"{comparison_key}_probability_margin_delta": (
            result[
                f"{comparison_key}_probability_margin_delta"
            ]
        ),

        f"{comparison_key}_top_k_mass_delta": (
            result[
                f"{comparison_key}_top_k_mass_delta"
            ]
        ),

        f"{comparison_key}_repetition_rate": (
            result[
                comparison_key
            ]["repetition_rate"]
        ),

        "baseline_repetition_rate": (
            result["baseline"][
                "repetition_rate"
            ]
        ),

        "baseline_num_output_tokens": (
            result["baseline"][
                "num_output_tokens"
            ]
        ),

        f"{comparison_key}_num_output_tokens": (
            result[
                comparison_key
            ]["num_output_tokens"]
        ),
    }

    return row


def main():
    """Run the complete experiment."""

    questions = load_json(
        QUESTIONS_PATH
    )

    interventions = load_json(
        INTERVENTIONS_PATH
    )

    print(
        f"Loading model: {MODEL_NAME}"
    )

    model, tokenizer = (
        load_model_and_tokenizer(
            MODEL_NAME
        )
    )

    results = []

    total = (
        len(questions)
        * len(interventions)
    )

    completed = 0

    for question in questions:
        for intervention_config in (
            interventions
        ):
            completed += 1

            intervention_type = (
                InterventionType(
                    intervention_config[
                        "type"
                    ]
                )
            )

            content = resolve_content(
                intervention_config[
                    "content"
                ],
                question,
            )

            intervention = Intervention(
                type=intervention_type,
                content=content,
            )

            print(
                f"[{completed}/{total}] "
                f"{question['id']} / "
                f"{intervention_config['type']}"
            )

            episode = run_episode(
                question=question[
                    "question"
                ],
                intervention=intervention,
                strategy_name=STRATEGY_NAME,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=(
                    MAX_NEW_TOKENS
                ),
                ground_truth=question[
                    "answer"
                ],
                acceptable_answers=(
                    question.get(
                        "acceptable_answers"
                    )
                ),
            )

            row = flatten_result(
                episode,
                question_id=question[
                    "id"
                ],
                category=question[
                    "category"
                ],
                intervention_id=(
                    intervention_config[
                        "id"
                    ]
                ),
            )

            results.append(row)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metadata": {
            "model": MODEL_NAME,
            "strategy": STRATEGY_NAME,
            "max_new_tokens": (
                MAX_NEW_TOKENS
            ),
            "num_questions": len(
                questions
            ),
            "num_conditions": len(
                interventions
            ),
            "num_episodes": len(
                results
            ),
            "conditions": [
                intervention[
                    "type"
                ]
                for intervention in (
                    interventions
                )
            ],
        },
        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved {len(results)} episodes "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()