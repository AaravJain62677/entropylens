"""Aggregate EntropyLens experiment results and create research plots."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "experiments"
    / "results.json"
)

OUT_DIR = (
    ROOT
    / "artifacts"
    / "experiments"
    / "analysis"
)


OUTCOME_ORDER = [
    "stable_correct",
    "stable_incorrect",
    "warranted_update",
    "unwarranted_update",
]


def mean(rows, key):
    """Calculate the mean of a numeric field."""

    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def evaluable_rows(rows):
    """
    Keep only episodes for which both baseline
    and comparison correctness are known.
    """

    return [
        row
        for row in rows
        if (
            row.get("baseline_correct")
            is not None
            and row.get("intervened_correct")
            is not None
        )
    ]


def comparable_rows(rows):
    """
    Keep only episodes where answer change
    can be evaluated.
    """

    return [
        row
        for row in rows
        if row.get("answer_changed")
        is not None
    ]


def calculate_accuracy(rows, key):
    """
    Calculate accuracy only over evaluable rows.

    Unknown / None values are excluded from
    the denominator.
    """

    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def calculate_update_fidelity(rows):
    """
    Calculate the fraction of belief updates that
    were warranted.

        warranted / (warranted + unwarranted)

    Stable answers are excluded.
    """

    warranted = sum(
        row["outcome"]
        == "warranted_update"
        for row in rows
    )

    unwarranted = sum(
        row["outcome"]
        == "unwarranted_update"
        for row in rows
    )

    total_updates = (
        warranted + unwarranted
    )

    if total_updates == 0:
        return None

    return warranted / total_updates


def plot_signal(
    rows,
    key,
    ylabel,
    filename,
):
    """
    Create a boxplot of a signal grouped
    by epistemic outcome.

    All four epistemic outcomes are retained
    when data exists for them.
    """

    values = defaultdict(list)

    for row in rows:
        value = row.get(key)

        if value is not None:
            values[row["outcome"]].append(
                value
            )

    data = []
    labels = []

    for outcome in OUTCOME_ORDER:
        if values[outcome]:
            data.append(
                values[outcome]
            )

            labels.append(
                f"{outcome.replace('_', ' ')}\n"
                f"(n={len(values[outcome])})"
            )

    if not data:
        return

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )

    plt.ylabel(ylabel)

    plt.title(
        "EntropyLens epistemic outcome signal"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / filename,
        dpi=160,
    )

    plt.close()


def build_condition_summary(rows):
    """
    Build intervention-condition-level metrics.
    """

    conditions = sorted(
        {
            row["intervention_type"]
            for row in rows
        }
    )

    summary = {}

    for condition in conditions:
        condition_rows = [
            row
            for row in rows
            if row["intervention_type"]
            == condition
        ]

        evaluable = evaluable_rows(
            condition_rows
        )

        summary[condition] = {
            "num_episodes": len(
                condition_rows
            ),

            "num_evaluable": len(
                evaluable
            ),

            "num_unknown": (
                len(condition_rows)
                - len(evaluable)
            ),

            "baseline_accuracy": (
                calculate_accuracy(
                    evaluable,
                    "baseline_correct",
                )
            ),

            "intervened_accuracy": (
                calculate_accuracy(
                    evaluable,
                    "intervened_correct",
                )
            ),

            "accuracy_change": (
                (
                    calculate_accuracy(
                        evaluable,
                        "intervened_correct",
                    )
                    - calculate_accuracy(
                        evaluable,
                        "baseline_correct",
                    )
                )
                if evaluable
                else None
            ),

            "answer_change_rate": (
                sum(
                    row["answer_changed"]
                    for row in comparable_rows(
                        condition_rows
                    )
                )
                / len(
                    comparable_rows(
                        condition_rows
                    )
                )
                if comparable_rows(
                    condition_rows
                )
                else None
            ),

            "warranted_updates": sum(
                row["outcome"]
                == "warranted_update"
                for row in evaluable
            ),

            "unwarranted_updates": sum(
                row["outcome"]
                == "unwarranted_update"
                for row in evaluable
            ),

            "update_fidelity": (
                calculate_update_fidelity(
                    evaluable
                )
            ),

            "mean_entropy_delta": mean(
                evaluable,
                "intervened_entropy_delta",
            ),

            "mean_probability_margin_delta": mean(
                evaluable,
                "intervened_probability_margin_delta",
            ),

            "mean_top_k_mass_delta": mean(
                evaluable,
                "intervened_top_k_mass_delta",
            ),
        }

    return summary


def main():
    """Analyze the completed experiment."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: "
            f"{RESULTS_PATH}"
        )

    with RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    rows = payload["results"]

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    control_rows = [
        row
        for row in rows
        if row["intervention_type"]
        == "control"
    ]

    intervention_rows = [
        row
        for row in rows
        if row["intervention_type"]
        != "control"
    ]

    evaluable_intervention_rows = (
        evaluable_rows(
            intervention_rows
        )
    )

    comparable_intervention_rows = (
        comparable_rows(
            intervention_rows
        )
    )

    outcomes = Counter(
        row["outcome"]
        for row in evaluable_intervention_rows
    )

    baseline_accuracy = (
        calculate_accuracy(
            evaluable_intervention_rows,
            "baseline_correct",
        )
    )

    intervened_accuracy = (
        calculate_accuracy(
            evaluable_intervention_rows,
            "intervened_correct",
        )
    )

    answer_change_rate = (
        sum(
            row["answer_changed"]
            for row in (
                comparable_intervention_rows
            )
        )
        / len(
            comparable_intervention_rows
        )
        if comparable_intervention_rows
        else None
    )

    summary = {
        "num_total_episodes": len(
            rows
        ),

        "num_control_episodes": len(
            control_rows
        ),

        "num_intervention_episodes": len(
            intervention_rows
        ),

        "num_evaluable_intervention_episodes": len(
            evaluable_intervention_rows
        ),

        "num_unknown_intervention_episodes": (
            len(intervention_rows)
            - len(
                evaluable_intervention_rows
            )
        ),

        "baseline_accuracy": (
            baseline_accuracy
        ),

        "intervened_accuracy": (
            intervened_accuracy
        ),

        "accuracy_change": (
            intervened_accuracy
            - baseline_accuracy
            if (
                baseline_accuracy
                is not None
                and intervened_accuracy
                is not None
            )
            else None
        ),

        "answer_change_rate": (
            answer_change_rate
        ),

        "outcomes": dict(
            outcomes
        ),

        "epistemic_update_fidelity": (
            calculate_update_fidelity(
                evaluable_intervention_rows
            )
        ),

        "control_mean_entropy_delta": (
            mean(
                control_rows,
                "control_entropy_delta",
            )
        ),

        "control_mean_probability_margin_delta": (
            mean(
                control_rows,
                "control_probability_margin_delta",
            )
        ),

        "control_mean_top_k_mass_delta": (
            mean(
                control_rows,
                "control_top_k_mass_delta",
            )
        ),

        "intervention_mean_entropy_delta": (
            mean(
                evaluable_intervention_rows,
                "intervened_entropy_delta",
            )
        ),

        "intervention_mean_probability_margin_delta": (
            mean(
                evaluable_intervention_rows,
                "intervened_probability_margin_delta",
            )
        ),

        "intervention_mean_top_k_mass_delta": (
            mean(
                evaluable_intervention_rows,
                "intervened_top_k_mass_delta",
            )
        ),
    }

    summary_path = (
        OUT_DIR
        / "summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    condition_summary = (
        build_condition_summary(
            intervention_rows
        )
    )

    condition_summary_path = (
        OUT_DIR
        / "condition_summary.json"
    )

    with condition_summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            condition_summary,
            f,
            indent=2,
        )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "\nCondition-level analysis:"
    )

    for condition, metrics in (
        condition_summary.items()
    ):
        print(
            f"\n{condition}"
        )

        print(
            f"  evaluable: "
            f"{metrics['num_evaluable']}"
        )

        print(
            f"  baseline accuracy: "
            f"{metrics['baseline_accuracy']}"
        )

        print(
            f"  intervened accuracy: "
            f"{metrics['intervened_accuracy']}"
        )

        print(
            f"  accuracy change: "
            f"{metrics['accuracy_change']}"
        )

        print(
            f"  answer change rate: "
            f"{metrics['answer_change_rate']}"
        )

        print(
            f"  warranted updates: "
            f"{metrics['warranted_updates']}"
        )

        print(
            f"  unwarranted updates: "
            f"{metrics['unwarranted_updates']}"
        )

        print(
            f"  update fidelity: "
            f"{metrics['update_fidelity']}"
        )

    plot_signal(
        evaluable_intervention_rows,
        "intervened_entropy_delta",
        "Mean entropy change",
        "delta_entropy.png",
    )

    plot_signal(
        evaluable_intervention_rows,
        "intervened_probability_margin_delta",
        "Probability-margin change",
        "delta_margin.png",
    )

    plot_signal(
        evaluable_intervention_rows,
        "intervened_top_k_mass_delta",
        "Top-k probability-mass change",
        "delta_top_k_mass.png",
    )

    print(
        "\nAnalysis written to:"
    )

    print(
        OUT_DIR
    )


if __name__ == "__main__":
    main()