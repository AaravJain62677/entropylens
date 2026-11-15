"""Analyze EntropyLens experiment results."""

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


def mean(rows, key):
    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def plot_signal(
    rows,
    key,
    ylabel,
    filename,
):
    groups = [
        "stable_correct",
        "stable_incorrect",
        "warranted_update",
        "unwarranted_update",
    ]

    grouped = defaultdict(list)

    for row in rows:
        value = row.get(key)

        if value is not None:
            grouped[row["outcome"]].append(
                value
            )

    data = []
    labels = []

    for group in groups:
        if grouped[group]:
            data.append(grouped[group])
            labels.append(
                group.replace("_", " ")
            )

    if not data:
        return

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        data,
        labels=labels,
        showmeans=True,
    )

    plt.ylabel(ylabel)
    plt.title(
        "EntropyLens epistemic outcome signal"
    )

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / filename,
        dpi=160,
    )

    plt.close()


def main():
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

    outcomes = Counter(
        row["outcome"]
        for row in intervention_rows
    )

    baseline_accuracy = (
        sum(
            row["baseline_correct"]
            for row in intervention_rows
            if row["baseline_correct"]
            is not None
        )
        / len(intervention_rows)
    )

    intervened_accuracy = (
        sum(
            row["intervened_correct"]
            for row in intervention_rows
            if row["intervened_correct"]
            is not None
        )
        / len(intervention_rows)
    )

    answer_change_rate = (
        sum(
            row["answer_changed"]
            for row in intervention_rows
            if row["answer_changed"]
            is not None
        )
        / len(intervention_rows)
    )

    warranted = outcomes[
        "warranted_update"
    ]

    unwarranted = outcomes[
        "unwarranted_update"
    ]

    total_updates = (
        warranted + unwarranted
    )

    epistemic_update_fidelity = (
        warranted / total_updates
        if total_updates
        else None
    )

    summary = {
        "total_episodes": len(rows),

        "control_episodes": len(
            control_rows
        ),

        "intervention_episodes": len(
            intervention_rows
        ),

        "baseline_accuracy":
            baseline_accuracy,

        "intervened_accuracy":
            intervened_accuracy,

        "answer_change_rate":
            answer_change_rate,

        "outcomes":
            dict(outcomes),

        "epistemic_update_fidelity":
            epistemic_update_fidelity,

        "control": {
            "mean_entropy_delta": mean(
                control_rows,
                "control_entropy_delta",
            ),

            "mean_probability_margin_delta":
                mean(
                    control_rows,
                    "control_probability_margin_delta",
                ),

            "mean_top_k_mass_delta":
                mean(
                    control_rows,
                    "control_top_k_mass_delta",
                ),
        },

        "interventions": {},
    }

    intervention_types = sorted(
        {
            row["intervention_type"]
            for row in intervention_rows
        }
    )

    for intervention_type in (
        intervention_types
    ):
        subset = [
            row
            for row in intervention_rows
            if row["intervention_type"]
            == intervention_type
        ]

        summary["interventions"][
            intervention_type
        ] = {
            "n": len(subset),

            "baseline_accuracy":
                sum(
                    row["baseline_correct"]
                    for row in subset
                )
                / len(subset),

            "intervened_accuracy":
                sum(
                    row["intervened_correct"]
                    for row in subset
                )
                / len(subset),

            "answer_change_rate":
                sum(
                    row["answer_changed"]
                    for row in subset
                )
                / len(subset),

            "mean_entropy_delta":
                mean(
                    subset,
                    "intervened_entropy_delta",
                ),

            "mean_probability_margin_delta":
                mean(
                    subset,
                    "intervened_probability_margin_delta",
                ),

            "mean_top_k_mass_delta":
                mean(
                    subset,
                    "intervened_top_k_mass_delta",
                ),
        }

    summary_path = (
        OUT_DIR / "summary.json"
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

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    plot_signal(
        intervention_rows,
        "intervened_entropy_delta",
        "Δ mean entropy",
        "delta_entropy.png",
    )

    plot_signal(
        intervention_rows,
        "intervened_probability_margin_delta",
        "Δ mean probability margin",
        "delta_margin.png",
    )


if __name__ == "__main__":
    main()