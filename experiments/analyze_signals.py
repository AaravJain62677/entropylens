"""Statistical analysis of internal-signal changes."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


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

SIGNALS = {
    "entropy": "intervened_entropy_delta",
    "probability_margin": (
        "intervened_probability_margin_delta"
    ),
    "top_k_mass": (
        "intervened_top_k_mass_delta"
    ),
}

OUTCOMES = [
    "stable_correct",
    "stable_incorrect",
    "warranted_update",
    "unwarranted_update",
]


def load_results():
    """Load experiment results."""

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

    return [
        row
        for row in payload["results"]
        if (
            row["intervention_type"]
            != "control"
        )
        and row["outcome"] != "unknown"
    ]


def describe(values):
    """Return descriptive statistics."""

    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "std": (
            stdev(values)
            if len(values) >= 2
            else 0.0
        ),
        "min": min(values),
        "max": max(values),
    }


def cohens_d(first, second):
    """
    Calculate Cohen's d for two independent groups.

    Positive values mean the first group has
    a larger mean.
    """

    if (
        len(first) < 2
        or len(second) < 2
    ):
        return None

    mean_first = mean(first)
    mean_second = mean(second)

    var_first = (
        stdev(first) ** 2
    )

    var_second = (
        stdev(second) ** 2
    )

    pooled_variance = (
        (
            (len(first) - 1)
            * var_first
        )
        + (
            (len(second) - 1)
            * var_second
        )
    ) / (
        len(first)
        + len(second)
        - 2
    )

    if pooled_variance == 0:
        return 0.0

    return (
        mean_first
        - mean_second
    ) / math.sqrt(
        pooled_variance
    )


def cliffs_delta(first, second):
    """
    Calculate Cliff's delta.

    Positive values mean values in `first`
    tend to be larger than values in `second`.
    """

    if not first or not second:
        return None

    greater = 0
    less = 0

    for x in first:
        for y in second:
            if x > y:
                greater += 1
            elif x < y:
                less += 1

    return (
        greater - less
    ) / (
        len(first)
        * len(second)
    )


def bootstrap_mean_difference(
    first,
    second,
    iterations=5000,
    seed=42,
):
    """
    Bootstrap the difference in means:

        mean(first) - mean(second)

    Returns the observed difference and
    a percentile 95% confidence interval.
    """

    if (
        not first
        or not second
    ):
        return None

    rng = random.Random(seed)

    observed = (
        mean(first)
        - mean(second)
    )

    differences = []

    for _ in range(iterations):
        first_sample = [
            first[
                rng.randrange(
                    len(first)
                )
            ]
            for _ in first
        ]

        second_sample = [
            second[
                rng.randrange(
                    len(second)
                )
            ]
            for _ in second
        ]

        differences.append(
            mean(first_sample)
            - mean(second_sample)
        )

    differences.sort()

    lower_index = int(
        0.025
        * len(differences)
    )

    upper_index = int(
        0.975
        * len(differences)
    )

    upper_index = min(
        upper_index,
        len(differences) - 1,
    )

    return {
        "observed_difference": (
            observed
        ),
        "ci_95_lower": (
            differences[
                lower_index
            ]
        ),
        "ci_95_upper": (
            differences[
                upper_index
            ]
        ),
    }


def permutation_p_value(
    first,
    second,
    iterations=5000,
    seed=42,
):
    """
    Two-sided permutation test for
    difference in means.
    """

    if (
        not first
        or not second
    ):
        return None

    observed = abs(
        mean(first)
        - mean(second)
    )

    combined = (
        list(first)
        + list(second)
    )

    n_first = len(first)

    rng = random.Random(seed)

    extreme = 0

    for _ in range(iterations):
        shuffled = combined.copy()

        rng.shuffle(shuffled)

        perm_first = (
            shuffled[:n_first]
        )

        perm_second = (
            shuffled[n_first:]
        )

        difference = abs(
            mean(perm_first)
            - mean(perm_second)
        )

        if difference >= observed:
            extreme += 1

    return (
        extreme + 1
    ) / (
        iterations + 1
    )


def analyze_outcomes(rows):
    """Calculate signal statistics by epistemic outcome."""

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["outcome"]
        ].append(row)

    output = {}

    for outcome in OUTCOMES:
        outcome_rows = grouped[
            outcome
        ]

        output[outcome] = {
            signal_name: describe(
                [
                    row[key]
                    for row in outcome_rows
                    if row.get(key)
                    is not None
                ]
            )
            for signal_name, key
            in SIGNALS.items()
        }

    return output


def analyze_update_discrimination(
    rows,
):
    """
    Compare warranted vs unwarranted updates
    for each internal signal.
    """

    warranted = [
        row
        for row in rows
        if row["outcome"]
        == "warranted_update"
    ]

    unwarranted = [
        row
        for row in rows
        if row["outcome"]
        == "unwarranted_update"
    ]

    output = {
        "n_warranted": len(
            warranted
        ),
        "n_unwarranted": len(
            unwarranted
        ),
        "signals": {},
    }

    for signal_name, key in SIGNALS.items():
        warranted_values = [
            row[key]
            for row in warranted
            if row.get(key)
            is not None
        ]

        unwarranted_values = [
            row[key]
            for row in unwarranted
            if row.get(key)
            is not None
        ]

        output["signals"][
            signal_name
        ] = {
            "warranted": describe(
                warranted_values
            ),
            "unwarranted": describe(
                unwarranted_values
            ),
            "cohens_d": cohens_d(
                warranted_values,
                unwarranted_values,
            ),
            "cliffs_delta": cliffs_delta(
                warranted_values,
                unwarranted_values,
            ),
            "bootstrap_mean_difference": (
                bootstrap_mean_difference(
                    warranted_values,
                    unwarranted_values,
                )
            ),
            "permutation_p_value": (
                permutation_p_value(
                    warranted_values,
                    unwarranted_values,
                )
            ),
        }

    return output


def analyze_conditions(rows):
    """Calculate signal means by intervention condition."""

    conditions = sorted(
        {
            row["intervention_type"]
            for row in rows
        }
    )

    output = {}

    for condition in conditions:
        condition_rows = [
            row
            for row in rows
            if row["intervention_type"]
            == condition
        ]

        output[condition] = {
            signal_name: describe(
                [
                    row[key]
                    for row in condition_rows
                    if row.get(key)
                    is not None
                ]
            )
            for signal_name, key
            in SIGNALS.items()
        }

    return output


def main():
    """Run Phase 4.1 statistical analysis."""

    rows = load_results()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outcome_analysis = (
        analyze_outcomes(rows)
    )

    update_analysis = (
        analyze_update_discrimination(
            rows
        )
    )

    condition_analysis = (
        analyze_conditions(rows)
    )

    output = {
        "metadata": {
            "num_evaluable_episodes": (
                len(rows)
            ),
            "signals": list(
                SIGNALS.keys()
            ),
            "outcomes": OUTCOMES,
            "bootstrap_iterations": 5000,
            "permutation_iterations": 5000,
            "random_seed": 42,
        },

        "outcome_statistics": (
            outcome_analysis
        ),

        "warranted_vs_unwarranted": (
            update_analysis
        ),

        "condition_statistics": (
            condition_analysis
        ),
    }

    output_path = (
        OUT_DIR
        / "signal_analysis.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(
        "Signal Analysis"
    )
    print(
        "================================"
    )

    print(
        f"\nEvaluable episodes: "
        f"{len(rows)}"
    )

    print(
        "\nOutcome statistics:"
    )

    for outcome in OUTCOMES:
        print(
            f"\n{outcome}"
        )

        for signal_name in SIGNALS:
            stats = (
                outcome_analysis[
                    outcome
                ][signal_name]
            )

            print(
                f"  {signal_name}: "
                f"n={stats['n']}, "
                f"mean={stats['mean']:.6f}, "
                f"median={stats['median']:.6f}"
            )

    print(
        "\nWarranted vs unwarranted updates:"
    )

    print(
        f"  warranted: "
        f"{update_analysis['n_warranted']}"
    )

    print(
        f"  unwarranted: "
        f"{update_analysis['n_unwarranted']}"
    )

    for signal_name, stats in (
        update_analysis[
            "signals"
        ].items()
    ):
        print(
            f"\n  {signal_name}"
        )

        print(
            f"    Cohen's d: "
            f"{stats['cohens_d']:.6f}"
        )

        print(
            f"    Cliff's delta: "
            f"{stats['cliffs_delta']:.6f}"
        )

        bootstrap = (
            stats[
                "bootstrap_mean_difference"
            ]
        )

        print(
            "    Mean difference "
            "(warranted - unwarranted): "
            f"{bootstrap['observed_difference']:.6f}"
        )

        print(
            "    Bootstrap 95% CI: "
            f"[{bootstrap['ci_95_lower']:.6f}, "
            f"{bootstrap['ci_95_upper']:.6f}]"
        )

        print(
            f"    Permutation p-value: "
            f"{stats['permutation_p_value']:.6f}"
        )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()