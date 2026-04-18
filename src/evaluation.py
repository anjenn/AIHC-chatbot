# Evaluation helpers for ranked diagnosis experiments.

from __future__ import annotations

import pandas as pd


def evaluate_prediction(
    gold_label: str,
    pred_top_labels: list[str],
    label_space: list[str],
    normalize_fn,
) -> dict[str, object]:
    # Compute top-1 and top-3 correctness against the allowed label set.
    gold_norm = normalize_fn(gold_label, label_space)

    top1_correct = int(len(pred_top_labels) > 0 and pred_top_labels[0] == gold_norm)
    top3_correct = int(gold_norm in pred_top_labels[:3])

    return {
        "gold_label": gold_norm,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
    }


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    # Return method-level summary metrics for a result table.
    return pd.DataFrame(
        [
            {
                "n_cases": len(results_df),
                "top1_accuracy": round(results_df["top1_correct"].mean(), 4),
                "top3_accuracy": round(results_df["top3_correct"].mean(), 4),
                "unknown_top1_rate": round((results_df["pred_1"] == "unknown").mean(), 4),
            }
        ]
    )


def show_failures(results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    # Return a compact table of failed top-1 cases for analysis.
    failures = results_df[results_df["top1_correct"] == 0].copy()
    return failures.head(n)[
        [
            "symptoms",
            "ground_truth",
            "pred_1",
            "pred_2",
            "pred_3",
            "conf_1",
            "conf_2",
            "conf_3",
            "guidance",
        ]
    ]
