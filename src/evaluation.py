# Evaluation helpers for ranked diagnosis experiments.

from __future__ import annotations

import ast
import json

import matplotlib.pyplot as plt
import pandas as pd


def ensure_list(value):
    # Convert list-like values from CSV-friendly storage back into Python lists.
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    return parsed
            return []
        return [stripped]
    return []


def _series_or_default(df: pd.DataFrame, column: str, default):
    # Return a column when present, otherwise a default-filled series.
    if column in df.columns:
        return df[column]
    if isinstance(default, pd.Series):
        return default.reindex(df.index)
    return pd.Series([default] * len(df), index=df.index)


def _extract_pred_top3(row: pd.Series) -> list[str]:
    # Collect ranked predictions from either list-based or flat columns.
    if "pred_top3" in row.index:
        pred_top3 = ensure_list(row["pred_top3"])
        if pred_top3:
            return [str(item) for item in pred_top3]

    labels = []
    for column in ("pred_1", "pred_2", "pred_3"):
        value = row.get(column, "unknown")
        labels.append(str(value) if pd.notna(value) else "unknown")
    return labels


def extract_top1_confidence(parsed_ranked):
    # Extract the first ranked confidence from a list of diagnosis dicts.
    ranked_items = ensure_list(parsed_ranked)
    if not ranked_items:
        return None

    first = ranked_items[0]
    if isinstance(first, dict):
        try:
            return float(first.get("confidence"))
        except (TypeError, ValueError):
            return None
    return None


def add_topk_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize result tables into a shared evaluation schema.
    out = df.copy()

    if "true_label" not in out.columns:
        if "ground_truth" in out.columns:
            out["true_label"] = out["ground_truth"]
        else:
            out["true_label"] = ""

    if "pred_top1" not in out.columns:
        if "primary_diagnosis" in out.columns:
            out["pred_top1"] = out["primary_diagnosis"]
        elif "pred_1" in out.columns:
            out["pred_top1"] = out["pred_1"]
        else:
            out["pred_top1"] = "unknown"

    out["pred_top3"] = out.apply(_extract_pred_top3, axis=1)

    if "top1_confidence" not in out.columns:
        if "conf_1" in out.columns:
            out["top1_confidence"] = pd.to_numeric(out["conf_1"], errors="coerce")
        elif "ranked_diagnoses" in out.columns:
            out["top1_confidence"] = out["ranked_diagnoses"].apply(extract_top1_confidence)
        else:
            out["top1_confidence"] = pd.NA
    else:
        out["top1_confidence"] = pd.to_numeric(out["top1_confidence"], errors="coerce")

    if "consultation_guidance" not in out.columns:
        out["consultation_guidance"] = _series_or_default(out, "guidance", "")

    out["top1_correct"] = (out["true_label"] == out["pred_top1"]).astype(int)
    out["top3_correct"] = out.apply(
        lambda row: int(row["true_label"] in row["pred_top3"][:3]),
        axis=1,
    )
    return out


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


def summarize_metrics(df: pd.DataFrame) -> pd.Series:
    # Build overall top-k accuracy metrics from a result table.
    df = add_topk_columns(df)
    unknown_mask = df["pred_top1"].fillna("unknown").astype(str).eq("unknown")
    return pd.Series(
        {
            "n": len(df),
            "top1_accuracy": round(df["top1_correct"].mean(), 4),
            "top3_accuracy": round(df["top3_correct"].mean(), 4),
            "unknown_top1_rate": round(unknown_mask.mean(), 4),
        }
    )


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    # Return method-level summary metrics for a result table.
    summary = summarize_metrics(results_df).to_dict()
    summary["n_cases"] = summary["n"]
    return pd.DataFrame([summary])


def per_label_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    # Build per-label top-1 and top-3 accuracy metrics.
    df = add_topk_columns(df)
    grouped = df.groupby("true_label").agg(
        n=("true_label", "size"),
        top1_accuracy=("top1_correct", "mean"),
        top3_accuracy=("top3_correct", "mean"),
    )
    return grouped.sort_values(["top1_accuracy", "n"], ascending=[True, False]).reset_index()


def confusion_matrix_table(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    # Build a raw confusion matrix from true labels and top-1 predictions.
    df = add_topk_columns(df)
    cm = pd.crosstab(
        pd.Categorical(df["true_label"], categories=labels),
        pd.Categorical(df["pred_top1"], categories=labels),
        dropna=False,
    )
    cm.index.name = "true_label"
    cm.columns.name = "predicted_label"
    return cm


def normalize_confusion_rows(cm: pd.DataFrame) -> pd.DataFrame:
    # Row-normalize a confusion matrix for easier error-pattern reading.
    row_sums = cm.sum(axis=1).replace(0, 1)
    return cm.div(row_sums, axis=0)


def plot_confusion_matrix(cm: pd.DataFrame, title: str = "Confusion Matrix"):
    # Plot a confusion matrix with inline values.
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm.values, aspect="auto")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm.iloc[i, j]
            display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            ax.text(j, i, display_value, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig, ax


def add_confidence_buckets(
    df: pd.DataFrame,
    confidence_col: str = "top1_confidence",
) -> pd.DataFrame:
    # Bucket predictions by their top-1 confidence proxy.
    out = add_topk_columns(df)
    out["confidence_bucket"] = pd.cut(
        out[confidence_col],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )
    return out


def confidence_accuracy_table(
    df: pd.DataFrame,
    confidence_col: str = "top1_confidence",
) -> pd.DataFrame:
    # Compare confidence buckets against observed accuracy.
    df = add_confidence_buckets(df, confidence_col=confidence_col)
    table = df.groupby("confidence_bucket", observed=False).agg(
        n=("top1_correct", "size"),
        mean_confidence=(confidence_col, "mean"),
        top1_accuracy=("top1_correct", "mean"),
        top3_accuracy=("top3_correct", "mean"),
    )
    return table.reset_index()


def plot_confidence_vs_accuracy(table: pd.DataFrame):
    # Plot confidence against observed top-1 accuracy.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["mean_confidence"], table["top1_accuracy"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed top-1 accuracy")
    ax.set_title("Confidence vs Observed Accuracy")
    plt.tight_layout()
    return fig, ax


def prepare_error_review_sheet(df: pd.DataFrame) -> pd.DataFrame:
    # Prepare a manual error review sheet with failure-category placeholders.
    df = add_topk_columns(df)
    errors = df[df["top1_correct"] == 0].copy()

    keep_cols = [
        column
        for column in [
            "case_id",
            "patient_index",
            "symptoms",
            "true_label",
            "pred_top1",
            "pred_top3",
            "top1_confidence",
            "consultation_guidance",
        ]
        if column in errors.columns
    ]

    errors = errors[keep_cols].copy()
    errors["failure_category"] = ""
    errors["notes"] = ""
    return errors


def show_failures(results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    # Return a compact table of failed top-1 cases for analysis.
    failures = add_topk_columns(results_df)
    failures = failures[failures["top1_correct"] == 0].copy()
    failures["ground_truth"] = _series_or_default(failures, "ground_truth", failures["true_label"])
    failures["pred_1"] = _series_or_default(failures, "pred_1", failures["pred_top1"])
    failures["pred_2"] = failures["pred_top3"].apply(
        lambda labels: labels[1] if len(labels) > 1 else "unknown"
    )
    failures["pred_3"] = failures["pred_top3"].apply(
        lambda labels: labels[2] if len(labels) > 2 else "unknown"
    )
    failures["conf_1"] = _series_or_default(failures, "conf_1", failures["top1_confidence"])
    failures["conf_2"] = _series_or_default(failures, "conf_2", 0.0)
    failures["conf_3"] = _series_or_default(failures, "conf_3", 0.0)
    failures["guidance"] = _series_or_default(
        failures,
        "guidance",
        failures["consultation_guidance"],
    )

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
