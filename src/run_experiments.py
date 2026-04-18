# Batch experiment runner for the healthcare chatbot core project.

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    EBM_DIR,
    EXPERIMENT_CASES,
    FEW_SHOT_EXAMPLES,
    KNOWLEDGE_TOP_K,
    OPENAI_MODEL,
    OUTPUT_DIR,
    REBALANCE_TOP_LABELS,
    SEED,
    SELF_CONSISTENCY_RUNS,
    SYNTHEA_DIR,
    TEST_SIZE,
    TOP_N_LABELS,
)
from src.data_prep import build_patient_dataset, filter_top_labels
from src.ebm_utils import load_knowledge_snippets, retrieve_knowledge
from src.evaluation import evaluate_prediction, summarize_results
from src.llm_runner import run_openai, run_self_consistency_ranked
from src.parsing import normalize_label_to_allowed, parse_ranked_output
from src.prompts import (
    prompt_few_shot_ranked,
    prompt_knowledge_ranked,
    prompt_zero_shot_ranked,
)
from src.retrieval import build_retrieved_few_shot_context


def load_split_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    # Load Synthea CSVs and return train/test splits for experiments.
    patients_df = pd.read_csv(SYNTHEA_DIR / "patients.csv")
    conditions_df = pd.read_csv(SYNTHEA_DIR / "conditions.csv")
    df = build_patient_dataset(patients_df, conditions_df)
    df, label_space = filter_top_labels(
        df,
        top_n=TOP_N_LABELS,
        rebalance=REBALANCE_TOP_LABELS,
    )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["diagnosis"],
    )

    return (
        df.reset_index(drop=True),
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        label_space,
    )


def build_eval_subset(test_df: pd.DataFrame, n_cases: int = EXPERIMENT_CASES) -> pd.DataFrame:
    # Create a deterministic bounded evaluation subset.
    if len(test_df) <= n_cases:
        return test_df.copy().reset_index(drop=True)

    return test_df.sample(n=n_cases, random_state=SEED).reset_index(drop=True)


def majority_baseline_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    # Compute simple non-LLM baselines for context in the report.
    label_counts = train_df["diagnosis"].value_counts()
    majority_label = label_counts.idxmax()
    top3_labels = label_counts.nlargest(3).index.tolist()

    return {
        "majority_label": majority_label,
        "majority_top1_accuracy": round(
            float((test_df["diagnosis"] == majority_label).mean()), 4
        ),
        "frequency_top3_labels": top3_labels,
        "frequency_top3_accuracy": round(
            float(test_df["diagnosis"].isin(top3_labels).mean()), 4
        ),
    }


def row_to_result_base(row: pd.Series, method: str, knowledge_used: list[str] | None = None):
    # Create a consistent result row scaffold.
    return {
        "method": method,
        "patient_index": int(row.name),
        "age": int(row["age"]),
        "sex": row["sex"],
        "symptoms": row["symptoms"],
        "ground_truth": row["diagnosis"],
        "knowledge_used": " | ".join(knowledge_used or []),
    }


def finalize_result_row(
    row: pd.Series,
    parsed: dict[str, object],
    method: str,
    label_space: list[str],
    knowledge_used: list[str] | None = None,
    error: str = "",
) -> dict[str, object]:
    # Merge parsed predictions with evaluation metrics.
    pred_labels = list(parsed["top_labels"])
    pred_confidences = list(parsed["top_confidences"])
    eval_row = evaluate_prediction(
        row["diagnosis"],
        pred_labels,
        label_space,
        normalize_label_to_allowed,
    )

    return {
        **row_to_result_base(row, method, knowledge_used),
        "pred_1": pred_labels[0],
        "pred_2": pred_labels[1],
        "pred_3": pred_labels[2],
        "conf_1": pred_confidences[0],
        "conf_2": pred_confidences[1],
        "conf_3": pred_confidences[2],
        "guidance": parsed["guidance"],
        "error": error,
        **eval_row,
    }


def run_baseline_experiment(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    label_space: list[str],
) -> pd.DataFrame:
    # Run the few-shot baseline experiment.
    results: list[dict[str, object]] = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Baseline few-shot"):
        few_shot_context = build_retrieved_few_shot_context(
            row, train_df, n=FEW_SHOT_EXAMPLES
        )
        prompt = prompt_few_shot_ranked(row, few_shot_context, label_space)

        try:
            raw_output = run_openai(prompt, model=OPENAI_MODEL, temperature=0)
            parsed = parse_ranked_output(raw_output, label_space, top_k=3)
            results.append(
                finalize_result_row(row, parsed, "baseline_few_shot", label_space)
            )
        except Exception as exc:
            parsed = {"top_labels": ["unknown"] * 3, "top_confidences": [0.0] * 3, "guidance": ""}
            results.append(
                finalize_result_row(
                    row,
                    parsed,
                    "baseline_few_shot",
                    label_space,
                    error=str(exc),
                )
            )

        time.sleep(0.1)

    return pd.DataFrame(results)


def run_knowledge_experiment(
    eval_df: pd.DataFrame,
    knowledge_df: pd.DataFrame,
    label_space: list[str],
) -> pd.DataFrame:
    # Run the prompt-level knowledge enrichment experiment.
    results: list[dict[str, object]] = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Knowledge prompt"):
        knowledge_used = retrieve_knowledge(
            row["symptoms"],
            knowledge_df,
            top_k=KNOWLEDGE_TOP_K,
            label_space=label_space,
        )
        prompt = prompt_knowledge_ranked(row, knowledge_used, label_space)

        try:
            raw_output = run_openai(prompt, model=OPENAI_MODEL, temperature=0)
            parsed = parse_ranked_output(raw_output, label_space, top_k=3)
            results.append(
                finalize_result_row(
                    row,
                    parsed,
                    "knowledge_prompt",
                    label_space,
                    knowledge_used=knowledge_used,
                )
            )
        except Exception as exc:
            parsed = {"top_labels": ["unknown"] * 3, "top_confidences": [0.0] * 3, "guidance": ""}
            results.append(
                finalize_result_row(
                    row,
                    parsed,
                    "knowledge_prompt",
                    label_space,
                    knowledge_used=knowledge_used,
                    error=str(exc),
                )
            )

        time.sleep(0.1)

    return pd.DataFrame(results)


def run_self_consistency_experiment(
    eval_df: pd.DataFrame,
    knowledge_df: pd.DataFrame,
    label_space: list[str],
) -> pd.DataFrame:
    # Run self-consistency with prompt-level knowledge enrichment.
    results: list[dict[str, object]] = []

    for _, row in tqdm(
        eval_df.iterrows(), total=len(eval_df), desc="Self-consistency knowledge"
    ):
        knowledge_used = retrieve_knowledge(
            row["symptoms"],
            knowledge_df,
            top_k=KNOWLEDGE_TOP_K,
            label_space=label_space,
        )
        prompt = prompt_knowledge_ranked(row, knowledge_used, label_space)

        try:
            sc_output = run_self_consistency_ranked(
                prompt,
                label_space,
                parse_ranked_output,
                n=SELF_CONSISTENCY_RUNS,
                model=OPENAI_MODEL,
            )
            parsed = {
                "top_labels": [item["label"] for item in sc_output["top_3"]],
                "top_confidences": [item["confidence"] for item in sc_output["top_3"]],
                "guidance": sc_output["guidance"],
            }
            results.append(
                finalize_result_row(
                    row,
                    parsed,
                    "self_consistency_knowledge",
                    label_space,
                    knowledge_used=knowledge_used,
                )
            )
        except Exception as exc:
            parsed = {"top_labels": ["unknown"] * 3, "top_confidences": [0.0] * 3, "guidance": ""}
            results.append(
                finalize_result_row(
                    row,
                    parsed,
                    "self_consistency_knowledge",
                    label_space,
                    knowledge_used=knowledge_used,
                    error=str(exc),
                )
            )

        time.sleep(0.1)

    return pd.DataFrame(results)


def build_method_summary(
    baseline_results: pd.DataFrame,
    knowledge_results: pd.DataFrame,
    self_consistency_results: pd.DataFrame,
) -> pd.DataFrame:
    # Build method-level summary metrics with error counts.
    summary_rows: list[dict[str, object]] = []

    for method_name, result_df in [
        ("baseline_few_shot", baseline_results),
        ("knowledge_prompt", knowledge_results),
        ("self_consistency_knowledge", self_consistency_results),
    ]:
        row = summarize_results(result_df).iloc[0].to_dict()
        row["method"] = method_name
        row["error_count"] = int(result_df["error"].fillna("").astype(str).str.strip().ne("").sum())
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)[
        ["method", "n_cases", "top1_accuracy", "top3_accuracy", "unknown_top1_rate", "error_count"]
    ]


def write_outputs(
    baseline_results: pd.DataFrame,
    knowledge_results: pd.DataFrame,
    self_consistency_results: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> None:
    # Persist experiment outputs to the standard outputs directory.
    baseline_results.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    knowledge_results.to_csv(OUTPUT_DIR / "knowledge_results.csv", index=False)
    self_consistency_results.to_csv(OUTPUT_DIR / "self_consistency_results.csv", index=False)
    method_summary.to_csv(OUTPUT_DIR / "method_summary.csv", index=False)


def run_all_experiments() -> dict[str, object]:
    # Run the bounded experiment suite and save CSV outputs.
    df, train_df, test_df, label_space = load_split_dataset()
    eval_df = build_eval_subset(test_df, n_cases=EXPERIMENT_CASES)
    knowledge_df = load_knowledge_snippets(EBM_DIR / "knowledge_snippets.csv")

    baseline_results = run_baseline_experiment(train_df, eval_df, label_space)
    knowledge_results = run_knowledge_experiment(eval_df, knowledge_df, label_space)
    self_consistency_results = run_self_consistency_experiment(
        eval_df, knowledge_df, label_space
    )
    method_summary = build_method_summary(
        baseline_results, knowledge_results, self_consistency_results
    )
    write_outputs(
        baseline_results, knowledge_results, self_consistency_results, method_summary
    )

    return {
        "full_df": df,
        "train_df": train_df,
        "test_df": test_df,
        "eval_df": eval_df,
        "label_space": label_space,
        "knowledge_df": knowledge_df,
        "baseline_results": baseline_results,
        "knowledge_results": knowledge_results,
        "self_consistency_results": self_consistency_results,
        "method_summary": method_summary,
        "majority_baseline": majority_baseline_metrics(train_df, eval_df),
    }


if __name__ == "__main__":
    artifacts = run_all_experiments()
    print("Saved outputs to:", OUTPUT_DIR)
    print(artifacts["method_summary"].to_string(index=False))
