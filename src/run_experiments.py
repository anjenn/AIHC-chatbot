# Batch experiment runner for the healthcare chatbot core project.

from __future__ import annotations

import json
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import (
    CANDIDATE_NEIGHBORS,
    CANDIDATE_TOP_K,
    EBM_DIR,
    EVIDENCE_TOP_K,
    EXPERIMENT_CASES,
    FEW_SHOT_EXAMPLES,
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
from src.ebm_utils import load_knowledge_snippets
from src.evaluation import (
    confidence_accuracy_table,
    confusion_matrix_table,
    evaluate_prediction,
    normalize_confusion_rows,
    per_label_accuracy,
    prepare_error_review_sheet,
    summarize_results,
)
from src.llm_runner import run_openai, run_self_consistency_ranked
from src.parsing import normalize_label_to_allowed, parse_ranked_output
from src.prompts import (
    DEFAULT_VALIDATION_QUESTION,
    build_case_query,
    build_retrieve_then_reason_prompt,
    build_structured_patient_summary,
)
from src.retrieval import (
    build_retrieved_few_shot_context,
    retrieve_candidate_labels,
    retrieve_candidate_labels_from_examples,
    retrieve_evidence_snippets,
    retrieve_similar_case_summaries,
    retrieve_similar_examples,
)


def _default_parsed_output(top_k: int = 3) -> dict[str, object]:
    # Create a stable empty parsed output for error paths.
    ranked_diagnoses = [
        {
            "label": "unknown",
            "confidence": 0.0,
            "display_score": 0.0,
            "supporting_evidence": [],
            "missing_or_uncertain": [],
        }
        for _ in range(top_k)
    ]
    return {
        "primary_diagnosis": "unknown",
        "ranked_diagnoses": ranked_diagnoses,
        "top_labels": ["unknown"] * top_k,
        "top_confidences": [0.0] * top_k,
        "top_display_scores": [0.0] * top_k,
        "consultation_guidance": "",
        "guidance": "",
        "similar_cases": [],
        "validation_question": DEFAULT_VALIDATION_QUESTION,
    }


def _json_dumps(value: object) -> str:
    # Serialize lists and dicts into CSV-friendly JSON.
    return json.dumps(value, ensure_ascii=True)


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
            float((test_df["diagnosis"] == majority_label).mean()),
            4,
        ),
        "frequency_top3_labels": top3_labels,
        "frequency_top3_accuracy": round(
            float(test_df["diagnosis"].isin(top3_labels).mean()),
            4,
        ),
    }


def prepare_retrieve_then_reason_case(
    row: pd.Series,
    label_space: list[str],
    train_df: pd.DataFrame | None = None,
    knowledge_snippets_df: pd.DataFrame | None = None,
    few_shot_context: str = "",
    candidate_k: int = CANDIDATE_TOP_K,
    evidence_k: int = EVIDENCE_TOP_K,
) -> dict[str, object]:
    # Build the full retrieval and prompt context before LLM execution.
    patient_summary = build_structured_patient_summary(
        age=row.get("age"),
        sex=row.get("sex"),
        symptoms=row.get("symptoms", ""),
        symptoms_started=row.get("symptoms_started", row.get("when_symptoms_started", "")),
        severity=row.get("severity", ""),
        fever=row.get("fever", ""),
        breathing_difficulty=row.get("breathing_difficulty", ""),
        chest_pain=row.get("chest_pain", ""),
        bleeding=row.get("bleeding", ""),
        confusion=row.get("confusion", ""),
        existing_conditions=row.get("existing_conditions", ""),
        recent_worsening=row.get("recent_worsening", ""),
    )
    case_query = build_case_query(
        age=row.get("age"),
        sex=row.get("sex"),
        symptoms=row.get("symptoms", ""),
        symptoms_started=row.get("symptoms_started", row.get("when_symptoms_started", "")),
        severity=row.get("severity", ""),
        fever=row.get("fever", ""),
        breathing_difficulty=row.get("breathing_difficulty", ""),
        chest_pain=row.get("chest_pain", ""),
        bleeding=row.get("bleeding", ""),
        confusion=row.get("confusion", ""),
        existing_conditions=row.get("existing_conditions", ""),
        recent_worsening=row.get("recent_worsening", ""),
        patient_summary=patient_summary,
    )

    similar_examples = None
    if train_df is not None:
        similar_examples = retrieve_similar_examples(
            row=row,
            train_df=train_df,
            n=max(CANDIDATE_NEIGHBORS, 6),
        )
        candidate_labels = retrieve_candidate_labels_from_examples(
            row=row,
            train_df=train_df,
            label_space=label_space,
            k_neighbors=CANDIDATE_NEIGHBORS,
            k=candidate_k,
        )
    else:
        candidate_labels = retrieve_candidate_labels(
            symptoms=str(row.get("symptoms", "")),
            label_space=label_space,
            k=candidate_k,
            case_query=case_query,
        )

    similar_cases: list[str] = []
    if train_df is not None:
        similar_cases = retrieve_similar_case_summaries(row=row, train_df=train_df, n=2)

    evidence_snippets = retrieve_evidence_snippets(
        case_query=case_query,
        knowledge_snippets=knowledge_snippets_df,
        k=evidence_k,
        label_space=label_space,
        candidate_labels=candidate_labels,
        row=row,
        train_df=train_df,
        similar_examples=similar_examples,
    )

    prompt = build_retrieve_then_reason_prompt(
        age=row.get("age"),
        sex=row.get("sex"),
        symptoms=row.get("symptoms", ""),
        symptoms_started=row.get("symptoms_started", row.get("when_symptoms_started", "")),
        severity=row.get("severity", ""),
        fever=row.get("fever", ""),
        breathing_difficulty=row.get("breathing_difficulty", ""),
        chest_pain=row.get("chest_pain", ""),
        bleeding=row.get("bleeding", ""),
        confusion=row.get("confusion", ""),
        existing_conditions=row.get("existing_conditions", ""),
        recent_worsening=row.get("recent_worsening", ""),
        candidate_labels=candidate_labels,
        evidence_snippets=evidence_snippets,
        few_shot_context=few_shot_context,
        similar_cases=similar_cases,
        patient_summary=patient_summary,
    )

    return {
        "patient_summary": patient_summary,
        "case_query": case_query,
        "candidate_labels": candidate_labels,
        "evidence_snippets": evidence_snippets,
        "similar_cases": similar_cases,
        "few_shot_context": few_shot_context,
        "prompt": prompt,
    }


def run_retrieve_then_reason_case(
    row: pd.Series,
    label_space: list[str],
    llm_call_fn,
    train_df: pd.DataFrame | None = None,
    knowledge_snippets_df: pd.DataFrame | None = None,
    few_shot_context: str = "",
    candidate_k: int = CANDIDATE_TOP_K,
    evidence_k: int = EVIDENCE_TOP_K,
) -> dict[str, object]:
    # Run the full retrieve-then-reason pipeline for one case.
    prepared = prepare_retrieve_then_reason_case(
        row=row,
        label_space=label_space,
        train_df=train_df,
        knowledge_snippets_df=knowledge_snippets_df,
        few_shot_context=few_shot_context,
        candidate_k=candidate_k,
        evidence_k=evidence_k,
    )
    raw_output = llm_call_fn(prepared["prompt"])
    parsed = parse_ranked_output(raw_output, label_space, top_k=3)

    return {
        **prepared,
        "raw_output": raw_output,
        "parsed": parsed,
        "pred_top1": parsed["primary_diagnosis"],
        "pred_top3": parsed["top_labels"][:3],
        "pred_confidences": parsed["top_confidences"][:3],
    }


def finalize_result_row(
    row: pd.Series,
    method: str,
    case_artifact: dict[str, object],
    label_space: list[str],
    error: str = "",
) -> dict[str, object]:
    # Merge parsed predictions with retrieval context and evaluation fields.
    parsed = case_artifact.get("parsed", _default_parsed_output())
    pred_labels = list(parsed["top_labels"])[:3]
    pred_confidences = list(parsed["top_confidences"])[:3]
    pred_display_scores = list(parsed.get("top_display_scores", pred_confidences))[:3]

    while len(pred_labels) < 3:
        pred_labels.append("unknown")
        pred_confidences.append(0.0)
        pred_display_scores.append(0.0)

    eval_row = evaluate_prediction(
        row["diagnosis"],
        pred_labels,
        label_space,
        normalize_label_to_allowed,
    )

    return {
        "method": method,
        "patient_index": int(row.name),
        "age": int(row["age"]),
        "sex": row["sex"],
        "symptoms": row["symptoms"],
        "true_label": row["diagnosis"],
        "ground_truth": row["diagnosis"],
        "patient_summary": case_artifact.get("patient_summary", ""),
        "case_query": case_artifact.get("case_query", ""),
        "candidate_labels": _json_dumps(case_artifact.get("candidate_labels", [])),
        "evidence_snippets": _json_dumps(case_artifact.get("evidence_snippets", [])),
        "knowledge_used": " | ".join(case_artifact.get("evidence_snippets", [])),
        "similar_cases": _json_dumps(parsed.get("similar_cases", case_artifact.get("similar_cases", []))),
        "validation_question": parsed.get("validation_question", DEFAULT_VALIDATION_QUESTION),
        "few_shot_context": case_artifact.get("few_shot_context", ""),
        "prompt": case_artifact.get("prompt", ""),
        "raw_output": case_artifact.get("raw_output", ""),
        "primary_diagnosis": parsed["primary_diagnosis"],
        "pred_top1": parsed["primary_diagnosis"],
        "pred_top3": _json_dumps(pred_labels),
        "top1_confidence": pred_confidences[0],
        "top1_display_score": pred_display_scores[0],
        "pred_display_scores": _json_dumps(pred_display_scores[:3]),
        "ranked_diagnoses": _json_dumps(parsed["ranked_diagnoses"]),
        "consultation_guidance": parsed["consultation_guidance"],
        "pred_1": pred_labels[0],
        "pred_2": pred_labels[1],
        "pred_3": pred_labels[2],
        "conf_1": pred_confidences[0],
        "conf_2": pred_confidences[1],
        "conf_3": pred_confidences[2],
        "display_1": pred_display_scores[0],
        "display_2": pred_display_scores[1],
        "display_3": pred_display_scores[2],
        "guidance": parsed["consultation_guidance"],
        "error": error,
        **eval_row,
    }


def run_baseline_experiment(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    label_space: list[str],
) -> pd.DataFrame:
    # Run the candidate-retrieval plus few-shot compare-and-rank baseline.
    results: list[dict[str, object]] = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Baseline few-shot"):
        few_shot_context = build_retrieved_few_shot_context(
            row,
            train_df,
            n=FEW_SHOT_EXAMPLES,
        )
        prepared = prepare_retrieve_then_reason_case(
            row=row,
            label_space=label_space,
            train_df=train_df,
            knowledge_snippets_df=None,
            few_shot_context=few_shot_context,
        )

        try:
            raw_output = run_openai(prepared["prompt"], model=OPENAI_MODEL, temperature=0)
            parsed = parse_ranked_output(raw_output, label_space, top_k=3)
            artifact = {
                **prepared,
                "raw_output": raw_output,
                "parsed": parsed,
            }
            results.append(
                finalize_result_row(
                    row=row,
                    method="baseline_few_shot",
                    case_artifact=artifact,
                    label_space=label_space,
                )
            )
        except Exception as exc:
            artifact = {
                **prepared,
                "raw_output": "",
                "parsed": _default_parsed_output(),
            }
            results.append(
                finalize_result_row(
                    row=row,
                    method="baseline_few_shot",
                    case_artifact=artifact,
                    label_space=label_space,
                    error=str(exc),
                )
            )

        time.sleep(0.1)

    return pd.DataFrame(results)


def run_knowledge_experiment(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    knowledge_df: pd.DataFrame,
    label_space: list[str],
) -> pd.DataFrame:
    # Run retrieve-then-reason with candidate labels and evidence snippets.
    results: list[dict[str, object]] = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Retrieve-then-reason"):
        prepared = prepare_retrieve_then_reason_case(
            row=row,
            label_space=label_space,
            train_df=train_df,
            knowledge_snippets_df=knowledge_df,
        )

        try:
            raw_output = run_openai(prepared["prompt"], model=OPENAI_MODEL, temperature=0)
            parsed = parse_ranked_output(raw_output, label_space, top_k=3)
            artifact = {
                **prepared,
                "raw_output": raw_output,
                "parsed": parsed,
            }
            results.append(
                finalize_result_row(
                    row=row,
                    method="retrieve_then_reason_evidence",
                    case_artifact=artifact,
                    label_space=label_space,
                )
            )
        except Exception as exc:
            artifact = {
                **prepared,
                "raw_output": "",
                "parsed": _default_parsed_output(),
            }
            results.append(
                finalize_result_row(
                    row=row,
                    method="retrieve_then_reason_evidence",
                    case_artifact=artifact,
                    label_space=label_space,
                    error=str(exc),
                )
            )

        time.sleep(0.1)

    return pd.DataFrame(results)


def run_self_consistency_experiment(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    knowledge_df: pd.DataFrame,
    label_space: list[str],
) -> pd.DataFrame:
    # Run self-consistency on the evidence-backed retrieve-then-reason prompt.
    results: list[dict[str, object]] = []

    for _, row in tqdm(
        eval_df.iterrows(),
        total=len(eval_df),
        desc="Self-consistency evidence",
    ):
        prepared = prepare_retrieve_then_reason_case(
            row=row,
            label_space=label_space,
            train_df=train_df,
            knowledge_snippets_df=knowledge_df,
        )

        try:
            aggregated = run_self_consistency_ranked(
                prompt=prepared["prompt"],
                label_space=label_space,
                parse_fn=parse_ranked_output,
                n=SELF_CONSISTENCY_RUNS,
                model=OPENAI_MODEL,
            )
            parsed = parse_ranked_output(_json_dumps(aggregated), label_space, top_k=3)
            artifact = {
                **prepared,
                "raw_output": _json_dumps(aggregated),
                "parsed": parsed,
            }
            results.append(
                finalize_result_row(
                    row=row,
                    method="self_consistency_evidence",
                    case_artifact=artifact,
                    label_space=label_space,
                )
            )
        except Exception as exc:
            artifact = {
                **prepared,
                "raw_output": "",
                "parsed": _default_parsed_output(),
            }
            results.append(
                finalize_result_row(
                    row=row,
                    method="self_consistency_evidence",
                    case_artifact=artifact,
                    label_space=label_space,
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
        ("retrieve_then_reason_evidence", knowledge_results),
        ("self_consistency_evidence", self_consistency_results),
    ]:
        row = summarize_results(result_df).iloc[0].to_dict()
        row["method"] = method_name
        row["error_count"] = int(
            result_df["error"].fillna("").astype(str).str.strip().ne("").sum()
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df[
        [
            "method",
            "n",
            "n_cases",
            "top1_accuracy",
            "top3_accuracy",
            "unknown_top1_rate",
            "error_count",
        ]
    ]


def _write_method_evaluation_outputs(stem: str, results_df: pd.DataFrame, label_space: list[str]) -> dict[str, pd.DataFrame]:
    # Persist per-method evaluation artifacts for deeper analysis.
    per_label_df = per_label_accuracy(results_df)
    confusion_df = confusion_matrix_table(results_df, labels=label_space)
    confusion_norm_df = normalize_confusion_rows(confusion_df)
    confidence_df = confidence_accuracy_table(results_df, confidence_col="top1_confidence")
    error_review_df = prepare_error_review_sheet(results_df)

    per_label_df.to_csv(OUTPUT_DIR / f"{stem}_per_label.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / f"{stem}_confusion_matrix.csv")
    confusion_norm_df.to_csv(OUTPUT_DIR / f"{stem}_confusion_matrix_row_norm.csv")
    confidence_df.to_csv(OUTPUT_DIR / f"{stem}_confidence_table.csv", index=False)
    error_review_df.to_csv(OUTPUT_DIR / f"{stem}_error_review.csv", index=False)

    return {
        "per_label": per_label_df,
        "confusion_matrix": confusion_df,
        "confusion_matrix_row_norm": confusion_norm_df,
        "confidence_table": confidence_df,
        "error_review": error_review_df,
    }


def write_outputs(
    baseline_results: pd.DataFrame,
    knowledge_results: pd.DataFrame,
    self_consistency_results: pd.DataFrame,
    method_summary: pd.DataFrame,
    label_space: list[str],
) -> dict[str, dict[str, pd.DataFrame]]:
    # Persist experiment outputs and deeper evaluation tables.
    baseline_results.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    knowledge_results.to_csv(OUTPUT_DIR / "knowledge_results.csv", index=False)
    self_consistency_results.to_csv(OUTPUT_DIR / "self_consistency_results.csv", index=False)
    method_summary.to_csv(OUTPUT_DIR / "method_summary.csv", index=False)

    return {
        "baseline_results": _write_method_evaluation_outputs(
            "baseline_results",
            baseline_results,
            label_space,
        ),
        "knowledge_results": _write_method_evaluation_outputs(
            "knowledge_results",
            knowledge_results,
            label_space,
        ),
        "self_consistency_results": _write_method_evaluation_outputs(
            "self_consistency_results",
            self_consistency_results,
            label_space,
        ),
    }


def run_all_experiments() -> dict[str, object]:
    # Run the upgraded retrieve-then-reason experiment suite and save outputs.
    df, train_df, test_df, label_space = load_split_dataset()
    eval_df = build_eval_subset(test_df, n_cases=EXPERIMENT_CASES)
    knowledge_df = load_knowledge_snippets(EBM_DIR / "knowledge_snippets.csv")

    baseline_results = run_baseline_experiment(train_df, eval_df, label_space)
    knowledge_results = run_knowledge_experiment(
        train_df,
        eval_df,
        knowledge_df,
        label_space,
    )
    self_consistency_results = run_self_consistency_experiment(
        train_df,
        eval_df,
        knowledge_df,
        label_space,
    )
    method_summary = build_method_summary(
        baseline_results,
        knowledge_results,
        self_consistency_results,
    )
    evaluation_outputs = write_outputs(
        baseline_results,
        knowledge_results,
        self_consistency_results,
        method_summary,
        label_space,
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
        "evaluation_outputs": evaluation_outputs,
        "majority_baseline": majority_baseline_metrics(train_df, eval_df),
    }


if __name__ == "__main__":
    artifacts = run_all_experiments()
    print("Saved outputs to:", OUTPUT_DIR)
    print(artifacts["method_summary"].to_string(index=False))
