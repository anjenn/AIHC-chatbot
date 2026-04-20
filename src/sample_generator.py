# Internal utility for building paste-ready chatbot experiment samples.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from src.config import OUTPUT_DIR, SEED
from src.prompts import format_structured_intake_message
from src.run_experiments import load_split_dataset

SEVERE_KEYWORDS = {
    "hypoxemia",
    "respiratory distress",
    "shortness of breath",
    "dyspnea",
    "chest pain",
    "confusion",
    "bleeding",
    "hemorrhage",
    "vomiting",
    "cardiac arrest",
}
MODERATE_KEYWORDS = {
    "cough",
    "fever",
    "fatigue",
    "nausea",
    "headache",
    "sore throat",
    "loss of taste",
    "weakness",
}
FEVER_KEYWORDS = {"fever", "febrile"}
BREATHING_KEYWORDS = {"shortness of breath", "dyspnea", "respiratory distress", "hypoxemia"}
CHEST_PAIN_KEYWORDS = {"chest pain", "chest pressure", "tightness in chest"}
BLEEDING_KEYWORDS = {"bleeding", "hemorrhage", "blood loss"}
CONFUSION_KEYWORDS = {"confusion", "confused", "altered mental status", "disoriented"}
CONDITION_KEYWORDS = {
    "history of",
    "history",
    "chronic",
    "disease",
    "syndrome",
    "disorder",
    "diabetes",
    "hypertension",
    "hyperlipidemia",
    "prediabetes",
    "kidney",
    "coronary",
    "obesity",
    "heart failure",
    "cancer",
    "carcinoma",
    "myocardial infarction",
}


def _clean_phrase(text: object) -> str:
    # Clean dataset phrases into chatbot-friendly surface text.
    cleaned = str(text).strip()
    cleaned = re.sub(r"\s*\((finding|disorder|situation)\)\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("  ", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,")


def _split_phrases(symptoms: object) -> list[str]:
    # Split the raw dataset symptom text into compact phrases.
    parts = [_clean_phrase(part) for part in str(symptoms).split(",")]
    return [part for part in parts if part]


def _normalize_sex(sex: object) -> str:
    # Normalize sex labels for experiment display.
    sex_text = str(sex).strip().lower()
    if sex_text in {"f", "female", "woman"}:
        return "female"
    if sex_text in {"m", "male", "man"}:
        return "male"
    return "not provided"


def _has_any_keyword(text: str, keywords: set[str]) -> bool:
    # Check for keyword matches in a cleaned phrase bundle.
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _infer_binary_flag(phrases: list[str], keywords: set[str]) -> str:
    # Convert symptom mentions into yes/no flags for paste-ready intake text.
    bundle = " | ".join(phrases)
    return "yes" if _has_any_keyword(bundle, keywords) else "no"


def _infer_severity(phrases: list[str]) -> str:
    # Infer a light-touch severity field from symptom wording when obvious.
    bundle = " | ".join(phrases)
    if _has_any_keyword(bundle, SEVERE_KEYWORDS):
        return "severe"
    if _has_any_keyword(bundle, MODERATE_KEYWORDS) or len(phrases) >= 4:
        return "moderate"
    return "not provided"


def _split_conditions_and_symptoms(phrases: list[str]) -> tuple[list[str], list[str]]:
    # Separate likely chronic conditions from likely current symptoms.
    condition_terms: list[str] = []
    symptom_terms: list[str] = []

    for phrase in phrases:
        if _has_any_keyword(phrase, CONDITION_KEYWORDS):
            condition_terms.append(phrase)
        else:
            symptom_terms.append(phrase)

    if not symptom_terms:
        symptom_terms = phrases[:]

    return symptom_terms, condition_terms


def row_to_chatbot_sample(
    row: pd.Series,
    sample_id: str,
    source_split: str,
) -> dict[str, object]:
    # Convert one dataset row into a paste-ready chatbot experiment sample.
    phrases = _split_phrases(row.get("symptoms", ""))
    symptom_terms, condition_terms = _split_conditions_and_symptoms(phrases)

    main_symptoms = ", ".join(symptom_terms[:6]) if symptom_terms else "not provided"
    existing_conditions = (
        ", ".join(condition_terms[:4]) if condition_terms else "not provided in dataset"
    )
    intake = {
        "age": int(float(row.get("age"))),
        "sex": _normalize_sex(row.get("sex", "")),
        "symptoms": main_symptoms,
        "symptoms_started": "not provided in dataset",
        "severity": _infer_severity(phrases),
        "fever": _infer_binary_flag(phrases, FEVER_KEYWORDS),
        "breathing_difficulty": _infer_binary_flag(phrases, BREATHING_KEYWORDS),
        "chest_pain": _infer_binary_flag(phrases, CHEST_PAIN_KEYWORDS),
        "bleeding": _infer_binary_flag(phrases, BLEEDING_KEYWORDS),
        "confusion": _infer_binary_flag(phrases, CONFUSION_KEYWORDS),
        "existing_conditions": existing_conditions,
        "recent_worsening": "not provided in dataset",
    }
    chatbot_message = format_structured_intake_message(**intake)

    return {
        "sample_id": sample_id,
        "source_split": source_split,
        "patient_index": int(row.name),
        "age": intake["age"],
        "sex": intake["sex"],
        "actual_diagnosis_label": str(row.get("diagnosis", "")).strip(),
        "source_symptoms": str(row.get("symptoms", "")).strip(),
        "main_symptoms_used": intake["symptoms"],
        "existing_conditions_used": intake["existing_conditions"],
        "chatbot_message": chatbot_message,
    }


def _sample_by_label(
    df: pd.DataFrame,
    per_label: int,
    seed: int,
) -> pd.DataFrame:
    # Pull a balanced sample across diagnosis labels.
    groups: list[pd.DataFrame] = []

    for diagnosis, group in df.groupby("diagnosis", sort=True):
        take_n = min(per_label, len(group))
        groups.append(group.sample(n=take_n, random_state=seed))

    sampled = pd.concat(groups, ignore_index=False)
    return sampled.sort_values(["diagnosis", "age", "sex"]).reset_index(drop=False).rename(
        columns={"index": "source_index"}
    )


def _sample_random(
    df: pd.DataFrame,
    count: int,
    seed: int,
) -> pd.DataFrame:
    # Pull a bounded random sample from the chosen split.
    count = min(count, len(df))
    sampled = df.sample(n=count, random_state=seed)
    return sampled.reset_index(drop=False).rename(columns={"index": "source_index"})


def _select_split(split_name: str) -> tuple[pd.DataFrame, list[str]]:
    # Load the requested split from the shared experiment dataset.
    full_df, train_df, test_df, label_space = load_split_dataset()
    split_map = {
        "full": full_df,
        "train": train_df,
        "test": test_df,
    }
    return split_map[split_name].copy().reset_index(drop=True), label_space


def build_chatbot_experiment_samples(
    split_name: str = "test",
    count: int | None = None,
    per_label: int | None = 2,
    seed: int = SEED,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    # Build experiment samples with paste-ready chatbot messages and gold labels.
    split_df, _ = _select_split(split_name)

    if labels:
        label_set = {label.strip().lower() for label in labels if label.strip()}
        split_df = split_df[
            split_df["diagnosis"].astype(str).str.strip().str.lower().isin(label_set)
        ].copy()

    if split_df.empty:
        raise ValueError("No dataset rows matched the requested split and label filters.")

    if per_label is not None:
        sampled_df = _sample_by_label(split_df, per_label=per_label, seed=seed)
    else:
        if count is None:
            count = min(12, len(split_df))
        sampled_df = _sample_random(split_df, count=count, seed=seed)

    records: list[dict[str, object]] = []
    for idx, row in enumerate(sampled_df.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        row_series.name = int(row_series.get("source_index", idx - 1))
        records.append(
            row_to_chatbot_sample(
                row=row_series,
                sample_id=f"{split_name.upper()}-{idx:03d}",
                source_split=split_name,
            )
        )

    return pd.DataFrame(records)


def _write_markdown(samples_df: pd.DataFrame, path: Path) -> None:
    # Save a human-friendly markdown file for copy/paste experiments.
    lines = [
        "# Chatbot Experiment Samples",
        "",
        "These samples are extracted from the filtered dataset.",
        "Each sample includes the actual diagnosis label as an answer key, but the paste-ready chatbot text does not include that label.",
        "",
    ]

    for _, row in samples_df.iterrows():
        lines.append(f"## {row['sample_id']}")
        lines.append("")
        lines.append(f"- Source split: `{row['source_split']}`")
        lines.append(f"- Patient index: `{row['patient_index']}`")
        lines.append(f"- Actual diagnosis label: `{row['actual_diagnosis_label']}`")
        lines.append("")
        lines.append("Paste into chatbot:")
        lines.append("")
        lines.append("```text")
        lines.append(str(row["chatbot_message"]))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_chatbot_experiment_samples(
    samples_df: pd.DataFrame,
    output_prefix: str,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    # Persist the sample set in CSV, JSONL, and Markdown formats.
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    jsonl_path = output_dir / f"{output_prefix}.jsonl"
    md_path = output_dir / f"{output_prefix}.md"

    samples_df.to_csv(csv_path, index=False)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in samples_df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    _write_markdown(samples_df, md_path)
    return {"csv": csv_path, "jsonl": jsonl_path, "md": md_path}


def parse_args() -> argparse.Namespace:
    # Parse CLI arguments for the internal sample generator.
    parser = argparse.ArgumentParser(
        description="Generate paste-ready chatbot experiment samples with gold labels.",
    )
    parser.add_argument(
        "--split",
        choices=["full", "train", "test"],
        default="test",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of random samples to draw when --per-label is not used.",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=2,
        help="Samples per diagnosis label. Set to 0 to disable balanced per-label sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional diagnosis label filter. Repeat to include multiple labels.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix inside the outputs directory.",
    )
    return parser.parse_args()


def main() -> None:
    # CLI entry point for generating experiment samples.
    args = parse_args()
    per_label = args.per_label if args.per_label and args.per_label > 0 else None
    samples_df = build_chatbot_experiment_samples(
        split_name=args.split,
        count=args.count,
        per_label=per_label,
        seed=args.seed,
        labels=args.label,
    )
    output_prefix = args.output_prefix or f"chatbot_experiment_samples_{args.split}"
    output_paths = write_chatbot_experiment_samples(samples_df, output_prefix=output_prefix)

    print(f"Wrote {len(samples_df)} samples.")
    for file_type, path in output_paths.items():
        print(f"{file_type}: {path}")


if __name__ == "__main__":
    main()
