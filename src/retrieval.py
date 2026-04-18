# Few-shot retrieval helpers based on symptom overlap.

from __future__ import annotations

import re

import pandas as pd


def normalize_symptom_text(text: object) -> str:
    # Lowercase and normalize punctuation for symptom text matching.
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9,\s\-()]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def symptom_token_set(text: object) -> set[str]:
    # Convert a comma-separated symptom string into a normalized token set.
    parts = [part.strip() for part in normalize_symptom_text(text).split(",") if part.strip()]
    return set(parts)


def retrieve_similar_examples(
    row: pd.Series,
    train_df: pd.DataFrame,
    n: int = 4,
) -> pd.DataFrame:
    # Retrieve training examples with the highest token-overlap similarity.
    row_tokens = symptom_token_set(row["symptoms"])
    scored_rows: list[tuple[int, float]] = []

    for idx, train_row in train_df.iterrows():
        train_tokens = symptom_token_set(train_row["symptoms"])
        overlap = len(row_tokens & train_tokens) / max(1, len(row_tokens | train_tokens))
        scored_rows.append((idx, overlap))

    scored_rows = sorted(scored_rows, key=lambda item: item[1], reverse=True)
    top_indices = [idx for idx, _ in scored_rows[:n]]
    return train_df.loc[top_indices].reset_index(drop=True)


def build_retrieved_few_shot_context(
    row: pd.Series,
    train_df: pd.DataFrame,
    n: int = 4,
) -> str:
    # Format retrieved training rows into a few-shot prompt block.
    sample = retrieve_similar_examples(row, train_df, n=n)
    chunks: list[str] = []

    for i, (_, retrieved_row) in enumerate(sample.iterrows(), start=1):
        chunks.append(
            f"Example {i}:\n"
            f"Age: {int(retrieved_row['age'])}\n"
            f"Sex: {retrieved_row['sex']}\n"
            f"Reported findings: {retrieved_row['symptoms']}\n"
            f"Diagnosis: {retrieved_row['diagnosis']}"
        )

    return "\n\n".join(chunks)
