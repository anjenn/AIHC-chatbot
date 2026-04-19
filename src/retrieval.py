# Few-shot retrieval helpers based on symptom overlap.

from __future__ import annotations

from collections import Counter
import re

import pandas as pd


def normalize_symptom_text(text: object) -> str:
    # Lowercase and normalize punctuation for symptom text matching.
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9,\s\-()]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_simple(text: object) -> list[str]:
    # Tokenize free text into lightweight lexical units for retrieval.
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def symptom_token_set(text: object) -> set[str]:
    # Convert a comma-separated symptom string into a normalized token set.
    parts = [
        part.strip()
        for part in normalize_symptom_text(text).split(",")
        if part.strip()
    ]
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


def retrieve_candidate_labels(
    symptoms: str,
    label_space: list[str],
    k: int = 5,
) -> list[str]:
    # Retrieve candidate labels from the closed set using lexical overlap.
    symptom_counts = Counter(tokenize_simple(symptoms))
    scored: list[tuple[str, int]] = []

    for label in label_space:
        label_tokens = tokenize_simple(label)
        overlap = sum(symptom_counts[token] for token in label_tokens)
        scored.append((label, overlap))

    scored.sort(key=lambda item: (-item[1], item[0]))
    candidates = [label for label, _ in scored[:k] if label]
    return candidates or label_space[:k]


def retrieve_candidate_labels_from_examples(
    row: pd.Series,
    train_df: pd.DataFrame,
    label_space: list[str],
    k_neighbors: int = 8,
    k: int = 5,
) -> list[str]:
    # Retrieve candidate labels from similar training examples with lexical fallback.
    sample = retrieve_similar_examples(row, train_df, n=k_neighbors)
    diagnosis_col = "diagnosis" if "diagnosis" in sample.columns else "label"
    candidate_scores: Counter[str] = Counter()

    for rank, (_, retrieved_row) in enumerate(sample.iterrows(), start=1):
        label = str(retrieved_row.get(diagnosis_col, "")).strip()
        if label in label_space:
            candidate_scores[label] += max(k_neighbors - rank + 1, 1)

    ordered_candidates = [label for label, _ in candidate_scores.most_common()]
    lexical_candidates = retrieve_candidate_labels(
        symptoms=str(row.get("symptoms", "")),
        label_space=label_space,
        k=max(k, len(ordered_candidates)),
    )

    merged: list[str] = []
    for label in ordered_candidates + lexical_candidates:
        if label and label not in merged:
            merged.append(label)
        if len(merged) >= k:
            break

    return merged or label_space[:k]


def score_snippet_overlap(case_query: str, snippet: str) -> int:
    # Score evidence overlap using lightweight lexical intersection.
    case_tokens = set(tokenize_simple(case_query))
    snippet_tokens = set(tokenize_simple(snippet))
    return len(case_tokens & snippet_tokens)


def retrieve_evidence_snippets(
    case_query: str,
    knowledge_snippets: pd.DataFrame,
    k: int = 3,
    label_space: list[str] | None = None,
) -> list[str]:
    # Retrieve the most relevant evidence snippets for the case query.
    text_col = "text" if "text" in knowledge_snippets.columns else "snippet"
    rows: list[tuple[str, int]] = []
    normalized_labels = [
        normalize_symptom_text(label) for label in (label_space or []) if str(label).strip()
    ]

    for _, row in knowledge_snippets.iterrows():
        text = " ".join(str(row.get(text_col, "")).split())
        if not text:
            continue

        if normalized_labels:
            text_norm = normalize_symptom_text(text)
            if any(label in text_norm for label in normalized_labels):
                continue

        score = score_snippet_overlap(case_query, text)
        rows.append((text, score))

    rows.sort(key=lambda item: item[1], reverse=True)
    top = [text for text, score in rows if score > 0][:k]
    return list(dict.fromkeys(top))
