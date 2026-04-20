# Few-shot retrieval helpers based on symptom overlap.

from __future__ import annotations

from collections import Counter
import re

import pandas as pd


def normalize_symptom_text(text: object) -> str:
    # Lowercase and normalize punctuation for symptom text matching.
    text = str(text).lower().strip()
    text = re.sub(r"\s*\((finding|disorder)\)\s*", " ", text)
    text = re.sub(r"[^a-z0-9,\s\-()]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_simple(text: object) -> list[str]:
    # Tokenize free text into lightweight lexical units for retrieval.
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def split_symptom_phrases(text: object) -> list[str]:
    # Break free text into short symptom-style phrases for overlap scoring.
    text_norm = normalize_symptom_text(text)
    if not text_norm:
        return []

    phrases = [
        part.strip(" -")
        for part in re.split(r"[,;\n]+|\band\b", text_norm)
        if part.strip(" -")
    ]
    return [phrase for phrase in phrases if phrase]


def symptom_token_set(text: object) -> set[str]:
    # Convert symptom text into a combined set of phrases and tokens.
    phrases = split_symptom_phrases(text)
    return set(phrases) | set(tokenize_simple(text))


def _record_value(record, key: str, default: object = "") -> object:
    # Read dict-like or Series-like values without caring about the concrete type.
    if hasattr(record, "get"):
        return record.get(key, default)
    try:
        return record[key]
    except Exception:
        return default


def _normalize_binary_answer(value: object) -> str:
    # Normalize tri-state intake flags for retrieval text construction.
    text = normalize_symptom_text(value)
    if text in {"yes", "y", "true"}:
        return "yes"
    if text in {"no", "n", "false"}:
        return "no"
    return ""


def _age_bucket(age: object) -> str:
    # Bucket ages so demographics can influence retrieval softly.
    try:
        age_value = int(float(age))
    except (TypeError, ValueError):
        return ""

    if age_value < 18:
        return "child"
    if age_value < 40:
        return "young adult"
    if age_value < 65:
        return "adult"
    return "older adult"


def build_retrieval_text(record) -> str:
    # Build a retrieval string from structured intake or dataset rows.
    parts = [
        _record_value(record, "symptoms", ""),
        _record_value(record, "symptoms_started", _record_value(record, "when_symptoms_started", "")),
        _record_value(record, "severity", ""),
        _record_value(record, "existing_conditions", ""),
        _record_value(record, "recent_worsening", ""),
        _record_value(record, "sex", ""),
        _age_bucket(_record_value(record, "age", "")),
    ]

    for key, label in [
        ("fever", "fever"),
        ("breathing_difficulty", "breathing difficulty"),
        ("chest_pain", "chest pain"),
        ("bleeding", "bleeding"),
        ("confusion", "confusion"),
    ]:
        if _normalize_binary_answer(_record_value(record, key, "")) == "yes":
            parts.append(label)

    return normalize_symptom_text(", ".join(str(part) for part in parts if str(part).strip()))


def _token_jaccard(left_text: str, right_text: str) -> float:
    # Score pairwise case similarity with lightweight lexical overlap.
    left_tokens = symptom_token_set(left_text)
    right_tokens = symptom_token_set(right_text)
    if not left_tokens and not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def retrieve_similar_examples(
    row,
    train_df: pd.DataFrame,
    n: int = 4,
) -> pd.DataFrame:
    # Retrieve training examples with the highest structured overlap similarity.
    if n <= 0 or train_df.empty:
        return train_df.head(0).copy()

    query_text = build_retrieval_text(row)
    scored_rows: list[tuple[int, float]] = []

    for idx, train_row in train_df.iterrows():
        train_text = build_retrieval_text(train_row)
        overlap = _token_jaccard(query_text, train_text)
        scored_rows.append((idx, overlap))

    scored_rows = sorted(scored_rows, key=lambda item: item[1], reverse=True)
    top_indices = [idx for idx, _ in scored_rows[:n]]
    return train_df.loc[top_indices].reset_index(drop=True)


def build_retrieved_few_shot_context(
    row,
    train_df: pd.DataFrame,
    n: int = 4,
) -> str:
    # Format retrieved training rows into a compact few-shot prompt block.
    sample = retrieve_similar_examples(row, train_df, n=n)
    chunks: list[str] = []

    for i, (_, retrieved_row) in enumerate(sample.iterrows(), start=1):
        chunks.append(
            f"Retrieved reference case {i}:\n"
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
    case_query: str = "",
) -> list[str]:
    # Retrieve candidate labels from the closed set using lexical overlap.
    query_text = case_query or symptoms
    symptom_counts = Counter(tokenize_simple(query_text))
    scored: list[tuple[str, int]] = []

    for label in label_space:
        label_tokens = tokenize_simple(label)
        overlap = sum(symptom_counts[token] for token in label_tokens)
        scored.append((label, overlap))

    scored.sort(key=lambda item: (-item[1], item[0]))
    candidates = [label for label, _ in scored[:k] if label]
    return candidates or label_space[:k]


def retrieve_candidate_labels_from_examples(
    row,
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
        symptoms=str(_record_value(row, "symptoms", "")),
        label_space=label_space,
        k=max(k, len(ordered_candidates)),
        case_query=build_retrieval_text(row),
    )

    merged: list[str] = []
    for label in ordered_candidates + lexical_candidates:
        if label and label not in merged:
            merged.append(label)
        if len(merged) >= k:
            break

    return merged or label_space[:k]


def _format_sex_for_display(value: object) -> str:
    # Render compact sex text for similar-case output.
    text = normalize_symptom_text(value)
    if text in {"f", "female", "woman"}:
        return "female"
    if text in {"m", "male", "man"}:
        return "male"
    return "patient"


def _short_symptom_summary(text: object, max_items: int = 3) -> str:
    # Trim longer symptom strings into presentation-friendly summaries.
    phrases = split_symptom_phrases(text)
    if not phrases:
        return "reported symptoms"
    return ", ".join(phrases[:max_items])


def format_similar_case_summary(case_row, prefix: str = "") -> str:
    # Format one retrieved case for user-facing display.
    try:
        age_text = f"{int(float(_record_value(case_row, 'age', '')))}-year-old"
    except (TypeError, ValueError):
        age_text = "adult"

    sex_text = _format_sex_for_display(_record_value(case_row, "sex", ""))
    diagnosis = str(
        _record_value(case_row, "diagnosis", _record_value(case_row, "label", "unknown"))
    ).strip() or "unknown"
    summary = _short_symptom_summary(_record_value(case_row, "symptoms", ""))
    prefix_text = f"{prefix}: " if prefix else ""
    return f"{prefix_text}{age_text} {sex_text} with {summary} -> {diagnosis}"


def retrieve_similar_case_summaries(
    row,
    train_df: pd.DataFrame,
    n: int = 2,
    preferred_labels: list[str] | None = None,
) -> list[str]:
    # Retrieve short similar-case strings for final presentation.
    sample = retrieve_similar_examples(row, train_df, n=max(n * 4, n))
    diagnosis_col = "diagnosis" if "diagnosis" in sample.columns else "label"

    selected_rows: list[pd.Series] = []
    if preferred_labels:
        preferred_set = {str(label).strip() for label in preferred_labels if str(label).strip()}
        preferred_sample = sample[
            sample[diagnosis_col].astype(str).str.strip().isin(preferred_set)
        ]
        for _, preferred_row in preferred_sample.iterrows():
            selected_rows.append(preferred_row)
            if len(selected_rows) >= n:
                break

    if len(selected_rows) < n:
        for _, fallback_row in sample.iterrows():
            if len(selected_rows) >= n:
                break
            row_signature = (
                _record_value(fallback_row, "age", ""),
                _record_value(fallback_row, "sex", ""),
                _record_value(fallback_row, "symptoms", ""),
                _record_value(fallback_row, diagnosis_col, ""),
            )
            if any(
                (
                    _record_value(existing_row, "age", ""),
                    _record_value(existing_row, "sex", ""),
                    _record_value(existing_row, "symptoms", ""),
                    _record_value(existing_row, diagnosis_col, ""),
                )
                == row_signature
                for existing_row in selected_rows
            ):
                continue
            selected_rows.append(fallback_row)

    labels = ["Case A", "Case B", "Case C", "Case D"]
    summaries: list[str] = []

    for idx, retrieved_row in enumerate(selected_rows[:n]):
        summaries.append(format_similar_case_summary(retrieved_row, prefix=labels[idx]))

    return summaries


def score_snippet_overlap(case_query: str, snippet: str) -> int:
    # Score evidence overlap using lightweight lexical intersection.
    case_tokens = set(tokenize_simple(case_query))
    snippet_tokens = set(tokenize_simple(snippet))
    return len(case_tokens & snippet_tokens)


def _compact_snippet(snippet: str, max_chars: int = 180) -> str:
    # Convert noisy external snippets into short prompt-friendly notes.
    text = " ".join(str(snippet).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_candidate_evidence_blocks(
    row,
    candidate_labels: list[str],
    train_df: pd.DataFrame,
    similar_examples: pd.DataFrame | None = None,
    max_cases_per_candidate: int = 2,
) -> list[str]:
    # Summarize nearby training examples into compact candidate-comparison evidence.
    if train_df.empty or not candidate_labels:
        return []

    if similar_examples is None:
        similar_examples = retrieve_similar_examples(
            row,
            train_df,
            n=max(8, len(candidate_labels) * max_cases_per_candidate),
        )

    query_phrases = split_symptom_phrases(_record_value(row, "symptoms", ""))
    query_tokens = set(tokenize_simple(build_retrieval_text(row)))
    diagnosis_col = "diagnosis" if "diagnosis" in similar_examples.columns else "label"
    neighborhood_labels = [
        str(example.get(diagnosis_col, "")).strip()
        for _, example in similar_examples.iterrows()
        if str(example.get(diagnosis_col, "")).strip()
    ]

    evidence_blocks: list[str] = []

    for label in candidate_labels:
        label_examples = similar_examples[
            similar_examples[diagnosis_col].astype(str).str.strip() == label
        ].head(max_cases_per_candidate)
        if label_examples.empty:
            continue

        overlap_counts: Counter[str] = Counter()
        phrase_counts: Counter[str] = Counter()

        for _, example_row in label_examples.iterrows():
            for phrase in split_symptom_phrases(example_row.get("symptoms", "")):
                phrase_counts[phrase] += 1
                phrase_tokens = set(tokenize_simple(phrase))
                if phrase in query_phrases or phrase_tokens & query_tokens:
                    overlap_counts[phrase] += 1

        overlap_terms = [term for term, _ in overlap_counts.most_common(3)]
        common_terms = [term for term, _ in phrase_counts.most_common(3)]
        symptom_patterns = overlap_terms or common_terms or ["limited direct overlap"]
        distinguishing_features = _short_symptom_summary(label_examples.iloc[0]["symptoms"])

        confusions: list[str] = []
        for other_label in neighborhood_labels:
            if other_label and other_label != label and other_label not in confusions:
                confusions.append(other_label)
            if len(confusions) >= 2:
                break

        confusion_text = ", ".join(confusions) if confusions else "limited nearby contrast"
        evidence_blocks.append(
            f"{label} | symptom patterns: {', '.join(symptom_patterns)} | "
            f"distinguishing features from similar cases: {distinguishing_features} | "
            f"common confusions: {confusion_text}"
        )

    return evidence_blocks


def retrieve_evidence_snippets(
    case_query: str,
    knowledge_snippets: pd.DataFrame | None,
    k: int = 3,
    label_space: list[str] | None = None,
    candidate_labels: list[str] | None = None,
    row=None,
    train_df: pd.DataFrame | None = None,
    similar_examples: pd.DataFrame | None = None,
) -> list[str]:
    # Retrieve compact evidence blocks for reasoning, not raw document dumps.
    evidence_blocks: list[str] = []

    if row is not None and train_df is not None and candidate_labels:
        evidence_blocks.extend(
            _build_candidate_evidence_blocks(
                row=row,
                candidate_labels=candidate_labels,
                train_df=train_df,
                similar_examples=similar_examples,
            )
        )

    if knowledge_snippets is not None and not knowledge_snippets.empty and k > 0:
        text_col = "text" if "text" in knowledge_snippets.columns else "snippet"
        rows: list[tuple[str, int]] = []
        normalized_labels = [
            normalize_symptom_text(label) for label in (label_space or []) if str(label).strip()
        ]

        for _, knowledge_row in knowledge_snippets.iterrows():
            text = " ".join(str(knowledge_row.get(text_col, "")).split())
            if not text:
                continue

            if normalized_labels:
                text_norm = normalize_symptom_text(text)
                if any(label in text_norm for label in normalized_labels):
                    continue

            score = score_snippet_overlap(case_query, text)
            if score <= 0:
                continue
            rows.append((text, score))

        rows.sort(key=lambda item: item[1], reverse=True)
        for text, _ in rows[:k]:
            evidence_blocks.append(f"External note: {_compact_snippet(text)}")

    return list(dict.fromkeys(evidence_blocks))
