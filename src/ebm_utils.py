# Prompt-level knowledge utilities for EBM-NLP enrichment.

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from src.retrieval import normalize_symptom_text

COMMON_KNOWLEDGE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "during",
    "were",
    "was",
    "have",
    "has",
    "had",
    "are",
    "been",
    "being",
    "about",
    "after",
    "before",
    "over",
    "under",
    "through",
    "than",
    "their",
    "them",
    "they",
    "adult",
    "adults",
    "patient",
    "patients",
    "study",
    "trial",
    "group",
    "background",
    "objective",
    "objectives",
    "results",
    "methods",
    "finding",
    "symptom",
    "disorder",
}


def symptom_keywords(symptoms: str) -> set[str]:
    # Extract phrase and token keywords from symptom text.
    phrases = [
        phrase.strip()
        for phrase in normalize_symptom_text(symptoms).split(",")
        if phrase.strip()
    ]
    keywords: set[str] = set()

    for phrase in phrases:
        if len(phrase) >= 4:
            keywords.add(phrase)

        for token in phrase.split():
            if len(token) >= 4 and token not in COMMON_KNOWLEDGE_STOPWORDS:
                keywords.add(token)

    return keywords


def snippet_has_label_match(snippet: str, label_space: list[str] | None = None) -> bool:
    # Return True when a snippet directly mentions one of the closed labels.
    if not label_space:
        return False

    snippet_norm = normalize_symptom_text(snippet)
    return any(normalize_symptom_text(label) in snippet_norm for label in label_space)


def load_knowledge_snippets(path: str | Path) -> pd.DataFrame:
    # Load a CSV containing short prompt-level knowledge snippets.
    knowledge_df = pd.read_csv(path)

    if "snippet" not in knowledge_df.columns:
        raise ValueError("Knowledge snippet file must contain a 'snippet' column.")

    return knowledge_df


def retrieve_knowledge(
    symptoms: str,
    knowledge_df: pd.DataFrame,
    top_k: int = 2,
    label_space: list[str] | None = None,
) -> list[str]:
    # Retrieve relevant snippets using symptom keywords and label-match filtering.
    symptom_terms = symptom_keywords(symptoms)
    scores: list[tuple[str, int]] = []

    for _, row in knowledge_df.iterrows():
        snippet = str(row["snippet"])
        if snippet_has_label_match(snippet, label_space):
            continue

        snippet_norm = normalize_symptom_text(snippet)
        snippet_terms = symptom_keywords(snippet)
        phrase_overlap = sum(
            1 for term in symptom_terms if " " in term and term in snippet_norm
        )
        token_overlap = len(
            {term for term in symptom_terms if " " not in term} & snippet_terms
        )
        if phrase_overlap == 0 and token_overlap < 2:
            continue

        score = phrase_overlap * 3 + token_overlap
        scores.append((snippet, score))

    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    return [snippet for snippet, score in scores[:top_k] if score > 0]


def load_ebm_documents(doc_dir: str | Path, max_docs: int = 100) -> pd.DataFrame:
    # Load raw EBM text documents for lightweight snippet generation.
    doc_dir = Path(doc_dir)
    docs: list[dict[str, str]] = []

    for fname in sorted(os.listdir(doc_dir)):
        if fname.endswith(".txt"):
            path = doc_dir / fname
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read().strip()

            if text:
                docs.append({"doc_id": fname, "text": text[:3000]})

        if len(docs) >= max_docs:
            break

    return pd.DataFrame(docs)


def make_simple_snippet(text: str, max_len: int = 220) -> str:
    # Convert longer EBM text into a short prompt-ready snippet.
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:max_len]
