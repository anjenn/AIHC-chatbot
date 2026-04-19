# Output normalization and JSON parsing utilities.

from __future__ import annotations

import ast
import json
import re


def normalize_text(text: object) -> str:
    # Lowercase and whitespace-normalize text.
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label_to_allowed(text: object, label_space: list[str]) -> str:
    # Map model output text back onto the closed label set.
    text_norm = normalize_text(text)

    for label in label_space:
        label_norm = normalize_text(label)
        if text_norm == label_norm:
            return label_norm

    for label in label_space:
        label_norm = normalize_text(label)
        if label_norm in text_norm or text_norm in label_norm:
            return label_norm

    return "unknown"


def safe_json_load(text: str):
    # Safely parse JSON after removing optional markdown fences.
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def _coerce_string_list(value: object) -> list[str]:
    # Convert a value into a compact list of strings.
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _coerce_confidence(value: object) -> float:
    # Parse numeric confidence values while keeping them inside [0, 1].
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _default_ranked_output(top_k: int) -> dict[str, object]:
    # Build a stable fallback output for parser failures.
    ranked_diagnoses = [
        {
            "label": "unknown",
            "confidence": 0.0,
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
        "guidance": "",
        "consultation_guidance": "",
    }


def parse_ranked_output(
    raw_output: str,
    label_space: list[str],
    top_k: int = 3,
) -> dict[str, object]:
    # Extract ranked diagnoses, confidence proxies, and guidance from model output.
    parsed = safe_json_load(raw_output)
    default_output = _default_ranked_output(top_k)

    if not isinstance(parsed, dict):
        return default_output

    items = parsed.get("ranked_diagnoses", [])
    if not isinstance(items, list):
        items = []

    if not items:
        legacy_items = parsed.get("top_3", [])
        if isinstance(legacy_items, list):
            items = legacy_items

    if not items and parsed.get("primary_diagnosis"):
        items = [{"label": parsed.get("primary_diagnosis", ""), "confidence": 0.0}]

    guidance = str(
        parsed.get("consultation_guidance", parsed.get("guidance", ""))
    ).strip()

    normalized_items: list[dict[str, object]] = []
    top_labels: list[str] = []
    top_confidences: list[float] = []

    for item in items[:top_k]:
        if not isinstance(item, dict):
            continue

        label = normalize_label_to_allowed(item.get("label", ""), label_space)
        confidence = _coerce_confidence(item.get("confidence", 0.0))
        supporting_evidence = _coerce_string_list(item.get("supporting_evidence", []))
        missing_or_uncertain = _coerce_string_list(item.get("missing_or_uncertain", []))

        normalized_items.append(
            {
                "label": label,
                "confidence": confidence,
                "supporting_evidence": supporting_evidence,
                "missing_or_uncertain": missing_or_uncertain,
            }
        )
        top_labels.append(label)
        top_confidences.append(confidence)

    while len(normalized_items) < top_k:
        normalized_items.append(
            {
                "label": "unknown",
                "confidence": 0.0,
                "supporting_evidence": [],
                "missing_or_uncertain": [],
            }
        )
        top_labels.append("unknown")
        top_confidences.append(0.0)

    primary = normalize_label_to_allowed(parsed.get("primary_diagnosis", ""), label_space)
    if primary == "unknown" and top_labels:
        primary = top_labels[0]

    return {
        "primary_diagnosis": primary,
        "ranked_diagnoses": normalized_items,
        "top_labels": top_labels[:top_k],
        "top_confidences": top_confidences[:top_k],
        "guidance": guidance,
        "consultation_guidance": guidance,
    }
