# Output normalization and JSON parsing utilities.

from __future__ import annotations

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
        return None


def parse_ranked_output(
    raw_output: str,
    label_space: list[str],
    top_k: int = 3,
) -> dict[str, object]:
    # Extract top labels, confidences, and guidance from model output.
    parsed = safe_json_load(raw_output)

    default_output = {
        "top_labels": ["unknown"] * top_k,
        "top_confidences": [0.0] * top_k,
        "guidance": "",
    }

    if parsed is None:
        return default_output

    items = parsed.get("top_3", [])
    guidance = str(parsed.get("guidance", "")).strip()

    labels: list[str] = []
    confidences: list[float] = []

    for item in items[:top_k]:
        label = normalize_label_to_allowed(item.get("label", ""), label_space)

        try:
            conf = float(item.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        labels.append(label)
        confidences.append(conf)

    while len(labels) < top_k:
        labels.append("unknown")
        confidences.append(0.0)

    return {
        "top_labels": labels,
        "top_confidences": confidences,
        "guidance": guidance,
    }
