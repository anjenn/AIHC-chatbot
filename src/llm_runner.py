# OpenAI model runners for single-pass and self-consistency experiments.

from __future__ import annotations

from collections import Counter

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.prompts import DEFAULT_VALIDATION_QUESTION


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Add it to .env before running experiments."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def run_openai(prompt: str, model: str = OPENAI_MODEL, temperature: float = 0):
    # Run a single OpenAI chat completion call.
    client = _get_client()

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
        if getattr(response, "output_text", None):
            return response.output_text.strip()
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    text_chunks: list[str] = []
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                text_chunks.append(text)

    return "\n".join(text_chunks).strip()


def aggregate_self_consistency(
    parsed_outputs: list[dict[str, object]],
    top_k: int = 3,
) -> dict[str, object]:
    # Convert repeated model samples into vote-based ranked diagnoses.
    vote_counter: Counter[str] = Counter()
    support_counter: Counter[str] = Counter()
    label_metadata: dict[str, dict[str, object]] = {}
    guidance = ""
    similar_cases: list[str] = []
    validation_question = DEFAULT_VALIDATION_QUESTION

    for parsed in parsed_outputs:
        if not guidance:
            guidance = str(
                parsed.get("consultation_guidance", parsed.get("guidance", ""))
            ).strip()
        if not similar_cases:
            similar_cases = list(parsed.get("similar_cases", []))
        if validation_question == DEFAULT_VALIDATION_QUESTION:
            validation_question = str(
                parsed.get("validation_question", DEFAULT_VALIDATION_QUESTION)
            ).strip() or DEFAULT_VALIDATION_QUESTION

        primary = str(parsed.get("primary_diagnosis", "")).strip()
        if primary and primary != "unknown":
            vote_counter[primary] += 1

        for rank, item in enumerate(parsed.get("ranked_diagnoses", []), start=1):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if not label or label == "unknown":
                continue
            if label not in label_metadata:
                label_metadata[label] = item
            support_counter[label] += max(top_k - rank + 1, 1)

    total_votes = sum(vote_counter.values())
    total_support = sum(support_counter.values())
    ranked_labels: list[str] = []
    for label, _ in vote_counter.most_common():
        if label not in ranked_labels:
            ranked_labels.append(label)
    for label, _ in support_counter.most_common():
        if label not in ranked_labels:
            ranked_labels.append(label)
        if len(ranked_labels) >= top_k:
            break

    ranked_diagnoses: list[dict[str, object]] = []

    for label in ranked_labels[:top_k]:
        count = vote_counter.get(label, 0)
        template = label_metadata.get(
            label,
            {"supporting_evidence": [], "missing_or_uncertain": []},
        )
        ranked_diagnoses.append(
            {
                "label": label,
                "confidence": round(count / total_votes, 4) if total_votes else 0.0,
                "display_score": round(
                    support_counter.get(label, 0) / total_support,
                    4,
                )
                if total_support
                else 0.0,
                "supporting_evidence": list(template.get("supporting_evidence", [])),
                "missing_or_uncertain": list(template.get("missing_or_uncertain", [])),
            }
        )

    while len(ranked_diagnoses) < top_k:
        ranked_diagnoses.append(
            {
                "label": "unknown",
                "confidence": 0.0,
                "display_score": 0.0,
                "supporting_evidence": [],
                "missing_or_uncertain": [],
            }
        )

    primary = ranked_diagnoses[0]["label"] if ranked_diagnoses else "unknown"
    top_labels = [str(item["label"]) for item in ranked_diagnoses[:top_k]]
    top_confidences = [float(item["confidence"]) for item in ranked_diagnoses[:top_k]]
    top_display_scores = [float(item["display_score"]) for item in ranked_diagnoses[:top_k]]
    top_3 = [
        {
            "label": label,
            "confidence": confidence,
            "display_score": display_score,
        }
        for label, confidence, display_score in zip(
            top_labels,
            top_confidences,
            top_display_scores,
        )
    ]

    return {
        "primary_diagnosis": primary,
        "ranked_diagnoses": ranked_diagnoses,
        "consultation_guidance": guidance,
        "guidance": guidance,
        "similar_cases": similar_cases[:3],
        "validation_question": validation_question,
        "top_labels": top_labels,
        "top_confidences": top_confidences,
        "top_display_scores": top_display_scores,
        "top_3": top_3,
        "vote_counter": dict(vote_counter),
        "support_counter": dict(support_counter),
    }


def run_self_consistency_ranked(
    prompt: str,
    label_space: list[str],
    parse_fn,
    n: int = 5,
    model: str = OPENAI_MODEL,
) -> dict[str, object]:
    # Aggregate multiple generations into a vote-based ranked result.
    parsed_outputs: list[dict[str, object]] = []

    for _ in range(n):
        raw = run_openai(prompt, model=model, temperature=0.7)
        parsed_outputs.append(parse_fn(raw, label_space, top_k=3))

    return aggregate_self_consistency(parsed_outputs, top_k=3)
