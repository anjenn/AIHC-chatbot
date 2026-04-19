# OpenAI model runners for single-pass and self-consistency experiments.

from __future__ import annotations

from collections import Counter

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL


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
    label_metadata: dict[str, dict[str, object]] = {}
    guidance = ""

    for parsed in parsed_outputs:
        if not guidance:
            guidance = str(
                parsed.get("consultation_guidance", parsed.get("guidance", ""))
            ).strip()

        primary = str(parsed.get("primary_diagnosis", "")).strip()
        if primary and primary != "unknown":
            vote_counter[primary] += 1

        for item in parsed.get("ranked_diagnoses", []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            if label and label not in label_metadata:
                label_metadata[label] = item

    total_votes = sum(vote_counter.values())
    ranked_votes = vote_counter.most_common(top_k)
    ranked_diagnoses: list[dict[str, object]] = []

    for label, count in ranked_votes:
        template = label_metadata.get(
            label,
            {"supporting_evidence": [], "missing_or_uncertain": []},
        )
        ranked_diagnoses.append(
            {
                "label": label,
                "confidence": round(count / total_votes, 4) if total_votes else 0.0,
                "supporting_evidence": list(template.get("supporting_evidence", [])),
                "missing_or_uncertain": list(template.get("missing_or_uncertain", [])),
            }
        )

    while len(ranked_diagnoses) < top_k:
        ranked_diagnoses.append(
            {
                "label": "unknown",
                "confidence": 0.0,
                "supporting_evidence": [],
                "missing_or_uncertain": [],
            }
        )

    primary = ranked_diagnoses[0]["label"] if ranked_diagnoses else "unknown"
    top_3 = [
        {"label": item["label"], "confidence": item["confidence"]}
        for item in ranked_diagnoses
    ]

    return {
        "primary_diagnosis": primary,
        "ranked_diagnoses": ranked_diagnoses,
        "consultation_guidance": guidance,
        "guidance": guidance,
        "top_3": top_3,
        "vote_counter": dict(vote_counter),
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
