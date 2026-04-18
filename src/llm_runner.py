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


def run_self_consistency_ranked(
    prompt: str,
    label_space: list[str],
    parse_fn,
    n: int = 5,
    model: str = OPENAI_MODEL,
) -> dict[str, object]:
    # Aggregate multiple generations into a vote-based ranked result.
    vote_counter: Counter[str] = Counter()
    guidance_texts: list[str] = []

    for _ in range(n):
        raw = run_openai(prompt, model=model, temperature=0.7)
        parsed = parse_fn(raw, label_space, top_k=3)

        for rank, label in enumerate(parsed["top_labels"]):
            if label != "unknown":
                vote_counter[label] += 3 - rank

        if parsed["guidance"]:
            guidance_texts.append(parsed["guidance"])

    ranked = vote_counter.most_common(3)
    total_votes = sum(vote_counter.values()) if sum(vote_counter.values()) > 0 else 1

    top_3: list[dict[str, object]] = []
    for label, count in ranked:
        top_3.append({"label": label, "confidence": round(count / total_votes, 4)})

    while len(top_3) < 3:
        top_3.append({"label": "unknown", "confidence": 0.0})

    guidance = guidance_texts[0] if guidance_texts else ""

    return {"top_3": top_3, "guidance": guidance}
