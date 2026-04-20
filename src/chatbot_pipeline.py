# End-to-end chatbot pipeline for structured intake and consultation-style output.

from __future__ import annotations

from functools import lru_cache
import json
import re

import pandas as pd

from src.config import (
    CANDIDATE_NEIGHBORS,
    CANDIDATE_TOP_K,
    EBM_DIR,
    EVIDENCE_TOP_K,
    OPENAI_MODEL,
    REBALANCE_TOP_LABELS,
    SELF_CONSISTENCY_RUNS,
    SYNTHEA_DIR,
    TOP_N_LABELS,
)
from src.data_prep import build_patient_dataset, filter_top_labels
from src.ebm_utils import load_knowledge_snippets
from src.llm_runner import run_openai, run_self_consistency_ranked
from src.parsing import parse_ranked_output
from src.prompts import (
    DEFAULT_VALIDATION_QUESTION,
    STRUCTURED_INTAKE_PROMPT,
    build_case_query,
    build_retrieve_then_reason_prompt,
    build_structured_patient_summary,
)
from src.retrieval import (
    retrieve_candidate_labels_from_examples,
    retrieve_evidence_snippets,
    retrieve_similar_case_summaries,
    retrieve_similar_examples,
)


NUMBERED_FIELDS = {
    "1": "age_sex",
    "2": "main_symptoms",
    "3": "symptoms_started",
    "4": "severity",
    "5": "risk_flags",
    "6": "existing_conditions",
    "7": "recent_worsening",
}

FIELD_ALIASES = {
    "age_sex": ["age / sex", "age and sex", "age sex"],
    "main_symptoms": ["main symptoms", "symptoms", "chief complaint", "presenting symptoms"],
    "symptoms_started": [
        "when the symptoms started",
        "when symptoms started",
        "when did the symptoms start",
        "when it started",
        "started",
    ],
    "severity": ["severity"],
    "risk_flags": [
        "fever? breathing difficulty? chest pain? bleeding? confusion?",
        "fever breathing difficulty chest pain bleeding confusion",
        "fever",
    ],
    "existing_conditions": [
        "existing medical conditions if relevant",
        "existing medical conditions",
        "existing conditions",
        "medical history",
        "relevant existing conditions",
    ],
    "recent_worsening": [
        "anything that has recently worsened",
        "anything recently worsened",
        "recent worsening",
        "anything worsening recently",
    ],
}


def _normalize_spaces(text: object) -> str:
    # Collapse repeated whitespace while preserving the user's meaning.
    return re.sub(r"\s+", " ", str(text)).strip()


def _extract_line_value(line: str) -> tuple[str | None, str]:
    # Map one intake line to a known field when possible.
    original = _normalize_spaces(line)
    if not original:
        return None, ""

    match = re.match(r"^(\d+)[\).:-]?\s*(.*)$", original)
    number = None
    remainder = original
    if match:
        number = match.group(1)
        remainder = match.group(2).strip()

    if number in NUMBERED_FIELDS:
        field_name = NUMBERED_FIELDS[number]
        lowered = remainder.lower()
        if field_name == "risk_flags":
            return field_name, remainder
        for alias in FIELD_ALIASES.get(field_name, []):
            if lowered.startswith(alias):
                value = remainder[len(alias) :].lstrip(" ?:-")
                return field_name, value.strip()
        return field_name, remainder

    lowered = remainder.lower()
    alias_pairs = sorted(
        (
            (field_name, alias)
            for field_name, aliases in FIELD_ALIASES.items()
            for alias in aliases
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for field_name, alias in alias_pairs:
        if lowered.startswith(alias):
            if field_name == "risk_flags":
                return field_name, remainder
            value = remainder[len(alias) :].lstrip(" ?:-")
            return field_name, value.strip()

    return None, remainder


def _extract_age(text: str) -> str:
    # Pull the first plausible age mention from free text.
    match = re.search(r"\b(\d{1,3})\b", text)
    if not match:
        return ""
    age_value = int(match.group(1))
    if 0 < age_value <= 120:
        return str(age_value)
    return ""


def _extract_sex(text: str) -> str:
    # Normalize free-text sex mentions for downstream prompting.
    lowered = text.lower()
    if re.search(r"\b(f|female|woman|girl)\b", lowered):
        return "female"
    if re.search(r"\b(m|male|man|boy)\b", lowered):
        return "male"
    return "not provided"


def _extract_binary_flag(text: str, keywords: list[str]) -> str:
    # Pull yes/no style risk flags from either structured or free-text intake.
    lowered = text.lower()
    if not lowered:
        return "not provided"

    keyword_group = "(?:" + "|".join(re.escape(keyword) for keyword in keywords) + ")"
    after_match = re.search(
        rf"{keyword_group}[^a-z0-9]{{0,20}}(yes|no|y|n)\b",
        lowered,
    )
    if after_match:
        return "yes" if after_match.group(1).startswith("y") else "no"

    before_match = re.search(
        rf"\b(yes|no|y|n)\b[^a-z0-9]{{0,20}}{keyword_group}",
        lowered,
    )
    if before_match:
        return "yes" if before_match.group(1).startswith("y") else "no"

    negative_match = re.search(
        rf"\b(no|denies|without|not)\b[^.;,\n]{{0,30}}{keyword_group}",
        lowered,
    )
    if negative_match:
        return "no"

    if re.search(keyword_group, lowered):
        return "yes"

    return "not provided"


def parse_structured_intake(user_text: str) -> dict[str, str]:
    # Normalize the user's intake message into one structured case dict.
    lines = [line.strip() for line in str(user_text).splitlines() if line.strip()]
    raw_fields: dict[str, list[str]] = {
        "age_sex": [],
        "main_symptoms": [],
        "symptoms_started": [],
        "severity": [],
        "risk_flags": [],
        "existing_conditions": [],
        "recent_worsening": [],
        "uncategorized": [],
    }

    for line in lines:
        field_name, value = _extract_line_value(line)
        target_field = field_name or "uncategorized"
        if value:
            raw_fields[target_field].append(value)

    full_text = _normalize_spaces(user_text)
    age_sex_text = " ".join(raw_fields["age_sex"]) or full_text
    risk_flag_text = " ".join(raw_fields["risk_flags"]) or full_text
    severity_text = " ".join(raw_fields["severity"]) or full_text
    onset_text = " ".join(raw_fields["symptoms_started"]) or full_text

    symptoms_text = " ".join(raw_fields["main_symptoms"]).strip()
    if not symptoms_text:
        symptoms_text = full_text

    severity_match = re.search(r"\b(mild|moderate|severe)\b", severity_text.lower())
    symptoms_started = " ".join(raw_fields["symptoms_started"]).strip()
    if not symptoms_started:
        start_match = re.search(
            r"\b(started|since|for)\b\s+([^.;\n]+)",
            onset_text,
            flags=re.IGNORECASE,
        )
        if start_match:
            symptoms_started = f"{start_match.group(1)} {start_match.group(2)}".strip()

    recent_worsening = " ".join(raw_fields["recent_worsening"]).strip()
    if not recent_worsening:
        worsening_match = re.search(
            r"\b(worse|worsening|rapid worsening|gett?ing worse)\b[^.;\n]*",
            full_text,
            flags=re.IGNORECASE,
        )
        if worsening_match:
            recent_worsening = worsening_match.group(0).strip()

    existing_conditions = " ".join(raw_fields["existing_conditions"]).strip()
    if not existing_conditions:
        history_match = re.search(
            r"\b(history of|existing conditions?|medical history)\b[:\s-]*([^.;\n]+)",
            full_text,
            flags=re.IGNORECASE,
        )
        if history_match:
            existing_conditions = history_match.group(2).strip()

    normalized = {
        "age": _extract_age(age_sex_text) or "not provided",
        "sex": _extract_sex(age_sex_text),
        "symptoms": symptoms_text or "not provided",
        "symptoms_started": symptoms_started or "not provided",
        "severity": severity_match.group(1).lower() if severity_match else "not provided",
        "fever": _extract_binary_flag(risk_flag_text, ["fever", "febrile", "temperature"]),
        "breathing_difficulty": _extract_binary_flag(
            risk_flag_text,
            ["breathing difficulty", "shortness of breath", "dyspnea", "trouble breathing"],
        ),
        "chest_pain": _extract_binary_flag(
            risk_flag_text,
            ["chest pain", "chest pressure", "tightness in chest"],
        ),
        "bleeding": _extract_binary_flag(risk_flag_text, ["bleeding", "hemorrhage", "blood loss"]),
        "confusion": _extract_binary_flag(
            risk_flag_text,
            ["confusion", "confused", "disoriented", "altered mental status"],
        ),
        "existing_conditions": existing_conditions or "not provided",
        "recent_worsening": recent_worsening or "not provided",
    }

    return {key: _normalize_spaces(value) for key, value in normalized.items()}


def needs_structured_intake(intake: dict[str, str]) -> bool:
    # Decide when the chatbot should ask for the intake template again.
    symptoms = intake.get("symptoms", "").strip().lower()
    if not symptoms or symptoms == "not provided":
        return True

    low_signal_inputs = {
        "hi",
        "hello",
        "help",
        "y",
        "n",
        "yes",
        "no",
    }
    return symptoms in low_signal_inputs


def build_intake_request_message() -> str:
    # Render the default intake instruction shown before diagnosis reasoning.
    return (
        "Please share your symptoms in the structured format below so I can compare "
        "them against the closed diagnosis set.\n\n"
        f"{STRUCTURED_INTAKE_PROMPT}"
    )


def _resources_tuple():
    # Provide a typed empty-resource fallback.
    return (
        pd.DataFrame(columns=["age", "sex", "symptoms", "diagnosis"]),
        [],
        pd.DataFrame(columns=["text", "snippet"]),
    )


@lru_cache(maxsize=1)
def load_chatbot_resources() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    # Load the retrieval corpus and optional knowledge base once per process.
    patients_path = SYNTHEA_DIR / "patients.csv"
    conditions_path = SYNTHEA_DIR / "conditions.csv"
    if not patients_path.exists() or not conditions_path.exists():
        return _resources_tuple()

    patients_df = pd.read_csv(patients_path)
    conditions_df = pd.read_csv(conditions_path)
    patient_df = build_patient_dataset(patients_df, conditions_df)
    patient_df, label_space = filter_top_labels(
        patient_df,
        top_n=TOP_N_LABELS,
        rebalance=REBALANCE_TOP_LABELS,
    )

    knowledge_path = EBM_DIR / "knowledge_snippets.csv"
    if knowledge_path.exists():
        knowledge_df = load_knowledge_snippets(knowledge_path)
    else:
        knowledge_df = pd.DataFrame(columns=["text", "snippet"])

    return patient_df.reset_index(drop=True), label_space, knowledge_df


def prepare_chatbot_case(
    intake: dict[str, str],
    patient_df: pd.DataFrame,
    label_space: list[str],
    knowledge_df: pd.DataFrame,
) -> dict[str, object]:
    # Assemble one retrieve-then-reason case package for the chatbot.
    row = pd.Series(intake)
    patient_summary = build_structured_patient_summary(
        age=intake.get("age"),
        sex=intake.get("sex"),
        symptoms=intake.get("symptoms"),
        symptoms_started=intake.get("symptoms_started"),
        severity=intake.get("severity"),
        fever=intake.get("fever"),
        breathing_difficulty=intake.get("breathing_difficulty"),
        chest_pain=intake.get("chest_pain"),
        bleeding=intake.get("bleeding"),
        confusion=intake.get("confusion"),
        existing_conditions=intake.get("existing_conditions"),
        recent_worsening=intake.get("recent_worsening"),
    )
    case_query = build_case_query(
        age=intake.get("age"),
        sex=intake.get("sex"),
        symptoms=intake.get("symptoms"),
        symptoms_started=intake.get("symptoms_started"),
        severity=intake.get("severity"),
        fever=intake.get("fever"),
        breathing_difficulty=intake.get("breathing_difficulty"),
        chest_pain=intake.get("chest_pain"),
        bleeding=intake.get("bleeding"),
        confusion=intake.get("confusion"),
        existing_conditions=intake.get("existing_conditions"),
        recent_worsening=intake.get("recent_worsening"),
        patient_summary=patient_summary,
    )

    similar_examples = retrieve_similar_examples(
        row=row,
        train_df=patient_df,
        n=max(CANDIDATE_NEIGHBORS, 6),
    )
    candidate_labels = retrieve_candidate_labels_from_examples(
        row=row,
        train_df=patient_df,
        label_space=label_space,
        k_neighbors=CANDIDATE_NEIGHBORS,
        k=CANDIDATE_TOP_K,
    )
    similar_cases = retrieve_similar_case_summaries(row=row, train_df=patient_df, n=2)
    evidence_snippets = retrieve_evidence_snippets(
        case_query=case_query,
        knowledge_snippets=knowledge_df,
        k=EVIDENCE_TOP_K,
        label_space=label_space,
        candidate_labels=candidate_labels,
        row=row,
        train_df=patient_df,
        similar_examples=similar_examples,
    )
    prompt = build_retrieve_then_reason_prompt(
        age=intake.get("age"),
        sex=intake.get("sex"),
        symptoms=intake.get("symptoms"),
        symptoms_started=intake.get("symptoms_started"),
        severity=intake.get("severity"),
        fever=intake.get("fever"),
        breathing_difficulty=intake.get("breathing_difficulty"),
        chest_pain=intake.get("chest_pain"),
        bleeding=intake.get("bleeding"),
        confusion=intake.get("confusion"),
        existing_conditions=intake.get("existing_conditions"),
        recent_worsening=intake.get("recent_worsening"),
        candidate_labels=candidate_labels,
        evidence_snippets=evidence_snippets,
        similar_cases=similar_cases,
        patient_summary=patient_summary,
    )

    return {
        "row": row,
        "patient_summary": patient_summary,
        "case_query": case_query,
        "candidate_labels": candidate_labels,
        "similar_cases": similar_cases,
        "evidence_snippets": evidence_snippets,
        "prompt": prompt,
    }


def _format_label(label: str) -> str:
    # Make diagnosis labels easier to scan in the chatbot response.
    lowered = str(label).strip().lower()
    special_labels = {
        "covid-19": "COVID-19",
        "alzheimer's disease": "Alzheimer's disease",
    }
    if lowered in special_labels:
        return special_labels[lowered]
    return lowered.title()


def _score_percentages(scores: list[float]) -> list[int]:
    # Convert 0-1 score shares into rounded percentages without renormalizing hidden mass.
    clipped = [max(0.0, min(1.0, float(score))) for score in scores]
    return [int(round(score * 100)) for score in clipped]


def _default_consultation_guidance() -> str:
    # Keep final wording aligned with consultation support framing.
    return (
        "This is a data-based support output, not a confirmed diagnosis. "
        "Please discuss these ranked possibilities with a clinician."
    )


def render_chatbot_response(result: dict[str, object]) -> str:
    # Turn the parsed JSON result into the approachable final chatbot message.
    ranked_diagnoses = list(result.get("ranked_diagnoses", []))[:3]
    display_scores = [
        float(item.get("display_score", item.get("confidence", 0.0)))
        for item in ranked_diagnoses
    ]
    percentages = _score_percentages(display_scores)
    leading_top1_agreement = (
        float(ranked_diagnoses[0].get("confidence", 0.0))
        if ranked_diagnoses
        else 0.0
    )

    lines = [
        "Most likely patterns from similar cases in the dataset:",
        "",
    ]

    for index, (item, percentage) in enumerate(zip(ranked_diagnoses, percentages), start=1):
        label = _format_label(str(item.get("label", "unknown")))
        lines.append(f"{index}. {label} - {percentage}% display support")

    lines.extend(
        [
            "",
            "Display score note:",
            "These percentages reflect candidate support across ranks 1-3 in repeated model runs. Internally, strict confidence still tracks only how often a condition finished first. These are not medical probabilities.",
            "",
            "Why this was suggested:",
        ]
    )

    for item in ranked_diagnoses:
        label = _format_label(str(item.get("label", "unknown")))
        evidence = ", ".join(item.get("supporting_evidence", [])[:3]) or "pattern overlap in retrieved cases"
        missing = ", ".join(item.get("missing_or_uncertain", [])[:2])
        explanation = f"- {label}: strongest overlap with {evidence}"
        if missing:
            explanation += f"; uncertainty remains around {missing}"
        lines.append(explanation)

    similar_cases = list(result.get("similar_cases", []))
    lines.extend(["", "Similar example cases:"])
    if similar_cases:
        lines.extend(f"- {case}" for case in similar_cases)
    else:
        lines.append("- No close display examples were retrieved.")

    validation_question = str(
        result.get("validation_question", DEFAULT_VALIDATION_QUESTION)
    ).strip() or DEFAULT_VALIDATION_QUESTION
    guidance = str(result.get("consultation_guidance", "")).strip() or _default_consultation_guidance()
    urgent_line = (
        "Seek urgent care if severe breathing difficulty, chest pain, confusion, "
        "heavy bleeding, or rapid worsening appears."
    )

    lines.extend(
        [
            "",
            f"Internal top-choice agreement for the leading pattern: {int(round(leading_top1_agreement * 100))}%",
            "",
            f"Validation question: {validation_question}",
            "",
            f"Consultation guidance: {guidance}",
            urgent_line,
        ]
    )

    return "\n".join(lines).strip()


def run_chatbot_pipeline(user_text: str, model: str = OPENAI_MODEL) -> dict[str, object]:
    # Run the full structured-intake chatbot flow for one user message.
    intake = parse_structured_intake(user_text)
    if needs_structured_intake(intake):
        return {
            "needs_intake": True,
            "intake": intake,
            "reply": build_intake_request_message(),
        }

    patient_df, label_space, knowledge_df = load_chatbot_resources()
    if patient_df.empty or not label_space:
        return {
            "needs_intake": False,
            "intake": intake,
            "reply": (
                "The local retrieval resources are not available yet, so the diagnostic "
                "reasoning pipeline cannot run. Please add the Synthea data files and restart the app."
            ),
        }

    prepared = prepare_chatbot_case(
        intake=intake,
        patient_df=patient_df,
        label_space=label_space,
        knowledge_df=knowledge_df,
    )

    if SELF_CONSISTENCY_RUNS > 1:
        aggregated = run_self_consistency_ranked(
            prompt=prepared["prompt"],
            label_space=label_space,
            parse_fn=parse_ranked_output,
            n=SELF_CONSISTENCY_RUNS,
            model=model,
        )
        parsed = parse_ranked_output(json.dumps(aggregated), label_space, top_k=3)
    else:
        raw_output = run_openai(prepared["prompt"], model=model, temperature=0)
        parsed = parse_ranked_output(raw_output, label_space, top_k=3)

    parsed["similar_cases"] = retrieve_similar_case_summaries(
        row=prepared["row"],
        train_df=patient_df,
        n=2,
        preferred_labels=parsed.get("top_labels", [])[:2],
    ) or prepared["similar_cases"]
    parsed["validation_question"] = DEFAULT_VALIDATION_QUESTION
    parsed["consultation_guidance"] = (
        str(parsed.get("consultation_guidance", "")).strip() or _default_consultation_guidance()
    )

    return {
        "needs_intake": False,
        "intake": intake,
        "patient_summary": prepared["patient_summary"],
        "case_query": prepared["case_query"],
        "candidate_labels": prepared["candidate_labels"],
        "evidence_snippets": prepared["evidence_snippets"],
        "similar_cases": prepared["similar_cases"],
        "prompt": prepared["prompt"],
        "parsed": parsed,
        "reply": render_chatbot_response(parsed),
    }
