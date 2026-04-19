# Prompt templates for the healthcare chatbot reasoning core.

from __future__ import annotations

from textwrap import dedent


def _safe_age_text(age: object) -> str:
    # Convert age-like values into a stable prompt string.
    try:
        return f"{int(float(age))}-year-old"
    except (TypeError, ValueError):
        return "adult"


def _safe_sex_text(sex: object) -> str:
    # Normalize the sex field without forcing a medically specific claim.
    sex_text = str(sex).strip().lower()
    if not sex_text or sex_text == "nan":
        return "patient"
    return sex_text


def patient_block(row) -> str:
    # Format a patient case into a compact prompt block.
    return (
        f"Age: {_safe_age_text(row.get('age'))}\n"
        f"Sex: {_safe_sex_text(row.get('sex'))}\n"
        f"Reported findings: {' '.join(str(row.get('symptoms', '')).split())}"
    )


def allowed_labels_text(label_space: list[str]) -> str:
    # Format the closed diagnosis label set.
    return ", ".join(label_space)


def build_case_query(age: object, sex: object, symptoms: object) -> str:
    # Build a case-style query for retrieval and comparison prompting.
    symptoms_clean = " ".join(str(symptoms).split())
    return dedent(
        f"""
        Clinical case query:
        Patient: {_safe_age_text(age)} {_safe_sex_text(sex)}
        Presenting symptoms/findings: {symptoms_clean}

        Task:
        Identify the most plausible differential diagnoses and the distinguishing findings
        that separate them. Focus on symptom pattern matching and differential diagnosis,
        not treatment.
        """
    ).strip()


def format_candidate_evidence(
    candidate_labels: list[str],
    evidence_snippets: list[str],
) -> str:
    # Format candidate diagnoses and retrieved evidence into comparable blocks.
    candidate_block = "\n".join(f"- {label}" for label in candidate_labels)

    if evidence_snippets:
        evidence_block = "\n".join(
            f"- Evidence snippet {idx + 1}: {' '.join(str(snippet).split())}"
            for idx, snippet in enumerate(evidence_snippets)
        )
    else:
        evidence_block = "- No external evidence retrieved."

    return (
        "Candidate diagnoses:\n"
        f"{candidate_block}\n\n"
        "Retrieved evidence:\n"
        f"{evidence_block}"
    )


def build_retrieve_then_reason_prompt(
    age: object,
    sex: object,
    symptoms: object,
    candidate_labels: list[str],
    evidence_snippets: list[str],
    few_shot_context: str = "",
) -> str:
    # Build the staged compare-and-rank prompt for the retrieve-then-reason pipeline.
    case_query = build_case_query(age=age, sex=sex, symptoms=symptoms)
    evidence_block = format_candidate_evidence(candidate_labels, evidence_snippets)

    sections = [
        dedent(
            """
            You are assisting with differential diagnosis ranking in a controlled benchmark.
            Do not claim clinical certainty.
            Do not invent diagnoses outside the candidate set.
            Do not rely on explicit diagnosis mentions in the symptom list.
            If a finding directly names a diagnosis or suspected diagnosis, ignore that surface form
            and focus on the remaining clinical pattern.
            """
        ).strip()
    ]

    if few_shot_context.strip():
        sections.append(
            dedent(
                f"""
                Reference benchmark examples:
                {few_shot_context.strip()}
                """
            ).strip()
        )

    sections.extend(
        [
            case_query,
            evidence_block,
            dedent(
                f"""
                Instructions:
                1. Compare the patient symptoms against each candidate diagnosis.
                2. Identify which findings support each candidate.
                3. Identify what evidence is missing, weak, or uncertain.
                4. Rank the top 3 diagnoses from most to least plausible.
                5. Provide brief consultation guidance.
                6. Return valid JSON only.

                Allowed candidate labels:
                {allowed_labels_text(candidate_labels)}

                Required JSON schema:
                {{
                  "primary_diagnosis": "one allowed label",
                  "ranked_diagnoses": [
                    {{
                      "label": "one allowed label",
                      "confidence": 0.0,
                      "supporting_evidence": ["short phrase", "short phrase"],
                      "missing_or_uncertain": ["short phrase"]
                    }},
                    {{
                      "label": "one allowed label",
                      "confidence": 0.0,
                      "supporting_evidence": ["short phrase"],
                      "missing_or_uncertain": ["short phrase"]
                    }},
                    {{
                      "label": "one allowed label",
                      "confidence": 0.0,
                      "supporting_evidence": ["short phrase"],
                      "missing_or_uncertain": ["short phrase"]
                    }}
                  ],
                  "consultation_guidance": "short guidance"
                }}
                """
            ).strip(),
        ]
    )

    return "\n\n".join(section for section in sections if section).strip()


def prompt_zero_shot_ranked(row, label_space: list[str]) -> str:
    # Backward-compatible zero-shot wrapper around the retrieve-then-reason prompt.
    candidate_labels = label_space[: min(5, len(label_space))] or label_space
    return build_retrieve_then_reason_prompt(
        age=row.get("age"),
        sex=row.get("sex"),
        symptoms=row.get("symptoms", ""),
        candidate_labels=candidate_labels,
        evidence_snippets=[],
    )


def prompt_few_shot_ranked(
    row,
    few_shot_context: str,
    label_space: list[str],
    candidate_labels: list[str] | None = None,
) -> str:
    # Backward-compatible few-shot wrapper around the retrieve-then-reason prompt.
    return build_retrieve_then_reason_prompt(
        age=row.get("age"),
        sex=row.get("sex"),
        symptoms=row.get("symptoms", ""),
        candidate_labels=candidate_labels or label_space,
        evidence_snippets=[],
        few_shot_context=few_shot_context,
    )


def prompt_knowledge_ranked(
    row,
    knowledge_snippets: list[str],
    label_space: list[str],
    candidate_labels: list[str] | None = None,
) -> str:
    # Backward-compatible evidence wrapper around the retrieve-then-reason prompt.
    return build_retrieve_then_reason_prompt(
        age=row.get("age"),
        sex=row.get("sex"),
        symptoms=row.get("symptoms", ""),
        candidate_labels=candidate_labels or label_space,
        evidence_snippets=knowledge_snippets,
    )
