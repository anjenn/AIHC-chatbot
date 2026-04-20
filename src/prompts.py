# Prompt templates for the healthcare chatbot reasoning core.

from __future__ import annotations

from textwrap import dedent


STRUCTURED_INTAKE_PROMPT = dedent(
    """
    Please summarize your condition in this format:

    1. Age / sex
    2. Main symptoms
    3. When the symptoms started
    4. Severity (mild / moderate / severe)
    5. Fever? Breathing difficulty? Chest pain? Bleeding? Confusion? (yes/no)
    6. Existing medical conditions if relevant
    7. Anything that has recently worsened
    """
).strip()

DEFAULT_VALIDATION_QUESTION = "Does your condition seem similar to these example patterns? (Y/N)"

DEFAULT_FEW_SHOT_CONTEXT = dedent(
    """
    Example 1
    Input case:
    - Age / sex: 44 / female
    - Main symptoms: cough, sore throat, fatigue
    - When symptoms started: 4 days ago
    - Severity: moderate
    - Fever: yes
    - Breathing difficulty: no
    - Chest pain: no
    - Bleeding: no
    - Confusion: no
    - Existing conditions: mild asthma
    - Recent worsening: cough is getting more frequent

    Output JSON:
    {
      "primary_diagnosis": "acute bronchitis",
      "ranked_diagnoses": [
        {
          "label": "acute bronchitis",
          "confidence": 0.68,
          "supporting_evidence": ["cough", "fatigue", "worsening respiratory symptoms"],
          "missing_or_uncertain": ["no sputum description"]
        },
        {
          "label": "acute viral pharyngitis",
          "confidence": 0.22,
          "supporting_evidence": ["sore throat", "short illness duration"],
          "missing_or_uncertain": ["cough is less typical as the main complaint"]
        },
        {
          "label": "covid-19",
          "confidence": 0.10,
          "supporting_evidence": ["cough", "fatigue", "fever"],
          "missing_or_uncertain": ["no known exposure or breathing difficulty"]
        }
      ],
      "similar_cases": [
        "Case A: 42-year-old female with cough, sore throat, and fatigue -> acute bronchitis",
        "Case B: 38-year-old male with sore throat and mild fever -> acute viral pharyngitis"
      ],
      "validation_question": "Does your condition seem similar to these example patterns? (Y/N)",
      "consultation_guidance": "This is a ranking from learned patterns, not a confirmed diagnosis. Discuss these possibilities with a clinician."
    }

    Example 2
    Input case:
    - Age / sex: 71 / male
    - Main symptoms: shortness of breath, chest discomfort, leg swelling
    - When symptoms started: today
    - Severity: severe
    - Fever: no
    - Breathing difficulty: yes
    - Chest pain: yes
    - Bleeding: no
    - Confusion: no
    - Existing conditions: recent immobility
    - Recent worsening: breathing became worse quickly

    Output JSON:
    {
      "primary_diagnosis": "acute pulmonary embolism",
      "ranked_diagnoses": [
        {
          "label": "acute pulmonary embolism",
          "confidence": 0.74,
          "supporting_evidence": ["shortness of breath", "chest discomfort", "rapid worsening"],
          "missing_or_uncertain": ["no oxygen saturation provided"]
        },
        {
          "label": "acute deep venous thrombosis",
          "confidence": 0.18,
          "supporting_evidence": ["leg swelling", "recent immobility"],
          "missing_or_uncertain": ["breathing symptoms suggest something more than isolated DVT"]
        },
        {
          "label": "covid-19",
          "confidence": 0.08,
          "supporting_evidence": ["shortness of breath"],
          "missing_or_uncertain": ["no fever or viral symptom cluster"]
        }
      ],
      "similar_cases": [
        "Case A: 69-year-old male with acute shortness of breath and chest pain -> acute pulmonary embolism",
        "Case B: 66-year-old female with leg swelling and sudden dyspnea -> acute deep venous thrombosis"
      ],
      "validation_question": "Does your condition seem similar to these example patterns? (Y/N)",
      "consultation_guidance": "This pattern may require urgent clinical assessment, especially if breathing difficulty or chest pain is severe."
    }
    """
).strip()


def _safe_age_text(age: object) -> str:
    # Convert age-like values into a stable prompt string.
    try:
        return f"{int(float(age))}-year-old"
    except (TypeError, ValueError):
        return "age not provided"


def _safe_sex_text(sex: object) -> str:
    # Normalize the sex field without forcing a medically specific claim.
    sex_text = str(sex).strip().lower()
    if not sex_text or sex_text == "nan":
        return "sex not provided"
    if sex_text in {"f", "female", "woman"}:
        return "female"
    if sex_text in {"m", "male", "man"}:
        return "male"
    return sex_text


def _safe_text(value: object, fallback: str = "not provided") -> str:
    # Normalize text fields for prompt readability.
    text = " ".join(str(value).split())
    if not text or text.lower() == "nan":
        return fallback
    return text


def _safe_binary_flag(value: object) -> str:
    # Normalize tri-state intake flags into yes/no/not provided text.
    text = _safe_text(value).lower()
    if text in {"y", "yes", "true"}:
        return "yes"
    if text in {"n", "no", "false"}:
        return "no"
    return "not provided"


def patient_block(row) -> str:
    # Format a patient case into a compact prompt block.
    return build_structured_patient_summary(
        age=row.get("age"),
        sex=row.get("sex"),
        symptoms=row.get("symptoms", ""),
        symptoms_started=row.get("symptoms_started", row.get("when_symptoms_started", "")),
        severity=row.get("severity", ""),
        fever=row.get("fever", ""),
        breathing_difficulty=row.get("breathing_difficulty", ""),
        chest_pain=row.get("chest_pain", ""),
        bleeding=row.get("bleeding", ""),
        confusion=row.get("confusion", ""),
        existing_conditions=row.get("existing_conditions", ""),
        recent_worsening=row.get("recent_worsening", ""),
    )


def allowed_labels_text(label_space: list[str]) -> str:
    # Format the closed diagnosis label set.
    return ", ".join(label_space)


def build_structured_patient_summary(
    age: object,
    sex: object,
    symptoms: object,
    symptoms_started: object = "",
    severity: object = "",
    fever: object = "",
    breathing_difficulty: object = "",
    chest_pain: object = "",
    bleeding: object = "",
    confusion: object = "",
    existing_conditions: object = "",
    recent_worsening: object = "",
) -> str:
    # Merge intake fields into one stable summary block for prompting and display.
    return dedent(
        f"""
        Structured patient summary:
        - Age / sex: {_safe_age_text(age)} {_safe_sex_text(sex)}
        - Main symptoms: {_safe_text(symptoms)}
        - When the symptoms started: {_safe_text(symptoms_started)}
        - Severity: {_safe_text(severity)}
        - Fever: {_safe_binary_flag(fever)}
        - Breathing difficulty: {_safe_binary_flag(breathing_difficulty)}
        - Chest pain: {_safe_binary_flag(chest_pain)}
        - Bleeding: {_safe_binary_flag(bleeding)}
        - Confusion: {_safe_binary_flag(confusion)}
        - Existing medical conditions: {_safe_text(existing_conditions)}
        - Anything recently worsened: {_safe_text(recent_worsening)}
        """
    ).strip()


def format_structured_intake_message(
    age: object,
    sex: object,
    symptoms: object,
    symptoms_started: object = "",
    severity: object = "",
    fever: object = "",
    breathing_difficulty: object = "",
    chest_pain: object = "",
    bleeding: object = "",
    confusion: object = "",
    existing_conditions: object = "",
    recent_worsening: object = "",
) -> str:
    # Format a paste-ready numbered intake message for chatbot experiments.
    return dedent(
        f"""
        1. Age / sex: {_safe_text(age)} / {_safe_text(sex)}
        2. Main symptoms: {_safe_text(symptoms)}
        3. When the symptoms started: {_safe_text(symptoms_started)}
        4. Severity: {_safe_text(severity)}
        5. Fever? {_safe_binary_flag(fever)}. Breathing difficulty? {_safe_binary_flag(breathing_difficulty)}. Chest pain? {_safe_binary_flag(chest_pain)}. Bleeding? {_safe_binary_flag(bleeding)}. Confusion? {_safe_binary_flag(confusion)}.
        6. Existing medical conditions if relevant: {_safe_text(existing_conditions)}
        7. Anything that has recently worsened: {_safe_text(recent_worsening)}
        """
    ).strip()


def build_case_query(
    age: object,
    sex: object,
    symptoms: object,
    symptoms_started: object = "",
    severity: object = "",
    fever: object = "",
    breathing_difficulty: object = "",
    chest_pain: object = "",
    bleeding: object = "",
    confusion: object = "",
    existing_conditions: object = "",
    recent_worsening: object = "",
    patient_summary: str = "",
) -> str:
    # Build a case-style query for retrieval and comparison prompting.
    structured_summary = patient_summary.strip() or build_structured_patient_summary(
        age=age,
        sex=sex,
        symptoms=symptoms,
        symptoms_started=symptoms_started,
        severity=severity,
        fever=fever,
        breathing_difficulty=breathing_difficulty,
        chest_pain=chest_pain,
        bleeding=bleeding,
        confusion=confusion,
        existing_conditions=existing_conditions,
        recent_worsening=recent_worsening,
    )
    return dedent(
        f"""
        Clinical case query:
        {structured_summary}

        Task:
        Identify the most plausible differential diagnoses from the allowed label space.
        Compare the candidates explicitly, note which findings support or weaken each one,
        and stay honest about missing evidence. Focus on consultation-oriented diagnostic
        reasoning support rather than definitive diagnosis.
        """
    ).strip()


def _format_list_block(title: str, items: list[str], fallback: str) -> str:
    # Render a compact bullet block with a stable fallback.
    if not items:
        return f"{title}\n- {fallback}"
    return f"{title}\n" + "\n".join(f"- {item}" for item in items)


def format_candidate_evidence(
    candidate_labels: list[str],
    evidence_snippets: list[str],
    similar_cases: list[str] | None = None,
) -> str:
    # Format candidate diagnoses, evidence, and similar cases into comparable blocks.
    candidate_block = _format_list_block(
        "Candidate diagnoses:",
        candidate_labels,
        "No candidate labels were retrieved.",
    )
    evidence_block = _format_list_block(
        "Retrieved evidence blocks:",
        evidence_snippets,
        "No compact evidence blocks were retrieved.",
    )
    similar_case_block = _format_list_block(
        "Retrieved similar cases:",
        similar_cases or [],
        "No similar cases were retrieved.",
    )
    return "\n\n".join([candidate_block, evidence_block, similar_case_block])


def build_retrieve_then_reason_prompt(
    age: object,
    sex: object,
    symptoms: object,
    candidate_labels: list[str],
    evidence_snippets: list[str],
    few_shot_context: str = "",
    symptoms_started: object = "",
    severity: object = "",
    fever: object = "",
    breathing_difficulty: object = "",
    chest_pain: object = "",
    bleeding: object = "",
    confusion: object = "",
    existing_conditions: object = "",
    recent_worsening: object = "",
    similar_cases: list[str] | None = None,
    patient_summary: str = "",
) -> str:
    # Build the staged compare-and-rank prompt for the retrieve-then-reason pipeline.
    case_query = build_case_query(
        age=age,
        sex=sex,
        symptoms=symptoms,
        symptoms_started=symptoms_started,
        severity=severity,
        fever=fever,
        breathing_difficulty=breathing_difficulty,
        chest_pain=chest_pain,
        bleeding=bleeding,
        confusion=confusion,
        existing_conditions=existing_conditions,
        recent_worsening=recent_worsening,
        patient_summary=patient_summary,
    )
    evidence_block = format_candidate_evidence(
        candidate_labels=candidate_labels,
        evidence_snippets=evidence_snippets,
        similar_cases=similar_cases,
    )

    example_sections = [DEFAULT_FEW_SHOT_CONTEXT]
    if few_shot_context.strip():
        example_sections.append(few_shot_context.strip())
    combined_examples = "\n\n".join(section for section in example_sections if section.strip())

    sections = [
        dedent(
            """
            You are assisting with differential diagnosis ranking in a controlled benchmark.
            Do not claim clinical certainty.
            Do not invent diagnoses outside the candidate set.
            Do not rely on explicit diagnosis mentions in the symptom list.
            If a finding directly names a diagnosis or suspected diagnosis, ignore that surface form
            and focus on the remaining clinical pattern.
            Confidence values must be proportions between 0 and 1 that roughly sum to 1 across the
            ranked diagnoses. Treat them as model-confidence estimates from learned patterns, not as
            true medical probabilities.
            """
        ).strip(),
        dedent(
            f"""
            Reference benchmark examples:
            {combined_examples}
            """
        ).strip(),
        case_query,
        evidence_block,
        dedent(
            f"""
            Instructions:
            1. Compare the patient symptoms against each candidate diagnosis.
            2. Identify which findings support each candidate.
            3. Identify what evidence is missing, weak, or uncertain.
            4. Rank the top 3 diagnoses from most to least plausible.
            5. Copy similar cases only from the retrieved similar case block when available.
            6. Use the validation question exactly as written below.
            7. Provide short consultation guidance.
            8. Return valid JSON only.

            Allowed candidate labels:
            {allowed_labels_text(candidate_labels)}

            Required validation question:
            {DEFAULT_VALIDATION_QUESTION}

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
              "similar_cases": ["Case A: ...", "Case B: ..."],
              "validation_question": "{DEFAULT_VALIDATION_QUESTION}",
              "consultation_guidance": "short guidance"
            }}
            """
        ).strip(),
    ]

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
        symptoms_started=row.get("symptoms_started", row.get("when_symptoms_started", "")),
        severity=row.get("severity", ""),
        fever=row.get("fever", ""),
        breathing_difficulty=row.get("breathing_difficulty", ""),
        chest_pain=row.get("chest_pain", ""),
        bleeding=row.get("bleeding", ""),
        confusion=row.get("confusion", ""),
        existing_conditions=row.get("existing_conditions", ""),
        recent_worsening=row.get("recent_worsening", ""),
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
        symptoms_started=row.get("symptoms_started", row.get("when_symptoms_started", "")),
        severity=row.get("severity", ""),
        fever=row.get("fever", ""),
        breathing_difficulty=row.get("breathing_difficulty", ""),
        chest_pain=row.get("chest_pain", ""),
        bleeding=row.get("bleeding", ""),
        confusion=row.get("confusion", ""),
        existing_conditions=row.get("existing_conditions", ""),
        recent_worsening=row.get("recent_worsening", ""),
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
        symptoms_started=row.get("symptoms_started", row.get("when_symptoms_started", "")),
        severity=row.get("severity", ""),
        fever=row.get("fever", ""),
        breathing_difficulty=row.get("breathing_difficulty", ""),
        chest_pain=row.get("chest_pain", ""),
        bleeding=row.get("bleeding", ""),
        confusion=row.get("confusion", ""),
        existing_conditions=row.get("existing_conditions", ""),
        recent_worsening=row.get("recent_worsening", ""),
    )
