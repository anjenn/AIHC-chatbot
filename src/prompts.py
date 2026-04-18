# Prompt templates for the healthcare chatbot reasoning core.

from __future__ import annotations


def patient_block(row) -> str:
    # Format a patient case into a compact prompt block.
    return (
        f"Age: {int(row['age'])}\n"
        f"Sex: {row['sex']}\n"
        f"Reported findings: {row['symptoms']}"
    )


def allowed_labels_text(label_space: list[str]) -> str:
    # Format the closed diagnosis label set.
    return ", ".join(label_space)


def prompt_zero_shot_ranked(row, label_space: list[str]) -> str:
    # Zero-shot prompt for ranked top-3 diagnosis output.
    return f"""
You are assisting as the diagnostic reasoning core of a healthcare chatbot.

Patient profile:
{patient_block(row)}

Allowed diagnosis labels:
{allowed_labels_text(label_space)}

Task:
1. Return the top 3 most likely diagnoses from the allowed labels only.
2. Rank them from most likely to least likely.
3. Give a short, cautious consultation guidance message.
4. Do not claim certainty.
5. Do NOT rely on explicit diagnosis mentions in the symptom list.
6. If a finding directly names a diagnosis or a suspected diagnosis, ignore that surface form and focus on the remaining clinical pattern.

Return valid JSON with this format:
{{
  "top_3": [
    {{"label": "<label1>", "confidence": 0.0}},
    {{"label": "<label2>", "confidence": 0.0}},
    {{"label": "<label3>", "confidence": 0.0}}
  ],
  "guidance": "<short cautious advice>"
}}
""".strip()


def prompt_few_shot_ranked(
    row,
    few_shot_context: str,
    label_space: list[str],
) -> str:
    # Few-shot prompt using retrieved examples.
    return f"""
Below are example patient cases and diagnoses.

{few_shot_context}

Now evaluate this patient.

Patient profile:
{patient_block(row)}

Allowed diagnosis labels:
{allowed_labels_text(label_space)}

Task:
1. Return the top 3 most likely diagnoses from the allowed labels only.
2. Rank them from most likely to least likely.
3. Give a short, cautious consultation guidance message.
4. Do not claim certainty.
5. Do NOT rely on explicit diagnosis mentions in the symptom list.
6. If a finding directly names a diagnosis or a suspected diagnosis, ignore that surface form and focus on the remaining clinical pattern.

Return valid JSON with this format:
{{
  "top_3": [
    {{"label": "<label1>", "confidence": 0.0}},
    {{"label": "<label2>", "confidence": 0.0}},
    {{"label": "<label3>", "confidence": 0.0}}
  ],
  "guidance": "<short cautious advice>"
}}
""".strip()


def prompt_knowledge_ranked(
    row,
    knowledge_snippets: list[str],
    label_space: list[str],
) -> str:
    # Prompt variant with EBM-derived knowledge snippets.
    knowledge_block = (
        "\n".join([f"- {snippet}" for snippet in knowledge_snippets])
        if knowledge_snippets
        else "None"
    )

    return f"""
You are assisting as the diagnostic reasoning core of a healthcare chatbot.

Relevant medical knowledge:
{knowledge_block}

Patient profile:
{patient_block(row)}

Allowed diagnosis labels:
{allowed_labels_text(label_space)}

Task:
1. Return the top 3 most likely diagnoses from the allowed labels only.
2. Rank them from most likely to least likely.
3. Give a short, cautious consultation guidance message.
4. Do not claim certainty.
5. Do NOT rely on explicit diagnosis mentions in the symptom list.
6. If a finding directly names a diagnosis or a suspected diagnosis, ignore that surface form and focus on the remaining clinical pattern.

Return valid JSON with this format:
{{
  "top_3": [
    {{"label": "<label1>", "confidence": 0.0}},
    {{"label": "<label2>", "confidence": 0.0}},
    {{"label": "<label3>", "confidence": 0.0}}
  ],
  "guidance": "<short cautious advice>"
}}
""".strip()
