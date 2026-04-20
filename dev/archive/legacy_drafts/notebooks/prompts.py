def prompt_zero_shot_ranked(row):
    return f"""
You are assisting as the diagnostic reasoning core of a healthcare chatbot.

Patient profile:
{patient_block(row)}

Allowed diagnosis labels:
{allowed_labels_text()}

Task:
1. Return the top 3 most likely diagnoses from the allowed labels only.
2. Rank them from most likely to least likely.
3. Give a short, cautious consultation guidance message.
4. Do not claim certainty.

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

def prompt_few_shot_ranked(row, few_shot_context):
    return f"""
Below are example patient cases and diagnoses.

{few_shot_context}

Now evaluate this patient.

Patient profile:
{patient_block(row)}

Allowed diagnosis labels:
{allowed_labels_text()}

Task:
1. Return the top 3 most likely diagnoses from the allowed labels only.
2. Rank them from most likely to least likely.
3. Give a short, cautious consultation guidance message.
4. Do not claim certainty.

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

def prompt_knowledge_ranked(row, knowledge_snippets):
    knowledge_block = "\n".join([f"- {k}" for k in knowledge_snippets]) if knowledge_snippets else "None"

    return f"""
You are assisting as the diagnostic reasoning core of a healthcare chatbot.

Relevant medical knowledge:
{knowledge_block}

Patient profile:
{patient_block(row)}

Allowed diagnosis labels:
{allowed_labels_text()}

Task:
1. Return the top 3 most likely diagnoses from the allowed labels only.
2. Rank them from most likely to least likely.
3. Give a short, cautious consultation guidance message.
4. Do not claim certainty.

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