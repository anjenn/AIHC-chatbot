def patient_block(row):
    return (
        f"Age: {int(row['age'])}\n"
        f"Sex: {row['sex']}\n"
        f"Reported findings: {row['symptoms']}"
    )

def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_symptom_text(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9,\s\-()]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def symptom_token_set(text):
    parts = [p.strip() for p in normalize_symptom_text(text).split(",") if p.strip()]
    return set(parts)

def retrieve_similar_examples(row, train_df, n=4):
    row_tokens = symptom_token_set(row["symptoms"])
    scored_rows = []

    for idx, train_row in train_df.iterrows():
        train_tokens = symptom_token_set(train_row["symptoms"])
        overlap = len(row_tokens & train_tokens) / max(1, len(row_tokens | train_tokens))
        scored_rows.append((idx, overlap))

    scored_rows = sorted(scored_rows, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in scored_rows[:n]]
    return train_df.loc[top_indices].reset_index(drop=True)

def build_retrieved_few_shot_context(row, train_df, n=4):
    sample = retrieve_similar_examples(row, train_df, n=n)
    chunks = []
    for i, (_, r) in enumerate(sample.iterrows(), start=1):
        chunks.append(
            f"Example {i}:\n"
            f"Age: {int(r['age'])}\n"
            f"Sex: {r['sex']}\n"
            f"Reported findings: {r['symptoms']}\n"
            f"Diagnosis: {r['diagnosis']}"
        )
    return "\n\n".join(chunks)

def allowed_labels_text():
    return ", ".join(LABEL_SPACE)

def normalize_label_to_allowed(text, label_space):
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