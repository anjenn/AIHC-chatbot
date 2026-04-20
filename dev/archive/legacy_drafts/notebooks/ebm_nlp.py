knowledge_df = pd.read_csv(os.path.join(EBM_DIR, "knowledge_snippets.csv"))
display(knowledge_df.head())

def retrieve_knowledge(symptoms, knowledge_df, top_k=2):
    symptom_text = normalize_symptom_text(symptoms)
    scores = []

    for _, row in knowledge_df.iterrows():
        snippet = str(row["snippet"])
        tokens = set(normalize_symptom_text(snippet).split())
        symptom_tokens = set(symptom_text.split())
        score = len(tokens & symptom_tokens)
        scores.append((snippet, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [snippet for snippet, score in scores[:top_k] if score > 0]