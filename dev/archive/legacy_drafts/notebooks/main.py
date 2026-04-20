def safe_json_load(text):
    text = text.strip()

    # Strip optional markdown fences before JSON parsing.
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        return None

def parse_ranked_output(raw_output, label_space, top_k=3):
    parsed = safe_json_load(raw_output)

    default_output = {
        "top_labels": ["unknown"] * top_k,
        "top_confidences": [0.0] * top_k,
        "guidance": ""
    }

    if parsed is None:
        return default_output

    items = parsed.get("top_3", [])
    guidance = str(parsed.get("guidance", "")).strip()

    labels = []
    confidences = []

    for item in items[:top_k]:
        label = normalize_label_to_allowed(item.get("label", ""), label_space)
        conf = item.get("confidence", 0.0)

        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        labels.append(label)
        confidences.append(conf)

    while len(labels) < top_k:
        labels.append("unknown")
        confidences.append(0.0)

    return {
        "top_labels": labels,
        "top_confidences": confidences,
        "guidance": guidance
    }


def run_self_consistency_ranked(prompt, label_space, n=5, model=OPENAI_MODEL):
    vote_counter = Counter()
    guidance_texts = []

    for _ in range(n):
        raw = run_openai(prompt, model=model, temperature=0.7)
        parsed = parse_ranked_output(raw, label_space, top_k=3)

        for rank, label in enumerate(parsed["top_labels"]):
            if label != "unknown":
                vote_counter[label] += (3 - rank)  # weighted votes

        if parsed["guidance"]:
            guidance_texts.append(parsed["guidance"])

    ranked = vote_counter.most_common(3)

    total_votes = sum(vote_counter.values()) if sum(vote_counter.values()) > 0 else 1
    top_3 = []
    for label, count in ranked:
        top_3.append({
            "label": label,
            "confidence": round(count / total_votes, 4)
        })

    while len(top_3) < 3:
        top_3.append({"label": "unknown", "confidence": 0.0})

    guidance = guidance_texts[0] if guidance_texts else ""

    return {
        "top_3": top_3,
        "guidance": guidance
    }

def evaluate_prediction(gold_label, pred_top_labels):
    gold_norm = normalize_label_to_allowed(gold_label, LABEL_SPACE)

    top1_correct = int(len(pred_top_labels) > 0 and pred_top_labels[0] == gold_norm)
    top3_correct = int(gold_norm in pred_top_labels[:3])

    return {
        "gold_label": gold_norm,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct
    }

def summarize_results(results_df):
    return pd.DataFrame([{
        "n_cases": len(results_df),
        "top1_accuracy": round(results_df["top1_correct"].mean(), 4),
        "top3_accuracy": round(results_df["top3_correct"].mean(), 4),
        "unknown_top1_rate": round((results_df["pred_1"] == "unknown").mean(), 4)
    }])

def run_baseline_experiment(train_df, test_df, n_cases=40):
    results = []

    test_subset = test_df.head(n_cases).copy()

    for _, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
        few_shot_context = build_retrieved_few_shot_context(row, train_df, n=4)

        prompt = prompt_few_shot_ranked(row, few_shot_context)
        raw_output = run_openai(prompt, temperature=0)
        parsed = parse_ranked_output(raw_output, LABEL_SPACE, top_k=3)

        ev = evaluate_prediction(row["diagnosis"], parsed["top_labels"])

        results.append({
            "symptoms": row["symptoms"],
            "ground_truth": row["diagnosis"],
            "pred_1": parsed["top_labels"][0],
            "pred_2": parsed["top_labels"][1],
            "pred_3": parsed["top_labels"][2],
            "conf_1": parsed["top_confidences"][0],
            "conf_2": parsed["top_confidences"][1],
            "conf_3": parsed["top_confidences"][2],
            "guidance": parsed["guidance"],
            **ev
        })

        time.sleep(0.2)

    return pd.DataFrame(results)


baseline_results = run_baseline_experiment(train_df, test_df, n_cases=40)
display(baseline_results.head())
display(summarize_results(baseline_results))

def run_knowledge_experiment(train_df, test_df, knowledge_df, n_cases=40):
    results = []

    test_subset = test_df.head(n_cases).copy()

    for _, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
        snippets = retrieve_knowledge(row["symptoms"], knowledge_df, top_k=2)

        prompt = prompt_knowledge_ranked(row, snippets)
        raw_output = run_openai(prompt, temperature=0)
        parsed = parse_ranked_output(raw_output, LABEL_SPACE, top_k=3)

        ev = evaluate_prediction(row["diagnosis"], parsed["top_labels"])

        results.append({
            "symptoms": row["symptoms"],
            "ground_truth": row["diagnosis"],
            "knowledge_used": " | ".join(snippets),
            "pred_1": parsed["top_labels"][0],
            "pred_2": parsed["top_labels"][1],
            "pred_3": parsed["top_labels"][2],
            "conf_1": parsed["top_confidences"][0],
            "conf_2": parsed["top_confidences"][1],
            "conf_3": parsed["top_confidences"][2],
            "guidance": parsed["guidance"],
            **ev
        })

        time.sleep(0.2)

    return pd.DataFrame(results)

knowledge_results = run_knowledge_experiment(train_df, test_df, knowledge_df, n_cases=40)
display(knowledge_results.head())
display(summarize_results(knowledge_results))

def run_self_consistency_experiment(test_df, knowledge_df=None, n_cases=40):
    results = []

    test_subset = test_df.head(n_cases).copy()

    for _, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
        if knowledge_df is not None:
            snippets = retrieve_knowledge(row["symptoms"], knowledge_df, top_k=2)
            prompt = prompt_knowledge_ranked(row, snippets)
            knowledge_used = " | ".join(snippets)
        else:
            prompt = prompt_zero_shot_ranked(row)
            knowledge_used = ""

        sc_output = run_self_consistency_ranked(prompt, LABEL_SPACE, n=5, model=OPENAI_MODEL)

        pred_labels = [x["label"] for x in sc_output["top_3"]]
        pred_confs = [x["confidence"] for x in sc_output["top_3"]]

        ev = evaluate_prediction(row["diagnosis"], pred_labels)

        results.append({
            "symptoms": row["symptoms"],
            "ground_truth": row["diagnosis"],
            "knowledge_used": knowledge_used,
            "pred_1": pred_labels[0],
            "pred_2": pred_labels[1],
            "pred_3": pred_labels[2],
            "conf_1": pred_confs[0],
            "conf_2": pred_confs[1],
            "conf_3": pred_confs[2],
            "guidance": sc_output["guidance"],
            **ev
        })

        time.sleep(0.2)

    return pd.DataFrame(results)

sc_results = run_self_consistency_experiment(test_df, knowledge_df=knowledge_df, n_cases=40)
display(sc_results.head())
display(summarize_results(sc_results))

summary_rows = []

for method_name, result_df in [
    ("baseline_few_shot", baseline_results),
    ("knowledge_prompt", knowledge_results),
    ("self_consistency_knowledge", sc_results),
]:
    summary_rows.append({
        "method": method_name,
        "n_cases": len(result_df),
        "top1_accuracy": round(result_df["top1_correct"].mean(), 4),
        "top3_accuracy": round(result_df["top3_correct"].mean(), 4),
        "unknown_top1_rate": round((result_df["pred_1"] == "unknown").mean(), 4),
    })

summary_df = pd.DataFrame(summary_rows)
display(summary_df)

def show_failures(results_df, n=10):
    failures = results_df[results_df["top1_correct"] == 0].copy()
    display(failures.head(n)[[
        "symptoms", "ground_truth", "pred_1", "pred_2", "pred_3",
        "conf_1", "conf_2", "conf_3", "guidance"
    ]])

show_failures(sc_results, n=10)

baseline_results.to_csv(os.path.join(OUTPUT_DIR, "baseline_results.csv"), index=False)
knowledge_results.to_csv(os.path.join(OUTPUT_DIR, "knowledge_results.csv"), index=False)
sc_results.to_csv(os.path.join(OUTPUT_DIR, "self_consistency_results.csv"), index=False)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "method_summary.csv"), index=False)

print("Saved outputs.")

import os

def load_ebm_documents(doc_dir, max_docs=100):
    docs = []
    for fname in os.listdir(doc_dir):
        if fname.endswith(".txt"):
            path = os.path.join(doc_dir, fname)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                docs.append({
                    "doc_id": fname,
                    "text": text[:3000]
                })
        if len(docs) >= max_docs:
            break
    return pd.DataFrame(docs)

ebm_docs_df = load_ebm_documents(os.path.join(EBM_DIR, "ebm_nlp_2_00", "documents"), max_docs=100)
display(ebm_docs_df.head())

def make_simple_snippet(text, max_len=220):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

ebm_docs_df["snippet"] = ebm_docs_df["text"].apply(make_simple_snippet)
ebm_docs_df[["doc_id", "snippet"]].to_csv(os.path.join(EBM_DIR, "knowledge_snippets_from_ebm.csv"), index=False)

