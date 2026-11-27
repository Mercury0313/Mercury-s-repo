"""
Convert TSRBench + oracle scores into LLMRouter's standard benchmark format.

Produces the files consumed by LLMRouter's DataLoader (data_path section of a
router YAML), so any standard router (knnrouter, mlprouter, graphrouter, ...)
can be trained on TSRBench routing data. Each (modality, model) candidate is
exposed as one LLMRouter "model" (e.g. "text|qwen3-8b").

Outputs (under --out-dir, default data/tsrbench_data/):
  query_data_train.jsonl / query_data_test.jsonl
      {task_name, query, ground_truth, metric, choices, task_id, embedding_id}
  routing_data_train.jsonl / routing_data_test.jsonl
      query fields + {model_name, performance, input_tokens, output_tokens,
                      token_num, response, response_time, api_key_used,
                      user_id, fig_id}
  llm_data.json            candidate → metadata (feature, pricing, endpoint)
  llm_embeddings.json      candidate → embedding of its description
  query_embeddings.pt      FloatTensor [N_query, D], indexed by embedding_id

Usage (from the tsrbench/ directory, after building the oracle):
    python convert_to_llmrouter.py \
        --oracle ../../../TSRouter/data/oracle_full.csv \
        --token-counts ../../../TSRouter/data/token_counts.json \
        --model-descriptions ../../../TSRouter/configs/model_descriptions.json \
        --out-dir ../tsrbench_data

    # Skip the (GPU/slow) embedding step:
    python convert_to_llmrouter.py --oracle ... --skip-embeddings
"""

import argparse
import csv
import json
import os
import random

# Same 12 task files as the inference scripts
TASKS = {
    "perception":                  "perception",
    "abductive_reasoning":         "reasoning",
    "causal_reasoning":            "reasoning",
    "deductive_reasoning":         "reasoning",
    "etiological_reasoning":       "reasoning",
    "inductive_reasoning":         "reasoning",
    "numerical_reasoning":         "reasoning",
    "temporal_relation_reasoning": "reasoning",
    "event_prediction":            "prediction",
    "time_series_forecasting":     "prediction",
    "qualitative_decision":        "decision",
    "quantitative_decision":       "decision",
}

MODALITY_PREFIX = {
    "text": "[Text modality] Processing time series as numerical text. ",
    "visual": "[Visual modality] Processing time series as plotted chart images. ",
    "mix": "[Mix modality] Processing time series as both numerical text and chart images. ",
}


def ts_stat_description(ts_data):
    """Concise natural-language stats of the time series, appended to the query."""
    if not ts_data:
        return ""
    series_list = [ts_data] if not isinstance(ts_data[0], list) else ts_data
    vals = []
    for s in series_list:
        try:
            vals.extend(float(v) for v in s if v is not None)
        except (TypeError, ValueError):
            pass
    if not vals:
        return ""
    n = len(vals)
    mean_v = sum(vals) / n
    std_v = (sum((v - mean_v) ** 2 for v in vals) / n) ** 0.5
    xs = list(range(n))
    x_mean = (n - 1) / 2
    denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
    slope = sum((x - x_mean) * (v - mean_v) for x, v in zip(xs, vals)) / denom
    trend = "upward" if slope > 0 else "downward"
    ch = "channel" if len(series_list) == 1 else "channels"
    return (f"\n[Time series: {len(series_list)} {ch}, length {len(series_list[0])}. "
            f"Mean={mean_v:.2f}, std={std_v:.2f}, "
            f"range=[{min(vals):.2f}, {max(vals):.2f}], trend: {trend}.]")


def load_queries(datasets_root):
    """Load all TSRBench queries as LLMRouter query records, keyed by (task, line_idx)."""
    queries = {}
    for stem, category in TASKS.items():
        path = os.path.join(datasets_root, category, f"{stem}.jsonl")
        if not os.path.exists(path):
            print(f"  Missing task file, skipping: {path}")
            continue
        with open(path) as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if stem == "abductive_reasoning":
                    mcq = entry.get("multiple_choice_question", {})
                    question = mcq.get("question", "")
                    answer = mcq.get("answer", "")
                    choices = mcq.get("choices", [])
                    events = entry.get("context", {}).get("history_events", [])
                    if events:
                        question = "Events: " + "; ".join(events[-5:]) + "\n" + question
                    ts = []
                    for val in entry.get("numerical_time_series", {}).values():
                        if isinstance(val, dict) and "history" in val:
                            ts = val["history"]
                            break
                        if isinstance(val, list):
                            ts = val
                            break
                else:
                    question = entry.get("question", "")
                    answer = entry.get("answer", "")
                    choices = entry.get("choices", [])
                    ts = entry.get("timeseries", [])
                if not question:
                    continue

                queries[(stem, line_idx)] = {
                    "task_name": stem,
                    "query": question + ts_stat_description(ts),
                    "ground_truth": str(answer),
                    "metric": "em_mc",
                    "choices": str(choices) if choices else None,
                    "task_id": None,
                }
    return queries


def main():
    parser = argparse.ArgumentParser(description="Convert TSRBench + oracle to LLMRouter benchmark format")
    parser.add_argument("--datasets-root", default="datasets/TSRBench")
    parser.add_argument("--oracle", default="../../../TSRouter/data/oracle_full.csv")
    parser.add_argument("--token-counts", default="../../../TSRouter/data/token_counts.json")
    parser.add_argument("--model-descriptions", default="../../../TSRouter/configs/model_descriptions.json")
    parser.add_argument("--out-dir", default="../tsrbench_data")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Do not compute query/model embeddings (routers that need them will not run)")
    args = parser.parse_args()

    # 1. Queries
    queries = load_queries(args.datasets_root)
    print(f"Loaded {len(queries)} TSRBench queries")

    # 2. Oracle scores → routing rows grouped per query
    oracle = {}
    with open(args.oracle) as f:
        for row in csv.DictReader(f):
            key = (row["task_type"], int(row["line_idx"]))
            oracle.setdefault(key, {})[row["candidate"]] = float(row["score"])
    print(f"Loaded oracle scores for {len(oracle)} queries")

    token_counts = {}
    if os.path.exists(args.token_counts):
        token_counts = json.load(open(args.token_counts))
    else:
        print(f"  No token counts at {args.token_counts} — using fixed 500/200")

    # 3. Candidates → llm_data (adopted split only, one entry per modality-model pair)
    model_desc = json.load(open(args.model_descriptions))
    llm_data = {}
    for group, modalities in (("LLMs", ["text"]), ("VLMs", None)):
        for name, info in model_desc.get(group, {}).items():
            if info.get("split", "adopted") != "adopted":
                continue
            for modality in (modalities or info.get("modalities", ["visual", "mix"])):
                price_scale = 1.5 if modality in ("visual", "mix") else 1.0
                llm_data[f"{modality}|{name}"] = {
                    "size": None,
                    "feature": MODALITY_PREFIX[modality] + info["feature"],
                    "input_price": info["input_price"] * price_scale,
                    "output_price": info["output_price"],
                    "model": name,
                    "modality": modality,
                    "service": "vLLM",
                    "api_endpoint": "http://localhost:8000/v1",
                }
    print(f"Candidates: {sorted(llm_data)}")

    # 4. Keep only queries that have oracle coverage; assign embedding ids; split
    keys = sorted(k for k in queries if k in oracle)
    dropped = len(queries) - len(keys)
    if dropped:
        print(f"  Dropping {dropped} queries without oracle scores")
    for emb_id, key in enumerate(keys):
        queries[key]["embedding_id"] = emb_id

    rng = random.Random(args.seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * args.test_ratio)
    test_keys = set(shuffled[:n_test])

    def routing_rows(key):
        q = queries[key]
        stem, line_idx = key
        rows = []
        for cand, score in sorted(oracle[key].items()):
            if cand not in llm_data:
                continue  # OOD / retired candidates
            tc = token_counts.get(f"{cand}|{stem}|{line_idx}", [500, 200])
            rows.append({**q,
                         "model_name": cand,
                         "response": None,
                         "token_num": tc[0] + tc[1],
                         "input_tokens": tc[0],
                         "output_tokens": tc[1],
                         "response_time": None,
                         "api_key_used": None,
                         "performance": score,
                         "user_id": None,
                         "fig_id": None})
        return rows

    os.makedirs(args.out_dir, exist_ok=True)

    def dump_jsonl(path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  Wrote {len(records):6d} records → {path}")

    train_keys = [k for k in keys if k not in test_keys]
    dump_jsonl(os.path.join(args.out_dir, "query_data_train.jsonl"), [queries[k] for k in train_keys])
    dump_jsonl(os.path.join(args.out_dir, "query_data_test.jsonl"), [queries[k] for k in sorted(test_keys)])
    dump_jsonl(os.path.join(args.out_dir, "routing_data_train.jsonl"),
               [r for k in train_keys for r in routing_rows(k)])
    dump_jsonl(os.path.join(args.out_dir, "routing_data_test.jsonl"),
               [r for k in sorted(test_keys) for r in routing_rows(k)])

    with open(os.path.join(args.out_dir, "llm_data.json"), "w") as f:
        json.dump(llm_data, f, indent=2)
    print(f"  Wrote {len(llm_data)} candidates → llm_data.json")

    # 5. Embeddings (query texts + candidate descriptions)
    if args.skip_embeddings:
        print("Skipping embeddings (--skip-embeddings). Routers requiring "
              "query_embedding_data / llm_embedding_data will not run.")
        return

    import torch
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.embedding_model)

    texts = [queries[k]["query"] for k in keys]
    emb = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    torch.save(torch.tensor(emb, dtype=torch.float32), os.path.join(args.out_dir, "query_embeddings.pt"))
    print(f"  Wrote query embeddings {emb.shape} → query_embeddings.pt")

    cand_names = sorted(llm_data)
    cand_emb = encoder.encode([llm_data[c]["feature"] for c in cand_names], convert_to_numpy=True)
    with open(os.path.join(args.out_dir, "llm_embeddings.json"), "w") as f:
        json.dump({c: cand_emb[i].tolist() for i, c in enumerate(cand_names)}, f)
    print(f"  Wrote {len(cand_names)} candidate embeddings → llm_embeddings.json")


if __name__ == "__main__":
    main()
