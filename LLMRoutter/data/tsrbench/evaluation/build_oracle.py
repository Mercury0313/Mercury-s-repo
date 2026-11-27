"""
Build the oracle files TSRouter trains on from candidate inference outputs.

Walks the inference results produced by the inference/ scripts
(evaluation/results/<modality_dir>/<dataset>_<model>/generated_answer.json),
scores every generated answer against the TSRBench ground truth, and writes:

  data/oracle_full.csv    Columns: task_type, file, line_idx, candidate, modality, score
  data/token_counts.json  { "modality|model|task_type|line_idx": [input_tokens, output_tokens] }

Usage (from the repository root, after running the inference scripts):
    python evaluation/build_oracle.py
    python evaluation/build_oracle.py --results-dir evaluation/results --datasets-root datasets/TSRBench
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_MCQ, read_jsonl_file

# Dataset stem → TSRBench category subdirectory
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

# Served model identifier → candidate short name (extend when adding models)
MODEL_NAMES = {
    "Qwen/Qwen3-8B":                    "qwen3-8b",
    "Qwen/Qwen3-32B":                   "qwen3-32b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b-instruct-turbo",
    "Qwen/Qwen3-VL-8B-Instruct":        "qwen3-vl-8b-instruct",
    "Qwen/Qwen3-VL-32B-Instruct":       "qwen3-vl-32b-instruct",
    "zai-org/GLM-4.5V":                 "glm-4.5v",
}


def modality_of(dir_name):
    """Map a results modality directory to an oracle modality."""
    if dir_name.startswith("text"):
        return "text"
    if dir_name.startswith("vision"):
        return "visual"
    if dir_name.startswith("multimodal"):
        return "mix"
    return None


def parse_exp(exp):
    """Split '<dataset>_<model>' into (dataset_stem, model_id), or None."""
    for stem in sorted(TASKS, key=len, reverse=True):
        if exp.startswith(stem + "_"):
            return stem, exp[len(stem) + 1:]
    return None


def load_answers(datasets_root, stem):
    """Return the ground-truth answer per line of a TSRBench task file."""
    path = os.path.join(datasets_root, TASKS[stem], f"{stem}.jsonl")
    dataset = read_jsonl_file(path)
    answers = []
    for entry in dataset:
        if stem == "abductive_reasoning":
            answers.append(entry.get("multiple_choice_question", {}).get("answer", ""))
        else:
            answers.append(entry.get("answer", ""))
    return answers


def main():
    parser = argparse.ArgumentParser(description="Build oracle_full.csv and token_counts.json from inference results")
    parser.add_argument("--results-dir", default="evaluation/results", help="Root of inference outputs")
    parser.add_argument("--datasets-root", default="datasets/TSRBench", help="TSRBench dataset root")
    parser.add_argument("--oracle-out", default="data/oracle_full.csv")
    parser.add_argument("--token-counts-out", default="data/token_counts.json")
    args = parser.parse_args()

    answer_cache = {}
    rows = {}          # (task_type, line_idx, candidate) -> row
    token_counts = {}

    n_files = 0
    for dirpath, _dirnames, filenames in os.walk(args.results_dir):
        if "generated_answer.json" not in filenames:
            continue
        rel = os.path.relpath(dirpath, args.results_dir)
        parts = rel.split(os.sep)
        modality = modality_of(parts[0])
        parsed = parse_exp("/".join([parts[1]] + parts[2:]) if len(parts) > 1 else "")
        if modality is None or parsed is None:
            print(f"  Skipping unrecognized results dir: {rel}")
            continue
        stem, model_id = parsed
        model = MODEL_NAMES.get(model_id, os.path.basename(model_id).lower())
        candidate = f"{modality}|{model}"

        if stem not in answer_cache:
            answer_cache[stem] = load_answers(args.datasets_root, stem)
        answers = answer_cache[stem]

        entries = json.load(open(os.path.join(dirpath, "generated_answer.json")))
        n_scored = 0
        for entry in entries:
            idx = entry.get("idx")
            response = entry.get("response")
            if idx is None or response is None or idx >= len(answers):
                continue
            try:
                score = evaluate_MCQ(answers[idx], response)
            except Exception:
                score = None
            if score is None:
                continue
            rows[(stem, idx, candidate)] = {
                "task_type": stem,
                "file": f"{stem}.jsonl",
                "line_idx": idx,
                "candidate": candidate,
                "modality": modality,
                "score": float(score),
            }
            if "num_tokens" in entry:
                token_counts[f"{candidate}|{stem}|{idx}"] = [
                    int(entry["num_tokens"]),
                    len(response) // 4,
                ]
            n_scored += 1
        n_files += 1
        print(f"  {rel}: scored {n_scored}/{len(entries)} → {candidate}")

    if not rows:
        print(f"No results found under {args.results_dir} — run the inference/ scripts first.")
        return

    os.makedirs(os.path.dirname(args.oracle_out) or ".", exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r["task_type"], r["candidate"], r["line_idx"]))
    with open(args.oracle_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "file", "line_idx", "candidate", "modality", "score"])
        writer.writeheader()
        writer.writerows(ordered)
    with open(args.token_counts_out, "w") as f:
        json.dump(token_counts, f)

    candidates = sorted({r["candidate"] for r in rows.values()})
    print(f"\nProcessed {n_files} result files.")
    print(f"Wrote {len(ordered)} rows to {args.oracle_out} ({len(candidates)} candidates):")
    for c in candidates:
        print(f"  {c}")
    print(f"Wrote {len(token_counts)} token-count entries to {args.token_counts_out}")


if __name__ == "__main__":
    main()
