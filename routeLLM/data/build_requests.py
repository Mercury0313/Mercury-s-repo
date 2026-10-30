"""Build a reproducible routing-request corpus from GSM8K, IFEval, and C-Eval."""

import argparse
import hashlib
import json
import random
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
IFEVAL_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/master/"
    "instruction_following_eval/data/input_data.jsonl"
)
CEVAL_DATASET = "ceval/ceval-exam"
CEVAL_ROWS_URL = "https://datasets-server.huggingface.co/rows"
SUBJECT_MAP_URL = "https://raw.githubusercontent.com/hkust-nlp/ceval/main/subject_mapping.json"


def fetch_bytes(url):
    request = urllib.request.Request(url, headers={"User-Agent": "routeLLM-dataset-builder/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def load_jsonl(url):
    return [json.loads(line) for line in fetch_bytes(url).decode("utf-8").splitlines() if line.strip()]


def build_gsm8k(rng, count):
    rows = load_jsonl(GSM8K_URL)
    if count > len(rows):
        raise ValueError(f"Requested {count} GSM8K rows, but only {len(rows)} are available")
    records = []
    for index, row in sorted(rng.sample(list(enumerate(rows)), count)):
        records.append(
            {
                "schema_version": "0.1",
                "request_id": f"gsm8k_test_{index:04d}",
                "source_dataset": "gsm8k",
                "source_split": "test",
                "source_id": str(index),
                "task_type": "math_reasoning",
                "origin_query": row["question"],
                "prompt": "Solve the following math problem. Give the final answer.\n\n" + row["question"],
                "ground_truth": row["answer"].rsplit("####", 1)[-1].strip(),
                "scorer_id": "gsm8k_final_answer_v1",
                "metadata": {"source_url": GSM8K_URL},
            }
        )
    return records


def build_ifeval(count):
    rows = load_jsonl(IFEVAL_URL)
    if count > len(rows):
        raise ValueError(f"Requested {count} IFEval rows, but only {len(rows)} are available")
    records = []
    for index, row in enumerate(rows[:count]):
        records.append(
            {
                "schema_version": "0.1",
                "request_id": f"ifeval_{index:04d}",
                "source_dataset": "ifeval",
                "source_split": "full",
                "source_id": str(row.get("key", index)),
                "task_type": "instruction_following",
                "origin_query": row["prompt"],
                "prompt": row["prompt"],
                "ground_truth": None,
                "scorer_id": "ifeval_instruction_following_v1",
                "metadata": {
                    "instruction_id_list": row.get("instruction_id_list", []),
                    "kwargs": row.get("kwargs", []),
                    "source_url": IFEVAL_URL,
                },
            }
        )
    return records


def fetch_ceval_subject(subject, subject_info):
    query = urllib.parse.urlencode(
        {"dataset": CEVAL_DATASET, "config": subject, "split": "val", "offset": 0, "length": 100}
    )
    try:
        payload = json.loads(fetch_bytes(CEVAL_ROWS_URL + "?" + query).decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Could not download C-Eval validation rows for {subject}") from error
    return [(subject, subject_info, item["row"]) for item in payload["rows"]]


def build_ceval(rng, count, subjects_per_category):
    subjects = json.loads(fetch_bytes(SUBJECT_MAP_URL).decode("utf-8"))
    by_category = {}
    for subject, subject_info in subjects.items():
        by_category.setdefault(subject_info[2], []).append((subject, subject_info))

    selected_subjects = []
    for category in sorted(by_category):
        candidates = sorted(by_category[category])
        if subjects_per_category > len(candidates):
            raise ValueError(f"Only {len(candidates)} C-Eval subjects are available in {category}")
        selected_subjects.extend(rng.sample(candidates, subjects_per_category))

    with ThreadPoolExecutor(max_workers=len(selected_subjects)) as executor:
        rows = [
            row
            for subject_rows in executor.map(lambda item: fetch_ceval_subject(*item), selected_subjects)
            for row in subject_rows
        ]
    if count > len(rows):
        raise ValueError(f"Requested {count} C-Eval rows, but the selected subjects provide only {len(rows)}")

    records = []
    for subject, subject_info, row in rng.sample(rows, count):
        subject_en, subject_zh, category = subject_info
        question = row["question"]
        records.append(
            {
                "schema_version": "0.1",
                "request_id": f"ceval_val_{subject}_{row['id']}",
                "source_dataset": "ceval",
                "source_split": "val",
                "source_id": f"{subject}:{row['id']}",
                "task_type": "multiple_choice",
                "origin_query": question,
                "prompt": (
                    f"以下是中国关于{subject_zh}考试的单项选择题，请选出其中的正确答案。\n\n"
                    f"{question}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n答案："
                ),
                "ground_truth": row["answer"],
                "scorer_id": "ceval_choice_exact_match_v1",
                "metadata": {
                    "subject": subject,
                    "subject_en": subject_en,
                    "subject_zh": subject_zh,
                    "category": category,
                    "choices": {key: row[key] for key in "ABCD"},
                    "source_url": f"https://huggingface.co/datasets/{CEVAL_DATASET}",
                },
            }
        )
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("requests.jsonl"))
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--gsm8k-count", type=int, default=400)
    parser.add_argument("--ifeval-count", type=int, default=541)
    parser.add_argument("--ceval-count", type=int, default=300)
    parser.add_argument("--ceval-subjects-per-category", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; rerun with --overwrite")
    rng = random.Random(args.seed)
    records = (
        build_gsm8k(rng, args.gsm8k_count)
        + build_ifeval(args.ifeval_count)
        + build_ceval(rng, args.ceval_count, args.ceval_subjects_per_category)
    )
    ids = [record["request_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request_id generated")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=args.output.parent, newline="\n") as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "total": len(records),
                "counts": dict(Counter(row["source_dataset"] for row in records)),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
