"""Score paired execution logs and build quality-cost routing examples."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).parent
VENDOR_DIR = DATA_DIR / "vendor"
# IFEval itself is vendored as pure Python.  Do not put vendor/python ahead of
# the active environment: it can contain binary wheels (e.g. regex) compiled
# for another Python/Conda version.
sys.path.insert(0, str(VENDOR_DIR))
# Fall back to the vendored pure-Python dependencies only when the active
# environment does not provide them.  Appending preserves the active regex.
sys.path.append(str(VENDOR_DIR / "python"))


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--requests", type=Path, default=DATA_DIR / "requests.jsonl")
#     parser.add_argument("--executions", type=Path, default=DATA_DIR / "executions.jsonl")
#     parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "derived")
#     return parser.parse_args()
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DATA_DIR / "requests.jsonl")
    parser.add_argument("--executions", type=Path, default=DATA_DIR / "executions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "derived")
    parser.add_argument(
        "--scored-name",
        default="scored_executions_smoketest.jsonl",
        help="输出文件名（仅文件名，不含目录），默认 scored_executions_smoketest.jsonl",
    )
    parser.add_argument(
        "--routing-name",
        default="routing_samples_smoketest.jsonl",
        help="输出文件名（仅文件名，不含目录），默认 routing_samples_smoketest.jsonl",
    )
    return parser.parse_args()

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_choice(response: str) -> str | None:
    matches = re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", response.upper())
    return matches[-1] if matches else None


def extract_gsm8k_answer(response: str) -> str | None:
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)|([-+]?\d[\d,]*(?:\.\d+)?)", response)
    if not matches:
        return None
    return next(value for pair in reversed(matches) for value in pair if value).replace(",", "")


def score_ifeval(request: dict[str, Any], response: str) -> tuple[float, list[bool]]:
    import nltk
    from instruction_following_eval import evaluation_lib

    project_nltk_data = DATA_DIR / "nltk_data"
    if project_nltk_data.is_dir():
        nltk.data.path.insert(0, str(project_nltk_data))

    metadata = request["metadata"]
    example = evaluation_lib.InputExample(
        key=int(request["source_id"]),
        instruction_id_list=metadata["instruction_id_list"],
        prompt=request["origin_query"],
        kwargs=metadata["kwargs"],
    )
    result = evaluation_lib.test_instruction_following_strict(
        example, {example.prompt: response}
    )
    return float(result.follow_all_instructions), result.follow_instruction_list


def score_execution(request: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    response = execution["response_text"]
    scorer_id = request["scorer_id"]
    details: dict[str, Any] = {"scorer_id": scorer_id}
    if scorer_id == "ceval_choice_exact_match_v1":
        prediction = extract_choice(response)
        details["prediction"] = prediction
        details["quality_score"] = float(prediction == request["ground_truth"])
    elif scorer_id == "gsm8k_final_answer_v1":
        prediction = extract_gsm8k_answer(response)
        details["prediction"] = prediction
        details["quality_score"] = float(prediction == request["ground_truth"])
    elif scorer_id == "ifeval_instruction_following_v1":
        score, checks = score_ifeval(request, response)
        details["prediction"] = None
        details["quality_score"] = score
        details["instruction_checks"] = checks
    else:
        raise ValueError(f"Unsupported scorer_id: {scorer_id}")
    return details


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    requests = {row["request_id"]: row for row in load_jsonl(args.requests)}
    executions = [row for row in load_jsonl(args.executions) if row["status"] == "success"]
    scored = []
    for execution in executions:
        request = requests.get(execution["request_id"])
        if request is None:
            raise ValueError(f"Missing request for {execution['request_id']}")
        scored.append({**execution, "score": score_execution(request, execution)})

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[row["request_id"]].append(row)
    routing_rows = []
    incomplete_request_ids = []
    for request_id, pair in sorted(grouped.items()):
        if len(pair) < 2:
            incomplete_request_ids.append(request_id)
            continue
        if len(pair) > 2:
            raise ValueError(f"Expected two successful executions for {request_id}, found {len(pair)}")
        if len({row["candidate_id"] for row in pair}) != 2:
            raise ValueError(f"Expected two different candidates for {request_id}")
        routing_rows.append(
            {
                "schema_version": "0.1",
                "request_id": request_id,
                "source_dataset": requests[request_id]["source_dataset"],
                "prompt": requests[request_id]["prompt"],
                "candidates": [
                    {
                        "candidate_id": row["candidate_id"],
                        "quality_score": row["score"]["quality_score"],
                        "cost_usd": row["cost_usd"],
                        "latency_ms": row["latency_ms"],
                        "execution_id": row["execution_id"],
                    }
                    for row in pair
                ],
            }
        )

    # args.output_dir.mkdir(parents=True, exist_ok=True)
    # write_jsonl(args.output_dir / "scored_executions_smoketest.jsonl", scored)
    # write_jsonl(args.output_dir / "routing_samples_smoketest.jsonl", routing_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / args.scored_name, scored)
    write_jsonl(args.output_dir / args.routing_name, routing_rows)
    print(json.dumps({
        "scored_executions": len(scored),
        "routing_samples": len(routing_rows),
        "skipped_incomplete_requests": len(incomplete_request_ids),
        "incomplete_request_ids": incomplete_request_ids,
    }))


if __name__ == "__main__":
    main()
