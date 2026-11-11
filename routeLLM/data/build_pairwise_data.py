"""Convert scored routing samples into MF pairwise training labels."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "derived" / "my_routing.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "derived" / "pairwise_data.jsonl",
    )
    parser.add_argument(
        "--model-a",
        default="qwen3_1p7b_local",
        help="Small model. Quality ties are assigned to model_a.",
    )
    parser.add_argument("--model-b", default="deepseek_flash_api", help="Large model.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_pairwise_rows(
    source_rows: list[dict[str, Any]], model_a: str, model_b: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output_rows = []
    outcomes: Counter[str] = Counter()

    for source_row in source_rows:
        candidates = {
            candidate["candidate_id"]: candidate
            for candidate in source_row["candidates"]
        }
        if model_a not in candidates or model_b not in candidates:
            raise ValueError(
                f"{source_row['request_id']} is missing {model_a} or {model_b}"
            )

        quality_a = float(candidates[model_a]["quality_score"])
        quality_b = float(candidates[model_b]["quality_score"])
        if quality_a == quality_b:
            winner = "model_a"
            outcomes["quality_tie_assigned_to_model_a"] += 1
        else:
            winner = "model_a" if quality_a > quality_b else "model_b"
            outcomes[winner] += 1
        output_rows.append(
            {
                "idx": len(output_rows),
                "request_id": source_row["request_id"],
                "source_dataset": source_row["source_dataset"],
                "prompt": source_row["prompt"],
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner,
                "quality_score_a": quality_a,
                "quality_score_b": quality_b,
            }
        )

    return output_rows, outcomes


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.input)
    pairwise_rows, outcomes = build_pairwise_rows(
        source_rows, args.model_a, args.model_b
    )
    write_jsonl(args.output, pairwise_rows)
    print(
        json.dumps(
            {
                "input_rows": len(source_rows),
                "pairwise_rows": len(pairwise_rows),
                "outcomes": outcomes,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
