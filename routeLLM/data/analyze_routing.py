"""Summarize quality, latency, and cost trade-offs in routing samples."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).parent
METRICS = ("quality_score", "latency_ms", "cost_usd")


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
        default=DATA_DIR / "derived" / "routing_analysis.json",
    )
    parser.add_argument("--small-candidate", default="qwen3_1p7b_local")
    parser.add_argument("--large-candidate", default="deepseek_flash_api")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def pair_candidates(
    rows: list[dict[str, Any]], small_id: str, large_id: str
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    pairs = []
    for row in rows:
        candidates = {candidate["candidate_id"]: candidate for candidate in row["candidates"]}
        if set(candidates) != {small_id, large_id}:
            raise ValueError(
                f"{row['request_id']} does not contain exactly {small_id} and {large_id}"
            )
        for candidate in candidates.values():
            for metric in METRICS:
                value = candidate.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {metric} for {row['request_id']}")
        pairs.append((row["source_dataset"], candidates[small_id], candidates[large_id]))
    return pairs


def summarize_pairs(
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]],
    small_id: str,
    large_id: str,
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {
        small_id: defaultdict(list), large_id: defaultdict(list)
    }
    quality_delta = []
    latency_delta = []
    cost_delta = []

    for _, small, large in pairs:
        for candidate_id, candidate in ((small_id, small), (large_id, large)):
            for metric in METRICS:
                values[candidate_id][metric].append(float(candidate[metric]))
        quality_delta.append(float(large["quality_score"] - small["quality_score"]))
        latency_delta.append(float(large["latency_ms"] - small["latency_ms"]))
        cost_delta.append(float(large["cost_usd"] - small["cost_usd"]))

    return {
        "num_pairs": len(pairs),
        "per_candidate": {
            candidate_id: {metric: describe(values[candidate_id][metric]) for metric in METRICS}
            for candidate_id in (small_id, large_id)
        },
        "paired_comparison": {
            "quality_large_minus_small": describe(quality_delta),
            "large_quality_wins": sum(delta > 0 for delta in quality_delta),
            "small_quality_wins": sum(delta < 0 for delta in quality_delta),
            "quality_ties": sum(delta == 0 for delta in quality_delta),
            "latency_large_minus_small_ms": describe(latency_delta),
            "large_faster_count": sum(delta < 0 for delta in latency_delta),
            "cost_large_minus_small_usd": describe(cost_delta),
            "small_cheaper_count": sum(delta > 0 for delta in cost_delta),
        },
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if not rows:
        raise ValueError(f"No samples in {args.input}")
    pairs = pair_candidates(rows, args.small_candidate, args.large_candidate)
    by_dataset: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_dataset[pair[0]].append(pair)

    report = {
        "input": str(args.input),
        "small_candidate": args.small_candidate,
        "large_candidate": args.large_candidate,
        "datasets": dict(Counter(dataset for dataset, _, _ in pairs)),
        "overall": summarize_pairs(pairs, args.small_candidate, args.large_candidate),
        "by_dataset": {
            dataset: summarize_pairs(dataset_pairs, args.small_candidate, args.large_candidate)
            for dataset, dataset_pairs in sorted(by_dataset.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison = report["overall"]["paired_comparison"]
    print(f"pairs: {report['overall']['num_pairs']}")
    print(f"large quality wins / ties / small wins: {comparison['large_quality_wins']} / "
          f"{comparison['quality_ties']} / {comparison['small_quality_wins']}")
    print(f"analysis written to: {args.output}")


if __name__ == "__main__":
    main()
