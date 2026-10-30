"""Collect paired Qwen and DeepSeek responses for routing training."""

import argparse
import hashlib
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.shared.reasoning_effort import ReasoningEffort


DATA_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, default=DATA_DIR / "requests.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "executions.jsonl")
    parser.add_argument("--models-config", type=Path, default=DATA_DIR.parent / "configs" / "models.yaml")
    parser.add_argument("--prompts-config", type=Path, default=DATA_DIR.parent / "configs" / "prompts.yaml")
    parser.add_argument("--prices-config", type=Path, default=DATA_DIR.parent / "configs" / "prices.yaml")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_yaml(path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_existing_successes(path):
    if not path.exists():
        return set()
    return {
        (row["request_id"], row["candidate_id"])
        for row in load_jsonl(path)
        if row.get("status") == "success"
    }


def choose_requests(records, limit, seed):
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    if limit > len(records):
        raise ValueError(f"Requested {limit} records, but only {len(records)} are available")
    return random.Random(seed).sample(records, limit)


def make_client(candidate):
    api_key = os.environ.get(candidate.get("api_key_env", "")) or "ollama"
    return OpenAI(api_key=api_key, base_url=candidate["base_url"])


def get_usage(response):
    usage = response.usage
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "prompt_tokens_details": prompt_details.model_dump() if prompt_details is not None else None,
    }


def is_peak(when):
    if when.weekday() >= 5:
        return False
    return (1 <= when.hour < 4) or (6 <= when.hour < 10)


def calculate_cost(candidate_id, usage, prices, started_at):
    price = prices["candidates"][candidate_id]
    if price.get("api_cost_only"):
        return 0.0, "local"

    band = "peak" if is_peak(started_at) else "off_peak"
    prompt_tokens = usage.get("prompt_tokens") or 0
    details = usage.get("prompt_tokens_details") or {}
    cache_hit_tokens = details.get("cached_tokens", 0)
    cache_miss_tokens = max(prompt_tokens - cache_hit_tokens, 0)
    completion_tokens = usage.get("completion_tokens") or 0
    rates = price[band]
    cost = (
        cache_hit_tokens * rates["input_cache_hit_per_million_tokens"]
        + cache_miss_tokens * rates["input_cache_miss_per_million_tokens"]
        + completion_tokens * rates["output_per_million_tokens"]
    ) / 1_000_000
    return cost, band


def collect_one(record, candidate, system_prompt, prompt_version, prices, run_id, dry_run):
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": record["prompt"]},
    ]
    base = {
        "schema_version": "0.1",
        "execution_id": str(uuid.uuid4()),
        "run_id": run_id,
        "request_id": record["request_id"],
        "candidate_id": candidate["candidate_id"],
        "model_requested": candidate["model"],
        "prompt_version": prompt_version,
        "messages_sha256": hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
    }
    if dry_run:
        return {**base, "status": "dry_run"}

    try:
        request: dict[str, Any] = candidate["request"]
        if request["stream"] is not False:
            raise ValueError("This collector requires request.stream: false")
        completion_args: CompletionCreateParamsNonStreaming = {
            "model": cast(str, candidate["model"]),
            "messages": messages,
            "temperature": cast(float, request["temperature"]),
            "top_p": cast(float, request["top_p"]),
            "max_tokens": cast(int, request["max_tokens"]),
            "stream": False,
        }
        if "reasoning_effort" in request:
            completion_args["reasoning_effort"] = cast(
                ReasoningEffort, request["reasoning_effort"]
            )
        extra_body = request.get("extra_body")
        if extra_body is None:
            response = make_client(candidate).chat.completions.create(**completion_args)
        else:
            response = make_client(candidate).chat.completions.create(
                **completion_args, extra_body=extra_body
            )
        usage = get_usage(response)
        cost, price_band = calculate_cost(candidate["candidate_id"], usage, prices, started_at)
        return {
            **base,
            "status": "success",
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "latency_ms": round((time.perf_counter() - started_clock) * 1000),
            "model_returned": response.model,
            "response_text": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "usage": usage,
            "cost_usd": cost,
            "price_band": price_band,
        }
    except Exception as error:
        return {
            **base,
            "status": "error",
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def main():
    args = parse_args()
    models = load_yaml(args.models_config)
    prompts = load_yaml(args.prompts_config)
    prices = load_yaml(args.prices_config)
    candidates = [models["candidates"][role] for role in ("small", "large")]
    records = choose_requests(load_jsonl(args.requests), args.limit, args.seed)
    existing = get_existing_successes(args.output)
    run_id = args.run_id or f"paired_{args.seed}_{args.limit}"
    rows = []
    for record in records:
        system_prompt = prompts["tasks"][record["source_dataset"]]["system"]
        for candidate in candidates:
            pair = (record["request_id"], candidate["candidate_id"])
            if pair not in existing:
                rows.append(
                    collect_one(
                        record,
                        candidate,
                        system_prompt,
                        prompts["prompt_version"],
                        prices,
                        run_id,
                        args.dry_run,
                    )
                )

    if not args.dry_run:
        with args.output.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"run_id": run_id, "selected_requests": [r["request_id"] for r in records], "records_written": len(rows), "dry_run": args.dry_run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
