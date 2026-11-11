"""Generate prompt embeddings for MF pairwise training data."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


DATA_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "derived" / "pairwise_data.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "derived" / "prompt_embeddings.npy",
    )
    parser.add_argument("--base-url", default=os.getenv("EMBEDDING_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("EMBEDDING_API_KEY"))
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-dim", type=int, default=1536)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("No pairwise rows found")

    indices = [row["idx"] for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("idx values must be consecutive and start at 0")

    prompts = [row["prompt"] for row in rows]
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError("Every pairwise row must contain a non-empty prompt")
    return prompts


def generate_embeddings(
    client: OpenAI, prompts: list[str], model: str, batch_size: int
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    embeddings = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [None] * len(batch)
        for item in response.data:
            batch_embeddings[item.index] = item.embedding
        if any(embedding is None for embedding in batch_embeddings):
            raise ValueError("Embedding response is missing an item")
        embeddings.extend(batch_embeddings)
        print(f"embedded {min(start + len(batch), len(prompts))}/{len(prompts)}")

    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Set EMBEDDING_API_KEY or pass --api-key")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to replace it")

    rows = load_jsonl(args.input)
    prompts = validate_rows(rows)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    embeddings = generate_embeddings(client, prompts, args.model, args.batch_size)
    if embeddings.ndim != 2 or embeddings.shape[1] != args.expected_dim:
        raise ValueError(
            f"Expected {args.expected_dim}-dimensional embeddings, got {embeddings.shape}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "shape": list(embeddings.shape),
                "model": args.model,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
