"""Create synthetic, offline-only records to exercise training and evaluation."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(160, 8, generator=generator)
    torch.save(x, args.output / "embeddings.pt")
    metadata = dict(
        model_id="synthetic/demo-v1",
        input_mode="synthetic",
        dimension=8,
        pooling="none",
        normalization="none",
        template_version="none",
    )
    (args.output / "embedding.json").write_text(json.dumps(metadata, indent=2))
    for split, indexes in (("train", range(128)), ("test", range(128, 160))):
        rows = []
        for i in indexes:
            easy = bool(x[i, 0] > 0)
            for model, reward, tokens in (
                ("model_a", float(easy), 100),
                ("model_b", 1.0, 400),
            ):
                rows.append(
                    dict(
                        query_id=f"demo/{i}",
                        query=f"Synthetic item {i}",
                        model_name=model,
                        performance=reward,
                        output_tokens=tokens,
                        embedding_id=i,
                    )
                )
        (args.output / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    print(
        f"Wrote synthetic demo to {args.output}. These features cannot encode real text."
    )


if __name__ == "__main__":
    main()
