"""Convert a public RACER safetensors shard to standard offline routing data.

The shard has no text/IDs. Optional manifest entries must be in the exact
embedding row order and include query_id, query, correct_instruct,
correct_reasoning, num_tokens_instruct, and num_tokens_reasoning.
"""

import argparse
import json
from pathlib import Path

import torch


def convert(
    source,
    output,
    candidates,
    dataset_name,
    separate=False,
    manifest=None,
    embedding_model="BAAI/bge-m3",
    seed=42,
):
    from safetensors.torch import load_file

    values = load_file(str(source))
    x = (
        torch.cat(
            [
                values[k]
                for k in (
                    "embeddings_prompt",
                    "embeddings_answer_a",
                    "embeddings_answer_b",
                )
            ],
            dim=1,
        )
        if separate
        else values["embeddings_prompt_answer"]
    )
    fields = [
        "correct_instruct",
        "correct_reasoning",
        "num_tokens_instruct",
        "num_tokens_reasoning",
    ]
    count = len(x)
    if (
        x.ndim != 2
        or not torch.isfinite(x).all()
        or any(values[k].numel() != count for k in fields)
    ):
        raise ValueError("Shard has invalid shapes or embeddings")
    if count < 5:
        raise ValueError("At least five samples are required for a 60/20/20 split")
    if len(candidates) != 2 or candidates[0] == candidates[1]:
        raise ValueError("Specify two distinct candidate model IDs")
    records = None
    if manifest:
        records = [
            json.loads(line)
            for line in Path(manifest).read_text().splitlines()
            if line.strip()
        ]
        if len(records) != count:
            raise ValueError("Manifest row count differs from shard")
        for i, row in enumerate(records):
            for key in fields:
                if row[key] != values[key].reshape(-1)[i].item():
                    raise ValueError(f"Manifest does not match shard at row {i}, {key}")
            if not isinstance(row["query"], str) or not row["query"]:
                raise ValueError(f"Invalid manifest query at row {i}")
        if len({str(row["query_id"]) for row in records}) != count:
            raise ValueError("Manifest IDs must be unique")
    from llmrouter.models.racerrouter.data import pair_records

    all_rows = []
    for i in range(count):
        for j, candidate in enumerate(candidates):
            all_rows.append(
                dict(
                    query_id=records[i]["query_id"]
                    if records
                    else f"{dataset_name}/{i}",
                    query=records[i]["query"]
                    if records
                    else f"[PRECOMPUTED ONLY] {dataset_name}/{i}",
                    model_name=candidate,
                    embedding_id=i,
                    performance=values[fields[j]].reshape(-1)[i].item(),
                    output_tokens=values[fields[j + 2]].reshape(-1)[i].item(),
                )
            )
    pair_records(
        all_rows, x, candidates, "output_tokens"
    )  # fail before writing partial data
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(x.float(), output / "embeddings.pt")
    metadata = dict(
        model_id=embedding_model,
        input_mode="separate" if separate else "combined",
        dimension=x.shape[1],
        backend="upstream_vllm",
        template_version="racer_9d80ecc",
        precomputed_only=True,
    )
    (output / "embedding.json").write_text(json.dumps(metadata, indent=2))
    indexes = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    n_train, n_val = int(0.6 * count), int(0.2 * count)
    splits = dict(
        train=indexes[:n_train],
        validation=indexes[n_train : n_train + n_val],
        test=indexes[n_train + n_val :],
    )
    for split, selected in splits.items():
        rows = [row for i in selected for row in all_rows[2 * i : 2 * i + 2]]
        (output / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    (output / "splits.json").write_text(json.dumps(dict(seed=seed, **splits), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", nargs=2, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--separate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert(
        args.source,
        args.output,
        args.candidates,
        args.dataset_name,
        args.separate,
        args.manifest,
        args.embedding_model,
        args.seed,
    )


if __name__ == "__main__":
    main()
