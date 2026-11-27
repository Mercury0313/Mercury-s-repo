"""Prepare a fixed, query-disjoint two-model experiment; never tune on test."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_MODELS = ["llama-3-8b-instruct-lite", "llama-3.3-70b-instruct-turbo"]


def prepare(frame, candidates):
    selected = frame[frame.model_name.isin(candidates)].copy()
    rows, rejected = [], []
    # Query text, not task_id (which can be missing), defines split identity.
    for query, group in selected.groupby("query", sort=True):
        qid = hashlib.sha256(query.encode()).hexdigest()
        if (
            len(group) != 2
            or set(group.model_name) != set(candidates)
            or group.task_name.nunique() != 1
        ):
            rejected.append(
                dict(
                    query_id=qid, reason="duplicate_or_incomplete_pair", rows=len(group)
                )
            )
            continue
        if (
            not np.isfinite(group.output_tokens).all()
            or (group.output_tokens <= 0).any()
        ):
            rejected.append(
                dict(
                    query_id=qid,
                    reason="nonpositive_or_invalid_tokens",
                    rows=len(group),
                )
            )
            continue
        if (
            not np.isfinite(group.performance).all()
            or not group.performance.between(0, 1).all()
        ):
            rejected.append(
                dict(query_id=qid, reason="invalid_reward", rows=len(group))
            )
            continue
        for model in candidates:
            row = group[group.model_name == model].iloc[0]
            rows.append(
                dict(
                    query_id=qid,
                    query=query,
                    task_name=row.task_name,
                    model_name=model,
                    performance=float(row.performance),
                    output_tokens=int(row.output_tokens),
                )
            )
    return rows, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "data/xroutebench"
    )
    parser.add_argument("--models", nargs=2, default=DEFAULT_MODELS)
    args = parser.parse_args()
    target = args.data / "prepared"
    target.mkdir(parents=True, exist_ok=True)
    frames = {
        split: pd.read_parquet(args.data / f"raw/llmrouter_generic/{split}.parquet")
        for split in ("train", "test")
    }
    pairs, rejections = {}, {}
    for split, frame in frames.items():
        pairs[split], rejections[split] = prepare(frame, args.models)
    assert not (
        {r["query_id"] for r in pairs["train"]} & {r["query_id"] for r in pairs["test"]}
    ), "Official splits overlap"
    unique = pd.DataFrame(pairs["train"]).drop_duplicates("query_id")
    train_ids, validation_ids = train_test_split(
        unique.query_id, test_size=0.2, random_state=20260907, stratify=unique.task_name
    )
    validation_ids = set(validation_ids)
    split_rows = dict(
        train=[r for r in pairs["train"] if r["query_id"] not in validation_ids],
        validation=[r for r in pairs["train"] if r["query_id"] in validation_ids],
        test=pairs["test"],
    )
    # Stable global indexes let train/validation/test share a query-only tensor.
    queries = {r["query_id"]: r["query"] for rows in split_rows.values() for r in rows}
    ids = sorted(queries)
    indexes = {qid: i for i, qid in enumerate(ids)}
    (target / "queries.json").write_text(
        json.dumps([dict(query_id=qid, query=queries[qid]) for qid in ids])
    )
    for split, rows in split_rows.items():
        with (target / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row, embedding_id=indexes[row["query_id"]])) + "\n"
                )
    train = pd.DataFrame(split_rows["train"])
    r = train.pivot(index="query_id", columns="model_name", values="performance")
    t = train.pivot(index="query_id", columns="model_name", values="output_tokens")
    a, b = args.models
    ratio = t[b] / t[a]
    budgets = [round(1 + f * (ratio.mean() - 1), 4) for f in (0.2, 0.5, 0.8)]
    manifest = dict(
        candidate_models=args.models,
        split_seed=20260907,
        counts={s: len(rows) // 2 for s, rows in split_rows.items()},
        split_query_ids={
            s: sorted({r["query_id"] for r in rows}) for s, rows in split_rows.items()
        },
        excluded=rejections,
        budgets=budgets,
        budget_rule="1 + [0.2, 0.5, 0.8] * (training mean B/A output-token ratio - 1)",
        training_statistics=dict(
            reward_a=r[a].mean(),
            reward_b=r[b].mean(),
            mean_ratio=ratio.mean(),
            ratio_quantiles=ratio.quantile([0, 0.5, 0.9, 0.99, 1]).to_dict(),
            a_better=float((r[a] > r[b]).mean()),
            b_better=float((r[b] > r[a]).mean()),
        ),
        test_policy="No test reward/cost aggregates or evaluation during tuning.",
    )
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("split_query_ids",)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
