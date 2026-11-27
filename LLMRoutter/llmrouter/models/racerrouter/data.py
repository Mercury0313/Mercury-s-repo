"""Pair standard LLMRouter records without reducing them to class labels."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path):
    with resolve_path(path).open() as handle:
        return json.load(handle)


@dataclass
class PairedData:
    x: torch.Tensor
    r0: torch.Tensor
    r1: torch.Tensor
    ratio: torch.Tensor
    keys: list
    filtered_rows: int = 0

    def dataset(self):
        return TensorDataset(self.x, self.r0, self.r1, self.ratio)


def pair_records(records, embeddings, candidates, token_column):
    """Require exactly one valid observation per selected model and query."""
    frame = (
        records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    )
    required = {"query", "model_name", "embedding_id", "performance", token_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Routing data missing columns: {sorted(missing)}")
    selected = frame[frame.model_name.isin(candidates)].copy()
    filtered = len(frame) - len(selected)
    if selected.empty:
        raise ValueError("No observations for the configured candidate_models")
    use_ids = "query_id" in selected.columns
    if use_ids and selected.query_id.isna().any():
        raise ValueError("query_id must be present on every selected record")
    groups = {}
    for row in selected.to_dict("records"):
        query = row["query"]
        if not isinstance(query, str) or not query:
            raise ValueError("Each query must be a nonempty string")
        identity = (
            ["id", row["query_id"]]
            if use_ids
            else ["query", row.get("task_name", ""), query]
        )
        key = json.dumps(identity, sort_keys=True, allow_nan=False)
        groups.setdefault(key, []).append(row)
    xs, r0s, r1s, ratios, keys = [], [], [], [], []
    for key, rows in groups.items():
        by_model = {row["model_name"]: row for row in rows}
        if len(rows) != 2 or set(by_model) != set(candidates):
            raise ValueError(
                f"Query {key}: need exactly one record per candidate (missing or duplicate)"
            )
        a, b = (by_model[m] for m in candidates)
        if a["query"] != b["query"] or a.get("task_name", "") != b.get("task_name", ""):
            raise ValueError(f"Query {key}: conflicting query text or task")
        indexes = [a["embedding_id"], b["embedding_id"]]
        if any(
            isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0
            for i in indexes
        ):
            raise ValueError(f"Query {key}: embedding_id must be a nonnegative integer")
        if indexes[0] != indexes[1]:
            raise ValueError(f"Query {key}: inconsistent embedding_id")
        try:
            x = torch.as_tensor(embeddings[indexes[0]], dtype=torch.float32)
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"Query {key}: invalid embedding_id") from exc
        if x.ndim != 1 or not x.numel() or not torch.isfinite(x).all():
            raise ValueError(f"Query {key}: embedding must be a finite nonempty vector")
        if xs and x.shape != xs[0].shape:
            raise ValueError(f"Query {key}: inconsistent embedding dimensions")
        rewards = np.asarray([a["performance"], b["performance"]], dtype=float)
        tokens = np.asarray([a[token_column], b[token_column]], dtype=float)
        if not np.isfinite(rewards).all() or ((rewards < 0) | (rewards > 1)).any():
            raise ValueError(f"Query {key}: performance must be finite and in [0, 1]")
        if not np.isfinite(tokens).all() or (tokens <= 0).any():
            raise ValueError(
                f"Query {key}: output token counts must be finite and positive"
            )
        # Match the reference's float32 conversion BEFORE division.
        ts = torch.tensor(tokens, dtype=torch.float32)
        ratio = ts[1] / (ts[0] + 1e-8)
        if not torch.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Query {key}: invalid float32 cost ratio")
        xs.append(x)
        r0s.append(rewards[0])
        r1s.append(rewards[1])
        ratios.append(ratio)
        keys.append(key)
    return PairedData(
        torch.stack(xs),
        torch.tensor(r0s, dtype=torch.float32),
        torch.tensor(r1s, dtype=torch.float32),
        torch.stack(ratios),
        keys,
        filtered,
    )


def load_pairs(routing_path, embedding_path, candidates, token_column):
    # Read the existing JSONL/PT format directly: no generated CSV side effects,
    # and no train/test data reads when constructing an inference-only router.
    with resolve_path(routing_path).open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    embeddings = torch.load(
        resolve_path(embedding_path), map_location="cpu", weights_only=True
    )
    return pair_records(records, embeddings, candidates, token_column)
