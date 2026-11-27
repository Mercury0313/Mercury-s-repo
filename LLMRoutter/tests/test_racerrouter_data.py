import pytest
import torch

from llmrouter.models.racerrouter.data import pair_records
from racer_test_utils import config_file


def test_pairing_and_candidate_order(tmp_path):
    _, _, rows, vectors = config_file(tmp_path)
    normal = pair_records(rows, vectors, ["a", "b"], "output_tokens")
    # Change within-pair order; preserve first occurrence order of queries.
    swapped_rows = [r for i in range(0, len(rows), 2) for r in rows[i : i + 2][::-1]]
    reordered = pair_records(swapped_rows, vectors, ["a", "b"], "output_tokens")
    torch.testing.assert_close(normal.r0, reordered.r0)
    reversed_models = pair_records(rows, vectors, ["b", "a"], "output_tokens")
    torch.testing.assert_close(normal.r1, reversed_models.r0)
    torch.testing.assert_close(reversed_models.ratio, 1 / normal.ratio)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: r.pop(), "missing or duplicate"),
        (lambda r: r.append(r[0].copy()), "missing or duplicate"),
        (lambda r: r[0].update(embedding_id=9999), "inconsistent embedding_id"),
        (lambda r: r[0].update(output_tokens=0), "positive"),
        (lambda r: r[0].update(performance=float("nan")), "performance"),
        (lambda r: r[0].update(query_id=None), "query_id"),
        (lambda r: r[0].update(query="different"), "conflicting"),
    ],
)
def test_invalid_pairs_fail(tmp_path, change, match):
    _, _, rows, vectors = config_file(tmp_path)
    change(rows)
    with pytest.raises(ValueError, match=match):
        pair_records(rows, vectors, ["a", "b"], "output_tokens")


def test_same_text_distinct_ids_are_not_merged(tmp_path):
    _, _, rows, vectors = config_file(tmp_path)
    for row in rows:
        row["query"] = "same text"
    assert len(pair_records(rows, vectors, ["a", "b"], "output_tokens").keys) == 12


def test_missing_ids_use_task_and_query(tmp_path):
    _, _, rows, vectors = config_file(tmp_path)
    for row in rows:
        del row["query_id"]
    assert len(pair_records(rows, vectors, ["a", "b"], "output_tokens").keys) == 12
