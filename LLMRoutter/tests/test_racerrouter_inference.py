import pytest
import torch

from llmrouter.models.racerrouter import RACERRouter
from llmrouter.models.racerrouter.data import pair_records
from llmrouter.models.racerrouter.evaluation import evaluate_probabilities
from racer_test_utils import config_file, write_config


def ready_router(tmp_path):
    path, _, _, vectors = config_file(tmp_path)
    router = RACERRouter(str(path))
    router.initialize_policy(3)
    router.ready = True
    return router, vectors


def test_seeded_single_and_batch_sampling(tmp_path):
    router, vectors = ready_router(tmp_path)
    batch = [dict(query=str(i), embedding=v) for i, v in enumerate(vectors)]
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()
    probabilities = router.predict_proba(batch)
    first = router.route_batch(batch)
    assert torch.equal(before, torch.random.get_rng_state())
    router._rng.manual_seed(7)
    second = [router.route_single(q) for q in batch]
    assert [r["model_name"] for r in first] == [r["model_name"] for r in second]
    assert probabilities.shape == (12, 2)
    assert router.route_batch([]) == []


@pytest.mark.parametrize("bias,chosen", [(-1000.0, "a"), (1000.0, "b")])
def test_probability_endpoints(tmp_path, bias, chosen):
    router, vectors = ready_router(tmp_path)
    with torch.no_grad():
        router.model.net[-1].weight.zero_()
        router.model.net[-1].bias.fill_(bias)
    assert router.route_single(dict(embedding=vectors[0]))["model_name"] == chosen


def test_bad_features_and_untrained_router(tmp_path):
    path, _, _, _ = config_file(tmp_path)
    router = RACERRouter(str(path))
    with pytest.raises(RuntimeError, match="trained"):
        router.route_single(dict(query="hello"))
    router.initialize_policy(3)
    router.ready = True
    for vector in ([1, 2], [1, float("nan"), 3]):
        with pytest.raises(ValueError, match="finite embedding"):
            router.route_single(dict(embedding=vector))
    with pytest.raises(ValueError, match="precomputed embedding"):
        router.route_single(dict(query="hello"))


def test_checkpoint_candidate_order_conflict(tmp_path):
    router, _ = ready_router(tmp_path)
    router.save_router(tmp_path / "policy.pt")
    config = dict(
        model_path=dict(load_model_path=str(tmp_path / "policy.pt")),
        racer=dict(candidate_models=["b", "a"]),
    )
    with pytest.raises(ValueError, match="order conflicts"):
        RACERRouter(str(write_config(tmp_path / "other.yaml", config)))


def test_expected_replay(tmp_path):
    _, _, rows, vectors = config_file(tmp_path)
    data = pair_records(rows, vectors, ["a", "b"], "output_tokens")
    probabilities = torch.tensor([[0.75, 0.25]]).repeat(len(data.keys), 1)
    result = evaluate_probabilities(probabilities, data, 2)
    assert result["expected_reward"] == pytest.approx(0.625)
    assert result["expected_relative_cost"] == pytest.approx(
        0.75 + 0.25 * data.ratio.mean().item()
    )
