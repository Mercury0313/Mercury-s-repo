import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from llmrouter.models.racerrouter import RACERRouter, RACERTrainer
from racer_test_utils import config_file, write_config


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("robust", [False, True])
def test_train_save_and_reload_without_data(tmp_path, robust):
    path, config, _, vectors = config_file(tmp_path, robust=robust)
    router = RACERRouter(str(path))
    trainer = RACERTrainer(router)
    summary = trainer.train()
    assert summary["method"] == ("RACER" if robust else "ACER")
    assert set(trainer.split_manifest["train"]).isdisjoint(
        trainer.split_manifest["validation"]
    )
    expected = router.predict_proba([dict(embedding=vectors[0])])
    (tmp_path / "train.jsonl").unlink()
    (tmp_path / "embeddings.pt").unlink()
    (tmp_path / "embedding.json").unlink()
    inference = dict(
        model_path=dict(load_model_path=config["model_path"]["save_model_path"])
    )
    restored = RACERRouter(str(write_config(tmp_path / "inference.yaml", inference)))
    torch.testing.assert_close(
        restored.predict_proba([dict(embedding=vectors[0])]), expected
    )
    assert restored.training_parameters["use_robust"] == robust
    assert (tmp_path / "policy.history.json").exists()


def test_infeasible_budget_is_reported(tmp_path):
    path, _, _, _ = config_file(tmp_path, budget=0.1)
    router = RACERRouter(str(path))
    with pytest.warns(UserWarning, match="No checkpoint"):
        result = RACERTrainer(router).train()
    assert not result["budget_feasible"]
    assert result["budget_violation"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("robust", [False, True])
def test_cuda_training_and_cross_device_checkpoint(tmp_path, robust):
    path, config, _, vectors = config_file(tmp_path, robust=robust)
    router = RACERRouter(str(path))
    summary = RACERTrainer(router, device="cuda").train()
    assert next(router.model.parameters()).device.type == "cuda"
    assert summary["method"] == ("RACER" if robust else "ACER")
    inputs = [dict(embedding=v) for v in vectors]
    expected = router.predict_proba(inputs)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(expected.sum(1), torch.ones(len(vectors)))

    inference = write_config(
        tmp_path / "inference.yaml",
        dict(model_path=dict(load_model_path=config["model_path"]["save_model_path"])),
    )
    restored = RACERRouter(str(inference))
    assert next(restored.model.parameters()).device.type == "cpu"
    torch.testing.assert_close(
        restored.predict_proba(inputs), expected, rtol=1e-5, atol=1e-6
    )
    restored.model.to("cuda")
    torch.testing.assert_close(
        restored.predict_proba([dict(embedding=v.cuda()) for v in vectors]),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rejects_composite_reward_configuration(tmp_path):
    path, config, _, _ = config_file(tmp_path)
    config["metric"] = dict(weights=dict(performance=1, cost=0.5))
    with pytest.raises(ValueError, match="raw performance"):
        RACERTrainer(RACERRouter(str(write_config(path, config))))


def test_explicit_validation_overlap(tmp_path):
    path, config, _, _ = config_file(tmp_path)
    config["racer"].update(
        validation_routing_path=config["data_path"]["routing_data_train"],
        validation_embedding_path=config["data_path"]["query_embedding_data"],
    )
    with pytest.raises(ValueError, match="overlap"):
        RACERTrainer(RACERRouter(str(write_config(path, config)))).train()


@pytest.mark.parametrize("robust", [False, True])
def test_matches_pinned_upstream_training(tmp_path, robust):
    fixture_dir = Path(__file__).parent / "fixtures" / "racerrouter"
    reference = json.loads((fixture_dir / "reference.json").read_text())
    fixture = np.load(fixture_dir / "reference.npz", allow_pickle=False)
    case = "racer" if robust else "acer"
    path, config, _, _ = config_file(tmp_path, robust=robust)
    config["hparam"].update(reference["hparam"], use_robust=robust)
    router = RACERRouter(str(write_config(path, config)))
    router.initialize_policy(3)
    router.model.load_state_dict(
        {
            k: torch.from_numpy(fixture[f"initial/{k}"])
            for k in router.model.state_dict()
        }
    )
    train = [
        tuple(torch.from_numpy(fixture[f"train/{i}/{j}"]) for j in range(4))
        for i in range(3)
    ]
    val = [tuple(torch.from_numpy(fixture[f"val/{j}"]) for j in range(4))]
    trainer = RACERTrainer(router)
    with (
        pytest.warns(UserWarning) if not reference[case]["feasible"] else nullcontext()
    ):
        trainer.train((train, val))
    for actual, expected in zip(trainer.history, reference[case]["history"]):
        for key in expected:
            assert actual[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-6)
    assert router.training_summary["best_epoch"] == reference[case]["best_epoch"]
    for name, value in router.model.state_dict().items():
        torch.testing.assert_close(
            value, torch.from_numpy(fixture[f"{case}/{name}"]), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize(
    "metrics,expected_epoch,feasible",
    [
        ([(0.9, 2.0), (1.0, 4.0), (0.8, 1.5)], 1, True),
        ([(0.9, 2.0), (0.9, 2.0), (0.8, 2.0)], 1, True),
        ([(0.9, 4.0), (1.0, 5.0), (0.8, 4.0)], 1, False),
    ],
)
def test_checkpoint_selection_and_selected_epoch_lambda(
    tmp_path, monkeypatch, metrics, expected_epoch, feasible
):
    path, _, _, _ = config_file(tmp_path)
    router = RACERRouter(str(path))
    snapshots = []

    def validation(model, loader, device):
        snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        reward, cost = metrics[len(snapshots) - 1]
        return reward, cost, 0.5

    monkeypatch.setattr(
        "llmrouter.models.racerrouter.trainer.evaluate_policy", validation
    )
    trainer = RACERTrainer(router)
    with nullcontext() if feasible else pytest.warns(UserWarning):
        result = trainer.train()
    assert result["best_epoch"] == expected_epoch
    assert result["budget_feasible"] == feasible
    assert (
        result["selected_epoch_lambda"] == trainer.history[expected_epoch - 1]["lambda"]
    )
    for key, value in router.model.state_dict().items():
        torch.testing.assert_close(value, snapshots[expected_epoch - 1][key])
