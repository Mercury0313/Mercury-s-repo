import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from llmrouter.models.racerrouter.data import load_pairs
from llmrouter.models.racerrouter import RACERRouter
from racer_test_utils import config_file, write_config


@pytest.mark.parametrize("separate", [False, True])
def test_upstream_shard_conversion(tmp_path, separate):
    script = Path(__file__).parents[1] / "examples/racerrouter/convert_upstream_data.py"
    spec = importlib.util.spec_from_file_location("convert_racer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    fields = dict(
        embeddings_prompt=x,
        embeddings_answer_a=x + 10,
        embeddings_answer_b=x + 20,
        embeddings_prompt_answer=x.clone(),
        correct_instruct=torch.zeros(10),
        correct_reasoning=torch.ones(10),
        num_tokens_instruct=torch.full((10,), 100),
        num_tokens_reasoning=torch.full((10,), 400),
    )
    source = tmp_path / "source.safetensors"
    save_file(fields, str(source))
    output = tmp_path / "converted"
    module.convert(source, output, ["a", "b"], "test", separate=separate)
    splits = json.loads((output / "splits.json").read_text())
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [6, 2, 2]
    data = load_pairs(
        output / "train.jsonl", output / "embeddings.pt", ["a", "b"], "output_tokens"
    )
    assert data.x.shape == (6, 6 if separate else 2)
    assert data.ratio.tolist() == [4.0] * 6
    if separate:
        index = splits["train"][0]
        torch.testing.assert_close(
            data.x[0], torch.cat([x[index], x[index] + 10, x[index] + 20])
        )


def test_manifest_mismatch_fails_before_output(tmp_path):
    script = Path(__file__).parents[1] / "examples/racerrouter/convert_upstream_data.py"
    spec = importlib.util.spec_from_file_location("convert_racer", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.safetensors"
    save_file(
        dict(
            embeddings_prompt_answer=torch.zeros(5, 2),
            correct_instruct=torch.zeros(5),
            correct_reasoning=torch.ones(5),
            num_tokens_instruct=torch.ones(5),
            num_tokens_reasoning=torch.ones(5),
        ),
        str(source),
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(dict(correct_instruct=1)) for _ in range(5))
    )
    with pytest.raises(ValueError, match="does not match"):
        module.convert(
            source, tmp_path / "output", ["a", "b"], "test", manifest=manifest
        )
    assert not (tmp_path / "output").exists()


def test_upstream_checkpoint_conversion(tmp_path):
    script = (
        Path(__file__).parents[1]
        / "examples/racerrouter/convert_upstream_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("convert_checkpoint", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path, _, _, vectors = config_file(tmp_path)
    router = RACERRouter(str(path))
    router.initialize_policy(3)
    router.ready = True
    expected = router.predict_proba([dict(embedding=vectors[0])])
    source = tmp_path / "upstream.pt"
    torch.save(
        dict(
            state_dict=router.model.state_dict(),
            policy_type="mlp",
            budget=3.0,
            use_robust=True,
            tau_r=1.0,
            tau_g=50.0,
            best_epoch=2,
            best_val_cost=2.5,
            best_val_reward=0.8,
            lambda_dual=0.1,
        ),
        source,
    )
    module.convert(source, path, tmp_path / "converted.pt")
    config = write_config(
        tmp_path / "infer.yaml",
        dict(model_path=dict(load_model_path=str(tmp_path / "converted.pt"))),
    )
    restored = RACERRouter(str(config))
    torch.testing.assert_close(
        restored.predict_proba([dict(embedding=vectors[0])]), expected
    )
    assert restored.training_summary["selected_epoch_lambda"] is None
