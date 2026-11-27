import json
from unittest.mock import patch

import torch

from llmrouter.cli.router_main import main
from llmrouter.cli.router_train import ROUTER_TRAINER_REGISTRY
from llmrouter.cli.router_inference import (
    ROUTER_REGISTRY,
    infer_query,
    load_router,
    route_query,
)
from llmrouter.models.racerrouter import RACERRouter, RACERTrainer
from llmrouter.models.racerrouter.router import LONGFORMER_METADATA
from racer_test_utils import config_file, write_config


def test_cli_train_and_route(tmp_path, capsys):
    path, config, _, _ = config_file(tmp_path, epochs=1)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        main(
            [
                "train",
                "--router",
                "racerrouter",
                "--config",
                str(path),
                "--device",
                "cpu",
            ]
        )
    finally:
        torch.set_num_threads(old)
    assert (tmp_path / "policy.pt").exists()
    assert ROUTER_TRAINER_REGISTRY["racerrouter"] == (RACERRouter, RACERTrainer)
    assert ROUTER_REGISTRY["racerrouter"] is RACERRouter
    from llmrouter.cli.router_chat import ROUTER_REGISTRY as chat_registry

    assert chat_registry["racerrouter"] is RACERRouter

    # Real text-path contract with the expensive encoder replaced by fixed
    # vectors. Training/checkpoint remains real; the encoder is mocked only.
    config["model_path"]["load_model_path"] = str(tmp_path / "policy.pt")
    inference = write_config(
        tmp_path / "infer.yaml", dict(model_path=config["model_path"])
    )
    router = load_router("racerrouter", str(inference))
    with patch.object(router, "_features", return_value=torch.zeros(1, 3)):
        with patch("llmrouter.cli.router_inference.call_api") as api:
            result = route_query("hello", router, "racerrouter")
            assert result["success"]
            api.assert_not_called()
        router.llm_data = {
            name: dict(model=name, api_endpoint="http://localhost:9000/v1")
            for name in ["a", "b"]
        }
        with patch(
            "llmrouter.cli.router_inference.call_api",
            return_value=dict(response="answer", prompt_tokens=2, completion_tokens=1),
        ) as api:
            result = infer_query("hello", router, "racerrouter")
            assert result["success"]
            assert result["response"] == "answer"
            assert api.call_count == 1
            assert api.call_args.args[0]["model_name"] in ("a", "b")


def test_existing_smallest_router_still_routes(tmp_path):
    metadata = {"a": dict(size="1B"), "b": dict(size="7B")}
    llms = tmp_path / "llms.json"
    llms.write_text(json.dumps(metadata))
    config = write_config(
        tmp_path / "baseline.yaml", dict(data_path=dict(llm_data=str(llms)))
    )
    router = load_router("smallest_llm", str(config))
    assert route_query("hello", router, "smallest_llm")["model_name"] == "a"


def test_text_only_cli_uses_declared_encoder(tmp_path):
    path, config, _, _ = config_file(tmp_path)
    config["embedding"] = dict(backend="longformer", metadata=LONGFORMER_METADATA)
    router = RACERRouter(str(write_config(path, config)))
    router.initialize_policy(768)
    router.ready = True
    router.save_router(tmp_path / "text.pt")
    inference = write_config(
        tmp_path / "text.yaml",
        dict(model_path=dict(load_model_path=str(tmp_path / "text.pt"))),
    )
    with patch(
        "llmrouter.utils.embeddings.get_longformer_embedding",
        return_value=torch.zeros(768),
    ) as encoder:
        with patch("llmrouter.cli.router_inference.call_api") as api:
            main(
                [
                    "infer",
                    "--router",
                    "racerrouter",
                    "--config",
                    str(inference),
                    "--query",
                    "hello",
                    "--route-only",
                    "--output",
                    str(tmp_path / "result.json"),
                ]
            )
            encoder.assert_called_once_with("hello")
            api.assert_not_called()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result[0]["success"]
    assert result[0]["model_name"] in ("a", "b")
