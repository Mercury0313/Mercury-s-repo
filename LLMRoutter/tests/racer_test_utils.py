import json

import torch
import yaml


def config_file(tmp_path, *, robust=True, epochs=3, budget=3.0, count=12):
    metadata = dict(model_id="test/fixed-embeddings", input_mode="text", dimension=3)
    metadata_path = tmp_path / "embedding.json"
    metadata_path.write_text(json.dumps(metadata))
    vectors = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10
    torch.save(vectors, tmp_path / "embeddings.pt")
    rows = []
    for i in range(count):
        for model, reward, tokens in (("a", float(i % 2), 100), ("b", 1.0, 300 + i)):
            rows.append(
                dict(
                    query_id=i,
                    query=f"question {i}",
                    model_name=model,
                    performance=reward,
                    output_tokens=tokens,
                    embedding_id=i,
                )
            )
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    config = dict(
        data_path=dict(
            routing_data_train=str(tmp_path / "train.jsonl"),
            query_embedding_data=str(tmp_path / "embeddings.pt"),
        ),
        model_path=dict(save_model_path=str(tmp_path / "policy.pt")),
        racer=dict(
            candidate_models=["a", "b"], token_column="output_tokens", inference_seed=7
        ),
        embedding=dict(backend="precomputed", metadata_path=str(metadata_path)),
        hparam=dict(
            budget=budget,
            use_robust=robust,
            epochs=epochs,
            batch_size=4,
            seed=42,
            lr=0.001,
            dual_lr=0.01,
            tau_r=1.0,
            tau_g=50.0,
            entropy_coef=0.005,
        ),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, config, rows, vectors


def write_config(path, config):
    path.write_text(yaml.safe_dump(config))
    return path
