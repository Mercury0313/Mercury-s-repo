"""Two-model probability routing using the public RACER policy."""

import copy
from pathlib import Path

import torch
from torch import nn
import yaml

from llmrouter.models.meta_router import MetaRouter
from .data import read_json, resolve_path
from .policy import PolicyNet


SOURCE_COMMIT = "9d80eccb5cae14684e372bce35929c6c8a13cead"
LONGFORMER_METADATA = {
    "model_id": "allenai/longformer-base-4096",
    "pooling": "masked_mean",
    "normalization": "none",
    "max_length": 4096,
    "input_mode": "text",
    "template_version": "raw_text_v1",
    "dimension": 768,
}


class RACERRouter(MetaRouter):
    """Select one of two configured models, sampling from the learned policy.

    Training records are loaded by the trainer only. Inference requires no
    historical rewards, token counts, or training files.
    """

    def __init__(self, yaml_path: str):
        # Avoid MetaRouter's eager train/test loading and CSV side effects.
        super().__init__(model=nn.Identity())
        with Path(yaml_path).open() as handle:
            self.cfg = yaml.safe_load(handle) or {}
        self.racer_config = self.cfg.get("racer", {})
        self.token_column = self.racer_config.get("token_column")
        self.candidate_models = self.racer_config.get("candidate_models")
        self.embedding_config = copy.deepcopy(self.cfg.get("embedding", {}))
        metadata_path = self.embedding_config.pop("metadata_path", None)
        if metadata_path:
            self.embedding_config["metadata"] = read_json(metadata_path)
        self.ready = False
        self.input_dim = None
        self.training_summary = {}
        self.training_parameters = {}
        self._rng = torch.Generator(device="cpu")
        seed = self.racer_config.get("inference_seed")
        self._rng.seed() if seed is None else self._rng.manual_seed(seed)
        llm_path = self.cfg.get("data_path", {}).get("llm_data")
        self.llm_data = read_json(llm_path) if llm_path else {}
        load_path = self.cfg.get("model_path", {}).get("load_model_path")
        if load_path:
            self.load_router(load_path)
        else:
            self._validate_candidates()
            self._validate_embedding_config()
        if self.llm_data and any(m not in self.llm_data for m in self.candidate_models):
            raise ValueError("llm_data must contain both configured candidate_models")

    def _validate_candidates(self):
        names = self.candidate_models
        if (
            not isinstance(names, list)
            or len(names) != 2
            or any(not isinstance(m, str) or not m for m in names)
            or names[0] == names[1]
        ):
            raise ValueError(
                "racer.candidate_models must contain two distinct model IDs in order"
            )

    def _validate_embedding_config(self):
        backend = self.embedding_config.get("backend")
        if backend not in ("precomputed", "longformer"):
            raise ValueError("embedding.backend must be precomputed or longformer")
        if backend == "longformer":
            metadata = self.embedding_config.setdefault(
                "metadata", copy.deepcopy(LONGFORMER_METADATA)
            )
            if any(metadata.get(k) != v for k, v in LONGFORMER_METADATA.items()):
                raise ValueError(
                    "Longformer metadata does not match the library's text encoder"
                )
        metadata = self.embedding_config.get("metadata", {})
        if not metadata.get("model_id") or not metadata.get("input_mode"):
            raise ValueError("Embedding metadata must declare model_id and input_mode")
        dimension = metadata.get("dimension")
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError(
                "Embedding metadata must declare a positive integer dimension"
            )

    def initialize_policy(self, input_dim, seed=42):
        if input_dim != self.embedding_config["metadata"]["dimension"]:
            raise ValueError(
                "Training embedding dimension differs from embedding metadata"
            )
        self.input_dim = input_dim
        # Keep initialization reproducible without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.model = PolicyNet(input_dim)
        self.ready = False

    def _features(self, batch):
        vectors = []
        for query in batch:
            if not isinstance(query, dict):
                raise ValueError("Each routing input must be a dictionary")
            if "embedding" in query:
                vector = torch.as_tensor(query["embedding"], dtype=torch.float32)
            else:
                if self.embedding_config["backend"] != "longformer":
                    raise ValueError(
                        "This checkpoint needs a precomputed embedding; text encoding is unavailable"
                    )
                text = query.get("query")
                if not isinstance(text, str) or not text:
                    raise ValueError("query must be a nonempty string")
                from llmrouter.utils.embeddings import get_longformer_embedding

                vector = get_longformer_embedding(text)
            if vector.shape != (self.input_dim,) or not torch.isfinite(vector).all():
                raise ValueError(
                    f"Expected a finite embedding of shape ({self.input_dim},)"
                )
            vectors.append(vector.float().cpu())
        return torch.stack(vectors)

    def predict_proba(self, batch):
        """Return CPU probabilities [N, 2]; never sample or call a candidate LLM."""
        if not self.ready:
            raise RuntimeError("RACER must be trained or loaded before routing")
        if not isinstance(batch, list):
            raise ValueError("predict_proba expects a list of input dictionaries")
        if not batch:
            return torch.empty((0, 2))
        features = self._features(batch).to(next(self.model.parameters()).device)
        self.model.eval()
        with torch.no_grad():
            p = self.model(features).reshape(-1).cpu()
        if not torch.isfinite(p).all():
            raise ValueError("Policy produced non-finite probabilities")
        return torch.stack((1 - p, p), dim=1)

    def route_single(self, query_input):
        return self.route_batch([query_input])[0]

    def route_batch(self, batch):
        probabilities = self.predict_proba(batch)
        # One draw per input gives the same RNG consumption for single/batch calls.
        choices = [
            int(torch.bernoulli(p[1], generator=self._rng).item())
            for p in probabilities
        ]
        return [
            dict(
                query=query.get("query", ""),
                model_name=self.candidate_models[choice],
                routing_probabilities=dict(zip(self.candidate_models, p.tolist())),
                selection_mode="sample",
            )
            for query, p, choice in zip(batch, probabilities, choices)
        ]

    def save_router(self, path):
        if not self.ready:
            raise RuntimeError("Cannot save an untrained RACER policy")
        checkpoint = {
            "format_version": 1,
            "source_commit": SOURCE_COMMIT,
            "state_dict": {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            },
            "input_dim": self.input_dim,
            "candidate_models": self.candidate_models,
            "embedding": self.embedding_config,
            "cost_mode": "relative_output_tokens",
            "token_column": self.token_column,
            "training_parameters": self.training_parameters,
            "training_summary": self.training_summary,
        }
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target)

    def load_router(self, path):
        checkpoint = torch.load(
            resolve_path(path), map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
            raise ValueError(
                "Unsupported RACER checkpoint; upstream weights need explicit metadata conversion"
            )
        if checkpoint.get("cost_mode") != "relative_output_tokens":
            raise ValueError("Unsupported RACER checkpoint cost mode")
        if self.token_column is not None and self.token_column != checkpoint.get(
            "token_column"
        ):
            raise ValueError("Configured token column conflicts with checkpoint")
        self.token_column = checkpoint.get("token_column")
        names = checkpoint["candidate_models"]
        if self.candidate_models is not None and self.candidate_models != names:
            raise ValueError("candidate_models order conflicts with checkpoint")
        if self.embedding_config and self.embedding_config != checkpoint["embedding"]:
            raise ValueError("Embedding configuration conflicts with checkpoint")
        self.candidate_models = names
        self.embedding_config = checkpoint["embedding"]
        self._validate_candidates()
        self._validate_embedding_config()
        self.initialize_policy(checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["state_dict"])
        if any(not torch.isfinite(t).all() for t in self.model.state_dict().values()):
            raise ValueError("Checkpoint contains non-finite policy parameters")
        self.training_parameters = checkpoint["training_parameters"]
        self.training_summary = checkpoint["training_summary"]
        for key in ("budget", "use_robust", "tau_r", "tau_g"):
            value = self.cfg.get("hparam", {}).get(key)
            if value is not None and value != self.training_parameters.get(key):
                raise ValueError(
                    f"hparam.{key} conflicts with the trained checkpoint; retrain to change it"
                )
        mode = self.racer_config.get("cost_mode", "relative_output_tokens")
        if mode != checkpoint["cost_mode"]:
            raise ValueError("Configured cost mode conflicts with checkpoint")
        self.model.eval()
        self.ready = True
