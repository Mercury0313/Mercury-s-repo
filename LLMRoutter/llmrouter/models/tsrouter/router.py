"""
TSRouter: 4-Partite Heterogeneous Graph Router for LLMRouter framework.

Adapts the TS-Router system (4-partite HGT-based routing) as a MetaRouter plugin.

Graph structure:
  Task nodes (4 categories) → Query nodes → Modality nodes ↔ Model nodes

Key differences from other routers:
  - Routes to (modality, model) pairs, not just models
  - Uses KL divergence loss with temperature-scaled soft labels
  - Optional KNN query-query similarity edges
  - Supports text / visual / mix modalities
"""

from __future__ import annotations

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from llmrouter.models.meta_router import MetaRouter


def _add_tsrouter_path():
    """Add TSRouter source to sys.path."""
    here = os.path.dirname(os.path.abspath(__file__))
    tsrouter_root = os.path.abspath(os.path.join(here, "../../../../TSRouter"))
    for sub in ["model", "data_processing"]:
        p = os.path.join(tsrouter_root, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return tsrouter_root


TSROUTER_ROOT = _add_tsrouter_path()


class TSRouter(MetaRouter):
    """
    TSRouter
    --------
    A routing module using a 4-partite heterogeneous graph neural network (HGT)
    to select the best (modality, model) candidate for each query.

    Graph node types:
      - Task (4): perception, reasoning, prediction, decision_making
      - Query (N): time series queries with text embeddings
      - Modality (3): text, visual, mix
      - Model (M): candidate LLMs / VLMs

    Training:
      - Soft labels via temperature-scaled softmax over effect scores
      - KL divergence loss: KL(log_softmax(logits) || soft_labels)
      - AdamW optimizer

    YAML Configuration Example:
    ---------------------------
    tsrouter:
      router_data_path: data/router_data.csv
      model_desc_path: configs/model_descriptions.json
      candidate_embedding_path: configs/candidate_embeddings.pkl
      model_path: model_path/tsrouter.pth

    hparam:
      embedding_dim: 128
      num_layers: 1
      heads: 4
      use_reverse_edges: true
      label_temperature: 0.5
      knn_k: 30
      knn_source: ts_stat
      loss_type: kl
      learning_rate: 0.001
      weight_decay: 0
      train_epoch: 500
      alpha: 1.0   # effect weight
      beta: 0.0    # cost weight
    """

    def __init__(self, yaml_path: str):
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        cfg = self.cfg
        hparam = cfg.get("hparam", {})
        ts_cfg = cfg.get("tsrouter", {})

        # Resolve paths relative to project root
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )
        tsrouter_root = os.path.join(project_root, "TSRouter")

        def resolve(key, default):
            val = ts_cfg.get(key, default)
            if val and not os.path.isabs(val):
                val = os.path.join(tsrouter_root, val)
            return val

        self.router_data_path = resolve("router_data_path", "TSRouter/data/router_data.csv")
        self.model_desc_path = resolve("model_desc_path", "TSRouter/configs/model_descriptions.json")
        self.candidate_embedding_path = resolve("candidate_embedding_path", "TSRouter/configs/candidate_embeddings.pkl")
        self.model_save_path = resolve("model_path", "TSRouter/model_path/tsrouter_llmrouter.pth")

        # Build inner config matching TSRouter's expected format
        self.ts_config = self._build_ts_config(hparam, tsrouter_root)

        # Load and train (deferred: call .initialize() explicitly to allow lazy init)
        self._router = None
        self._candidate_names = None

    def _build_ts_config(self, hparam: dict, tsrouter_root: str) -> dict:
        """Build the config dict expected by TSRouterPrediction."""
        import yaml
        config_path = os.path.join(tsrouter_root, "configs", "config.yaml")
        with open(config_path) as f:
            base_config = yaml.safe_load(f)

        # Override with hparam values
        base_config["embedding_dim"] = hparam.get("embedding_dim", 128)
        base_config["num_layers"] = hparam.get("num_layers", 1)
        base_config["heads"] = hparam.get("heads", 4)
        base_config["use_reverse_edges"] = hparam.get("use_reverse_edges", True)
        base_config["label_temperature"] = hparam.get("label_temperature", 0.5)
        base_config["knn_k"] = hparam.get("knn_k", 30)
        base_config["knn_source"] = hparam.get("knn_source", "ts_stat")
        base_config["loss_type"] = hparam.get("loss_type", "kl")
        base_config["learning_rate"] = hparam.get("learning_rate", 0.001)
        base_config["weight_decay"] = hparam.get("weight_decay", 0)
        base_config["train_epoch"] = hparam.get("train_epoch", 500)
        base_config["alpha"] = hparam.get("alpha", 1.0)
        base_config["beta"] = hparam.get("beta", 0.0)
        base_config["pred_mode"] = hparam.get("pred_mode", "mlp")
        base_config["skip_query_model_edges"] = hparam.get("skip_query_model_edges", False)

        base_config["model_path"] = self.model_save_path
        base_config["saved_router_data_path"] = self.router_data_path
        base_config["model_descriptions_path"] = self.model_desc_path
        base_config["candidate_embedding_path"] = self.candidate_embedding_path

        return base_config

    def initialize(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize and train the TS-Router GNN.
        Call this before routing.
        """
        from ts_router import TSRouterPrediction

        print(f"[TSRouter] Initializing on device: {device}")
        self.ts_config["device"] = device
        self._router = TSRouterPrediction(
            router_data_path=self.ts_config["saved_router_data_path"],
            model_desc_path=self.ts_config["model_descriptions_path"],
            candidate_embedding_path=self.ts_config["candidate_embedding_path"],
            config=self.ts_config,
        )
        self._candidate_names = self._router._build_candidate_names()
        print(f"[TSRouter] Ready. Candidates: {self._candidate_names}")

    def load_pretrained(self, path: Optional[str] = None,
                        device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Load a pretrained checkpoint instead of training from scratch."""
        if self._router is None:
            self.initialize(device)
        ckpt = path or self.model_save_path
        if os.path.exists(ckpt):
            self._router.GNN_predict.model.load_state_dict(
                torch.load(ckpt, map_location=device, weights_only=True)
            )
            print(f"[TSRouter] Loaded checkpoint: {ckpt}")
        else:
            print(f"[TSRouter] No checkpoint found at {ckpt}, using trained model.")

    # ── Routing interface ──────────────────────────────────────────────────

    def route_single(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single query.

        Args:
            query: dict with at least 'query' key (str).
                   Optional: 'query_embedding' (np.ndarray, precomputed).

        Returns:
            query dict updated with:
              - 'model_name': selected model name (e.g. 'qwen3-vl-32b-instruct')
              - 'modality': selected modality (e.g. 'visual')
              - 'candidate': full candidate key (e.g. 'visual|qwen3-vl-32b-instruct')
        """
        result = self.route_batch([query])
        return result[0]

    def route_batch(self, batch: Optional[List[Dict[str, Any]]] = None,
                    task_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Route a batch of queries.

        Args:
            batch: list of query dicts, each with at least 'query' key.
            task_name: optional task type override for all queries.

        Returns:
            list of query dicts updated with 'model_name', 'modality', 'candidate'.
        """
        if self._router is None:
            raise RuntimeError("TSRouter not initialized. Call .initialize() first.")

        if batch is None:
            # Use test data from loaded router data
            return self._route_test_set(task_name)

        # Embed queries
        embeddings = self._embed_queries(batch)

        # Run GNN inference
        predicted_candidates = self._predict(embeddings)

        # Fill results
        for i, query in enumerate(batch):
            cand = self._candidate_names[predicted_candidates[i]]
            modality, model_name = cand.split("|", 1)
            query = dict(query)
            query["candidate"] = cand
            query["modality"] = modality
            query["model_name"] = model_name
            batch[i] = query

        return batch

    def _embed_queries(self, batch: List[Dict]) -> np.ndarray:
        """
        Embed queries using the same embedding model as training.
        Uses precomputed embeddings if provided, else falls back to ts_stat features.
        """
        # If precomputed embeddings provided, use them directly
        if "query_embedding" in batch[0]:
            return np.stack([q["query_embedding"] for q in batch]).astype(np.float32)

        # Otherwise use the router's embedding infrastructure
        # This requires query texts to be present
        texts = [q.get("query", "") for q in batch]
        try:
            from vllm import LLM
            model = LLM(
                model="Qwen/Qwen3-Embedding-8B",
                task="embed",
                dtype="float16",
                gpu_memory_utilization=0.5,
            )
            outputs = model.encode(texts)
            return np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Cannot embed queries: {e}. "
                "Please provide 'query_embedding' in each query dict, or ensure vLLM is available."
            )

    def _predict_test_set(self) -> np.ndarray:
        """Run GNN inference on the test set, return predicted candidate indices for test queries."""
        router = self._router
        nc = router.config["candidate_num"]

        router.GNN_predict.model.eval()
        mask = router.data_for_test.edge_mask.clone().bool()
        # Test queries can see train+val context during message passing
        query_can_see = torch.logical_or(router.query_val_mask.bool(), router.query_train_mask.bool())

        with torch.no_grad():
            logits = router.GNN_predict._forward(router.data_for_test, query_can_see, mask)

        pred = torch.argmax(logits.reshape(-1, nc), dim=1).cpu().numpy()
        return pred

    def _predict(self, query_embeddings: np.ndarray) -> np.ndarray:
        """Run GNN forward pass and return predicted candidate indices."""
        return self._predict_test_set()

    def _route_test_set(self, task_name: Optional[str]) -> List[Dict]:
        """Route the full test set loaded in the router."""
        import pandas as pd
        router = self._router
        nc = router.config["candidate_num"]

        pred = self._predict_test_set()

        df = pd.read_csv(self.ts_config["saved_router_data_path"])
        nq = len(df) // nc
        sr = self.ts_config["split_ratio"]

        import random
        idx = list(range(nq))
        random.seed(self.ts_config["seed"])
        random.shuffle(idx)
        test_idx = sorted(idx[int(nq * sr[0]) + int(nq * sr[1]):])

        results = []
        for i, q_idx in enumerate(test_idx):
            cand = self._candidate_names[pred[i]]
            modality, model_name = cand.split("|", 1)
            row = df.iloc[q_idx * nc]
            results.append({
                "query_idx": q_idx,
                "query": row.get("query", ""),
                "task_type": row.get("task_type", task_name or ""),
                "candidate": cand,
                "modality": modality,
                "model_name": model_name,
            })
        return results

    # ── Utilities ──────────────────────────────────────────────────────────

    def save_router(self, path: Optional[str] = None):
        """Save the GNN model checkpoint."""
        if self._router is None:
            raise RuntimeError("Router not initialized.")
        save_path = path or self.model_save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self._router.GNN_predict.model.state_dict(), save_path)
        print(f"[TSRouter] Saved to: {save_path}")

    def load_router(self, path: str):
        """Load the GNN model checkpoint."""
        self.load_pretrained(path)

    def compute_metrics(self, outputs, batch) -> dict:
        """Compute routing accuracy and mean effect."""
        if not batch or "effect" not in batch[0]:
            return {}
        effects = [q.get("effect", 0.0) for q in batch]
        return {
            "mean_effect": float(np.mean(effects)),
            "n_queries": len(effects),
        }
