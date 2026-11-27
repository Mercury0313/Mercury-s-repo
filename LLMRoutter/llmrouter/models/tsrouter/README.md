# TSRouter

## Overview

**TSRouter** routes time-series queries to the best **(modality, model)** pair using a **4-partite Heterogeneous Graph Transformer (HGT)**. Unlike other routers that select among text-only LLMs, TSRouter jointly models three modalities (text / visual / mix) and six candidate models in a single heterogeneous graph, making routing decisions that are sensitive to both task category and query content.

### Key Differences from Other Routers

| Aspect | TSRouter | Other Routers (KNN, MLP, Graph, …) |
|--------|----------|-------------------------------------|
| **Output** | (modality, model) pair | model only |
| **Modalities** | text / visual / mix | text only |
| **Graph** | 4-partite HGT (Task, Query, Modality, Model) | bipartite query-LLM or none |
| **Label** | temperature-scaled soft labels via KL divergence | hard labels or BCE |
| **Query edges** | optional KNN similarity edges | none |
| **Node types** | 4 (Task, Query, Modality, Model) | 1–2 |

---

## Architecture

### 4-Partite Heterogeneous Graph

```
Task (4)
  │  task→query
  ▼
Query (N) ──knn──▶ Query (N)
  │  query→modality        query→model
  ▼                          ▼
Modality (3)            Model (6)
  │        modality↔model
  └───────────────────────────┘

          HGT message passing
                 ↓
     s(q, c) = MLP(h_q ⊙ (h_m + h_l))
                 ↓
         argmax over candidates
```

**Node types:**

| Type | Count | Features |
|------|-------|----------|
| Task | 4 | LLM embedding of category description (4096-dim) |
| Query | N | LLM embedding of query text (4096-dim) |
| Modality | 3 | LLM embedding of modality description (4096-dim) |
| Model | 6 | LLM embedding of model description (4096-dim) |

**Edge types** (+ optional reverse edges):

- `task → query` (task membership)
- `query → modality` (compatibility, edge features: 4-dim)
- `query → model` (compatibility, edge features: 4-dim)
- `modality ↔ model` (co-occurrence in candidates)
- `query → query` (KNN similarity, optional)

### Prediction Head

For each query node `q` and candidate `c = (modality m, model l)`:

```
s(q, c) = MLP( h_q ⊙ (h_m + h_l) )
```

The candidate with the highest score is selected.

### Training

- **Soft labels**: `soft_label = softmax(effect_score / τ)` — temperature-scaled over all candidates per query
- **Loss**: KL divergence `KL( log_softmax(logits) ∥ soft_labels )`
- **Optimizer**: AdamW
- **Model selection**: checkpoint saved at best validation reward; reloaded after training

---

## Configuration

### YAML Example (`configs/model_config_train/tsrouter.yaml`)

```yaml
tsrouter:
  router_data_path: data/router_data.csv          # query × candidate performance CSV
  model_desc_path: configs/model_descriptions.json
  candidate_embedding_path: configs/candidate_embeddings.pkl
  model_path: model_path/tsrouter_llmrouter.pth

hparam:
  # GNN architecture
  embedding_dim: 128        # hidden dim after FeatureAlign (4096 → D)
  num_layers: 1             # number of HGTConv layers
  heads: 4                  # multi-head attention heads
  use_reverse_edges: true   # add reverse edges

  # Soft-label training
  loss_type: kl             # 'kl' (recommended) or 'bce'
  label_temperature: 0.5    # temperature τ for softmax over effect scores
  alpha: 1.0                # effect weight
  beta: 0.0                 # cost weight

  # KNN query-similarity edges
  knn_k: 30                 # KNN neighbours per query
  knn_source: ts_stat       # 'ts_stat' or 'embedding'

  # Prediction head
  pred_mode: mlp            # 'mlp' or 'dot'

  # Optimiser
  learning_rate: 0.001
  weight_decay: 0
  train_epoch: 500
```

### Hyperparameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | `128` | Hidden dimension D after linear projection of 4096-dim node features |
| `num_layers` | `1` | Number of HGTConv message-passing layers |
| `heads` | `4` | Multi-head attention heads in HGT |
| `use_reverse_edges` | `true` | Add reverse edges to all edge types |
| `loss_type` | `kl` | `kl` = KL divergence on soft labels; `bce` = binary cross-entropy |
| `label_temperature` | `0.5` | τ for softmax over effect scores; lower = sharper labels |
| `alpha` | `1.0` | Weight of effect score in combined label |
| `beta` | `0.0` | Weight of cost penalty in combined label |
| `knn_k` | `30` | Number of KNN neighbours for query–query similarity edges |
| `knn_source` | `ts_stat` | Feature used for KNN: `ts_stat` (statistical features) or `embedding` |
| `pred_mode` | `mlp` | Prediction head: `mlp` (Hadamard + MLP) or `dot` (element-wise sum) |
| `learning_rate` | `0.001` | AdamW learning rate |
| `weight_decay` | `0` | AdamW weight decay |
| `train_epoch` | `500` | Training epochs |

---

## CLI Usage

### Training

```bash
llmrouter train --router tsrouter --config configs/model_config_train/tsrouter.yaml --device cuda
```

### Inference

```bash
# Route a single query
llmrouter infer --router tsrouter --config configs/model_config_train/tsrouter.yaml \
    --query "Forecast the next 12 steps of this time series." --route-only

# Batch routing from file
llmrouter infer --router tsrouter --config configs/model_config_train/tsrouter.yaml \
    --input queries.jsonl --output results.json --route-only
```

### Interactive Chat

```bash
llmrouter chat --router tsrouter --config configs/model_config_train/tsrouter.yaml
```

---

## Python API

### Training

```python
from llmrouter.models import TSRouter, TSRouterTrainer

router  = TSRouter("configs/model_config_train/tsrouter.yaml")
trainer = TSRouterTrainer(router=router, device="cuda")
trainer.train()   # builds graph, trains GNN, saves best-val checkpoint
```

### Inference — test set

```python
# Route the internal test split (826 queries)
results = router.route_batch()
print(f"Routed {len(results)} queries")
print(results[0])
# {'query_idx': 2, 'task_type': 'perception',
#  'candidate': 'visual|qwen3-vl-32b-instruct',
#  'modality': 'visual', 'model_name': 'qwen3-vl-32b-instruct', ...}
```

### Inference — custom queries

```python
batch = [
    {"query": "Predict the next value of this sequence.", "query_embedding": emb_array},
    {"query": "Are these two time series correlated?",   "query_embedding": emb_array2},
]
results = router.route_batch(batch)
for r in results:
    print(r["candidate"], r["model_name"])
```

> **Note**: If `query_embedding` is not provided, TSRouter will call `Qwen3-Embedding-8B` via vLLM to embed the query text on the fly. Pre-computing embeddings is recommended for batch inference.

### Load pretrained checkpoint

```python
router = TSRouter("configs/model_config_train/tsrouter.yaml")
router.initialize(device="cuda")
router.load_pretrained("model_path/tsrouter_llmrouter.pth")
results = router.route_batch()
```

---

## Data Requirements

TSRouter uses its **own data files** (separate from the standard LLMRouter JSONL format), resolved relative to the `TSRouter/` directory:

| File | Description |
|------|-------------|
| `configs/model_descriptions.json` | Model metadata (name, modality, description, pricing) — hand-maintained |
| `data/oracle_full.csv` | Oracle correctness scores of all candidates on TSRBench — **generated** via [`tsrbench/`](../../../data/tsrbench/) |
| `data/router_data.csv` | `N_query × N_candidate` rows of `(query_id, candidate, effect_score, cost, …)` — **generated** |
| `configs/candidate_embeddings.pkl` | Candidate (modality-model) embeddings — **generated** |
| `data/query_embeddings.npy`, `data/*.npy` | Query / task / modality / model node embeddings — **generated** |

Only `model_descriptions.json` is hand-maintained; `oracle_full.csv` is produced by running the candidate models on TSRBench with the [`tsrbench/`](../../../data/tsrbench/) scripts. Everything else marked **generated** is built automatically by TSRouter's data-construction phase, which samples TSRBench, embeds queries and node descriptions with a sentence-transformer, and joins the oracle scores:

```bash
cd ../TSRouter          # sibling of LLMRouter/
python run_exp.py       # Phase 1 generates router_data.csv + all embeddings, then trains
```

To rebuild the oracle scores themselves (or add new candidate models), see [`tsrbench/`](../../../data/tsrbench/). The `data_path` section in the YAML is populated with placeholder paths to satisfy the MetaRouter base class but is **not consumed** by TSRouter.

---

## When to Use TSRouter

**Good fit:**
- Time-series analysis tasks spanning perception / reasoning / prediction / decision-making
- You have both text-only LLMs and vision-language models (VLMs) as candidates
- Queries vary in whether visual context helps (chart images vs. raw numbers)
---

## Technical Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 with CUDA
- PyTorch Geometric (`torch_geometric`) ≥ 2.4
- vLLM (optional, for on-the-fly query embedding)
- TSRouter source code at `../../TSRouter/` (relative to `LLMRouter/`)
---

## Reproducing the TSRBench Oracle

The oracle correctness scores that TSRouter trains on can be reproduced from scratch — downloading TSRBench and running all six candidate models locally via vLLM — with the self-contained scripts in [`tsrbench/`](../../../data/tsrbench/) at the repository root.
