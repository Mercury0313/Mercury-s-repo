# RACER Router

RACER learns a lightweight probability policy over two candidate LLMs using a
relative output-token budget and distributionally robust reward/cost weighting.
Set `hparam.use_robust: false` to train the ACER ablation using ordinary averages.
Each candidate pair and budget requires its own trained checkpoint.

This implementation follows [onepounchman/RACER at 9d80ecc](https://github.com/onepounchman/RACER/tree/9d80eccb5cae14684e372bce35929c6c8a13cead),
especially `racer/policy.py`, `racer/trainer.py`, and `racer/evaluator.py`.
RACER was introduced in the **ICML 2026** paper
[Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](https://openreview.net/forum?id=NoP2UQymHU).
See the paper and upstream repository for the original experiments.

**Training data**

Use LLMRouter's JSONL routing records, with one row per query and candidate:

```json
{"query_id":"q1","query":"Example question","model_name":"model_a","performance":0.0,"output_tokens":100,"embedding_id":0}
{"query_id":"q1","query":"Example question","model_name":"model_b","performance":1.0,"output_tokens":500,"embedding_id":0}
```

The PT embedding file is indexed by `embedding_id`. Keep the two original
performance scores; do not replace them with a best-model label or a score
already penalized by cost. Scores must be finite, in [0,1], and higher-is-better.
Explicitly configure `racer.token_column` to a positive output-token count.
The library's `token_num` may include input tokens; use `output_tokens` when
available to reproduce the original cost definition.

Other candidate models are filtered out. Missing/duplicate selected observations,
conflicting query text, invalid embeddings, nonpositive tokens and invalid scores
raise errors. Pairing uses `query_id` if present on all selected rows, otherwise
`(task_name, query)`. Different queries must have distinct identifiers. Train/val
are split by query, not by individual model record. An explicit validation pair
of files can be provided via `racer.validation_routing_path` and
`racer.validation_embedding_path`; otherwise 20% of the supplied training set
is reserved. The trainer does not read `routing_data_test`.

**Budget and algorithm**

Candidate order `[model_a, model_b]` defines actions 0 and 1. The MLP architecture
is D → 256 → 128 → 64 → 1 with ReLU hidden layers and sigmoid output `p`.

- Expected reward per example: `(1-p) * reward_a + p * reward_b`.
- Relative cost per example: `(1-p) + p * tokens_b / (tokens_a + 1e-8)`.
- RACER increases weights on low-reward examples and high-cost examples within
  each training minibatch; the weights are detached during differentiation.
- ACER uses the ordinary minibatch means, ignoring both temperatures.

A budget of 2 refers to the average of per-example relative costs. It does not
mean total token use is bounded by twice the baseline's total token use.

The public implementation uses robust cost in the policy update, but updates
the dual variable once per epoch using the **unweighted mean of minibatch mean
costs**. Validation/checkpoint selection uses ordinary expected costs. These
semantics, AdamW defaults, entropy/dual regularization and checkpoint tie-breaking
are preserved, including a final partially filled batch. This is a fixed-temperature
reweighting implementation; it does not expose a KL-radius solver.

The highest-reward validation-feasible epoch is selected. If none is feasible,
the closest-cost epoch is returned with a warning and `budget_feasible: false`.
Training a budgeted policy is not a guarantee about finite sampled traffic or
OOD deployment cost.

**Configuration**

Start with [training config](../../../configs/model_config_train/racerrouter.yaml)
and [inference config](../../../configs/model_config_test/racerrouter.yaml).

| Field | Default / meaning |
|---|---|
| `racer.candidate_models` | Required: two distinct candidate IDs in fixed order |
| `racer.token_column` | Required: output-token column |
| `racer.cost_mode` | `relative_output_tokens` |
| `racer.validation_fraction` | 0.2 |
| `racer.inference_seed` | Optional, initializes a private sampling RNG |
| `hparam.budget` | Required positive relative-cost budget |
| `hparam.use_robust` | true; false trains ACER |
| `hparam.tau_r`, `tau_g` | 1.0, 50.0 |
| `hparam.entropy_coef` | 0.005 |
| `hparam.lr`, `dual_lr` | 0.001, 0.01 |
| `hparam.epochs`, `batch_size` | 20, 128 |
| `hparam.seed` | 42 |
| `embedding.backend` | `precomputed` or `longformer` |
| `embedding.metadata_path` | JSON containing model ID, input mode and dimension; record the encoding recipe too |

Paths inside YAML are relative to the LLMRouter package/project root, as for
other routers, or absolute. Pure performance `metric.weights` is accepted;
cost-composite weights are rejected. Use the budget parameter instead.

**Train and evaluate without external services**

From the repository root, after `pip install -e .`:

```bash
python examples/racerrouter/prepare_demo.py
llmrouter train --router racerrouter --config configs/model_config_train/racerrouter.yaml --device cpu
python examples/racerrouter/evaluate_offline.py \
  --config configs/model_config_test/racerrouter.yaml \
  --routing-data examples/racerrouter/data/test.jsonl \
  --embeddings examples/racerrouter/data/embeddings.pt
```

The demo has synthetic features and outcomes; its scores are a functionality
check, not evidence of performance on real tasks. To train ACER, copy the training
config, set `use_robust: false`, and choose a different checkpoint output path.
Changing this flag on a trained RACER checkpoint does not turn it into ACER.

For public model outcomes and BGE-M3 query features, use the
[xRouteBench example](../../../examples/racerrouter/XROUTEBENCH_EXPERIMENT.md).
The [validation report](../../../examples/racerrouter/VALIDATION.md) records
automated checks, RACER validation results, and their selection limitations.

**Inference**

```python
from llmrouter.models import RACERRouter

router = RACERRouter("configs/model_config_test/racerrouter.yaml")
probabilities = router.predict_proba([{"embedding": query_embedding}])  # [N, 2]
decision = router.route_single({"query": "...", "embedding": query_embedding})
```

`decision` includes `model_name`, `routing_probabilities` keyed by candidate ID,
and `selection_mode: "sample"`. Single and batch routes sample once per input.
`predict_proba` consumes no sampling randomness. The RNG advances across requests;
same seed and request order reproduce draws in the same execution environment.
`route_batch` only selects models, without calling either LLM. Offline evaluation
reports exact probability-weighted outcomes separately from sampled replay.

For text-only CLI/chat, train using the library's Longformer features, set
`embedding.backend: longformer`, and use matching metadata. The standard recipe
is `allenai/longformer-base-4096`, masked-mean pooling, no normalization,
max length 4096, raw text input, dimension 768. Then:

```bash
llmrouter infer --router racerrouter --config your_inference.yaml --query "Your question" --route-only
```

For live answers, also configure `data_path.llm_data` with the two API endpoints
and the usual API credentials, and omit `--route-only`. Two modes of the same
model need endpoints actually configured to implement those modes; two aliases
alone do not switch reasoning on/off.

Upstream BGE-M3 combined/separate features require compatible precomputed vectors
or a caller-owned encoder. Text-only input for these checkpoints raises an error;
it never silently uses Longformer. The current CLI passes strings, so use the
Python API/offline example for precomputed embeddings. For judge tasks the input
answers A/B are already available before routing; no candidate judge output is
needed to select the judge.

**Checkpoint and reproducibility**

A versioned checkpoint stores weights, candidate order, feature metadata, budget,
training parameters and validation summary. Inference reads no training data.
Conflicting model order, feature settings, or training budget/robustness parameters
are rejected. Endpoint configuration and sampling seed can change independently.
Adjacent `.history.json`, `.summary.json` and `.splits.json` record the training run.
Both selected-epoch and final lambda are recorded. This checkpoint supports
inference, not exact optimizer-state training resumption.

Tests cover hand-calculated gradients, RACER/ACER equivalence to the pinned
upstream training function, malformed pairs, held-out validation, persistence,
sampling and CLI registration. Run the targeted tests with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest -q tests/test_racerrouter_*.py
```

Reference fixture regeneration is a separate development step:

```bash
python tests/fixtures/racerrouter/generate_reference.py /path/to/pinned/RACER
```

The xRouteBench all-router sweep is not registered in this first integration:
it needs a budget-based sweep and raw rewards, rather than its existing
alpha/beta composite-reward preprocessing.
