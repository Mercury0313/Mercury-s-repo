# RACER examples

RACER was introduced in the **ICML 2026** paper
[Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](https://openreview.net/forum?id=NoP2UQymHU).

See the [router guide](../../llmrouter/models/racerrouter/README.md) for the
offline synthetic demo, configuration, ACER switch and inference contracts.

To use a shard produced by the public RACER repository:

```bash
python examples/racerrouter/convert_upstream_data.py \
  --source /path/to/Qwen3-4B_instruct_Qwen3-4B_reasoning_combined.safetensors \
  --output examples/racerrouter/data/upstream \
  --candidates Qwen3-4B_instruct Qwen3-4B_reasoning \
  --dataset-name combined --separate --seed 42
```

This writes standard train/validation/test JSONL files, one unified PT embedding
file, embedding metadata and split indices. It uses the upstream 60/20/20 split.
Set the training YAML to these paths, set `racer.validation_routing_path` and
`racer.validation_embedding_path` to the explicit validation data, and use the
same two model IDs. Use `embedding.backend: precomputed` and the generated
`embedding.json`. Evaluate on `test.jsonl` with `evaluate_offline.py`.

Without `--separate`, the converter uses `embeddings_prompt_answer`. With it,
the concatenation order is prompt, answer A, answer B. The original shard lacks
text and IDs; by default the converter creates dataset-scoped row IDs and visibly
marked placeholder queries, usable only with embeddings. These placeholders are
not valid text inference inputs. `--manifest` can supply original queries/IDs in
the exact embedding row order. It must include the four original reward/token
columns as well, which are cross-checked against the shard. Do not align an
unrelated JSON file by row position or infer the order from sorted query text.

The converter preserves scores and positive output-token counts. Model-generation
failures or missing pairs must be resolved in the source data; they are not
silently replaced by zero-cost examples. No model calls or new embeddings are
generated during conversion.

For an OOD experiment, convert its shard separately and evaluate all three output
partitions or concatenate their routing records; keep its original embedding
indices. Do not select a checkpoint or tune the budget on OOD/test rewards.

To import an existing public RACER MLP checkpoint, supply the correct candidate
order and embedding metadata explicitly in a training-style config:

```bash
python examples/racerrouter/convert_upstream_checkpoint.py \
  --source /path/to/policy_mlp_B3.00_rep0.pt \
  --config /path/to/matching_metadata_config.yaml \
  --output saved_models/racerrouter/imported.pt
```

The config must not contain `model_path.load_model_path`. Budget/robustness fields,
if specified, must agree with the source checkpoint. This conversion preserves
the weights and known training metadata. It does not infer the original encoder
from its dimension, invent missing optimizer settings, or claim to recover the
lambda value of the selected epoch when the source only saved the final lambda.

## Real xRouteBench example

[XROUTEBENCH_EXPERIMENT.md](XROUTEBENCH_EXPERIMENT.md) gives pinned data and
encoder revisions, fixed splits and commands to reproduce the reported RACER
configuration. [VALIDATION.md](VALIDATION.md) records automated checks and
real-data validation results, including the tuning and evaluation limitations.
The official test split remains reserved. The scripts do not call candidate LLMs.
