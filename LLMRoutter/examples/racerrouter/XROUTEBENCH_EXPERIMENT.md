# RACER on xRouteBench

This optional example validates the router on public, precomputed model outcomes.
It trains on raw performance scores and relative output-token costs. It does not
call candidate LLMs. See [VALIDATION.md](VALIDATION.md) for measured results and
selection limitations.

## Data and features

- Dataset: `ulab-ai/xRouteBench`, configuration `llmrouter_generic`, revision
  `ea4b6e1b29d9a734f55f0a637baf326bad6aa681`.
- Candidate A: `llama-3-8b-instruct-lite`; candidate B:
  `llama-3.3-70b-instruct-turbo`. Candidate order defines the relative cost.
- Group by exact query text, excluding ambiguous, incomplete or invalid pairs.
  Two ambiguous query groups are excluded from the official training data.
  The preparation manifest records exclusions and verifies no overlap with test.
- Fixed task-stratified 80/20 train/validation split, seed 20260907:
  3,588 training queries, 897 validation queries and 3,729 reserved test queries.
- Encoder: `BAAI/bge-m3`, revision
  `5617a9f61b028005a4858fdac845db406aefb181`. Raw query text, CLS pooling,
  L2 normalization, float32, max length 2048, dimension 1024. The completed
  encoding had zero truncated queries. This general routing example uses a
  different input template from the original RACER judge experiments.
- Budgets: `1 + f * (training mean token ratio - 1)`, for f = 0.2, 0.5, 0.8,
  rounded to 3.6890, 7.7226, 11.7561. Cost is the mean of per-query ratios,
  not a ratio of aggregate tokens or dollar cost.

## Reproduce the reported RACER configuration

Run from the repository root after installing the library. Data preparation
also needs a parquet reader (for example `pyarrow`); the downloader uses
`requests`. The download includes about 2.3 GB of encoder weights, and CPU
encoding can take tens of minutes. Downloaded files are pinned by revision,
with SHA256 manifests for subsequent cache verification.

```bash
python examples/racerrouter/download_xroutebench.py
python examples/racerrouter/prepare_xroutebench.py
python examples/racerrouter/embed_xroutebench.py --threads 8 --batch-size 8
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True python examples/racerrouter/run_xroutebench.py --methods RACER --label weak_r30_g500_h0 --overrides '{"epochs":100,"lr":0.0003,"dual_lr":0.001,"entropy_coef":0.0,"tau_r":30.0,"tau_g":500.0}'
```

The final command runs three budgets and seeds 42, 43, 44. Training uses one CPU
thread by default. `--workers` enables independent worker processes. Encoding
can resume from its progress file; training consumes the immutable
`embeddings_train_ready.pt` tensor containing all train/validation features.

Outputs are written under `examples/racerrouter/data/xroutebench/` (ignored by
Git). Each `runs/<label>/b<index>/s<seed>/racer/` directory contains its config,
checkpoint, training history, validation metrics and per-query predictions.
Budget indices 0, 1 and 2 correspond to the budgets above. Average each budget's
three `result.json` files at `metrics.validation.expected_reward` and
`metrics.validation.expected_relative_cost` to reproduce the table. Check
`metrics.validation.budget_feasible` for every seed individually.

The runner also writes always-A, always-B and fixed-random validation baselines.
For a default-parameter smoke run, use `--phase smoke --methods RACER`; for the
ACER ablation use `--methods ACER` with a separate label. Completed runs are
reused only if their configuration matches. Use a fresh label when changing
parameters. The runner evaluates train and validation only; it never evaluates
the held-out test outcomes.

The reported settings were chosen through exploratory validation tuning; this
command reproduces the selected configuration, not the full search. See the
selection disclosure in [VALIDATION.md](VALIDATION.md). Use an independent test
or prespecified OOD evaluation before drawing generalization conclusions.

## GPU validation

The embedding example accepts `--device cuda` (or `cuda:N`) and uses FP32 with
TF32 disabled. Use a separate data directory so an existing CPU embedding cache
is not reused. For example, after completing the CPU data preparation:

```bash
mkdir -p examples/racerrouter/data/xroutebench_gpu_validation/prepared
cp examples/racerrouter/data/xroutebench/prepared/{queries.json,manifest.json,train.jsonl,validation.jsonl} examples/racerrouter/data/xroutebench_gpu_validation/prepared/
ln -s ../xroutebench/bge-m3 examples/racerrouter/data/xroutebench_gpu_validation/bge-m3
python examples/racerrouter/embed_xroutebench.py --data examples/racerrouter/data/xroutebench_gpu_validation --device cuda --threads 2 --batch-size 4
```

Create a training config using the GPU directory's training records,
`embeddings_train_ready.pt`, `embedding.json`, and explicit validation records.
For the single-run compatibility check, use budget 7.7226, seed 42, and the tuned
parameters above, with a separate checkpoint output path. Run it through the
standard training CLI:

```bash
llmrouter train --router racerrouter --config /path/to/gpu_config.yaml --device cuda
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest -q tests/test_racerrouter_training.py -k cuda
```

The CUDA tests exercise both RACER and ACER, GPU training, checkpoint loading
on CPU, and inference after moving the restored model back to GPU. They skip
when CUDA is unavailable. Cross-device probability comparisons use
`rtol=1e-5, atol=1e-6`; CPU and GPU training trajectories need not be identical.
The multi-run `run_xroutebench.py` remains a CPU experiment runner; use the
standard CLI above for this GPU check.
