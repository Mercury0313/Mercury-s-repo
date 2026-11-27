# RACER integration validation

Validated on CPU and NVIDIA RTX A6000 on 2026-09-07 against LLMRouter base
`da3430baaea672743c3957457b0c76faba19876e` and public RACER reference
`9d80eccb5cae14684e372bce35929c6c8a13cead`.

## Automated checks

The initial CPU run passed 37 tests. The CUDA follow-up passed all 39 tests,
including two added GPU training/checkpoint tests (RACER and ACER). These two
tests skip when CUDA is unavailable. The suite covers objective values and
gradients, numerical training
agreement with the pinned upstream implementation in both RACER and ACER modes,
partial minibatches, query pairing and validation isolation, checkpoint selection
and persistence, sampling, upstream data/checkpoint conversion, train/infer/chat
registration, mocked text encoding/API calls, and an existing SmallestLLM route.

The upstream reference fixture was generated using the unmodified upstream
modules. Its commit, settings and arrays are checked in under
`tests/fixtures/racerrouter/`; normal tests need no upstream checkout or downloads.

Run from the repository root after installing the library and pytest:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LITELLM_LOCAL_MODEL_COST_MAP=True OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_racerrouter_*.py
```

The validation environment used Python 3.11.4, torch 2.0.1, numpy 1.24.3,
pandas 2.0.3, transformers 4.40.2, LiteLLM 1.79.3, Gradio 4.44.1 and pytest 7.4.0.
These describe the tested environment; they are not new dependency pins.
The test run emitted three existing dependency warnings from LiteLLM,
PyTorch TypedStorage and Gradio's multipart import. Infeasible-budget warnings
are tested explicitly.

Also passed: `ruff check --select F,E9` on the contributed Python files,
`git diff --check`, and the existing plugin discovery/loading/CLI integration
script (`PYTHONPATH=. python tests/test_plugin_system.py`).

## Real-data validation

The [xRouteBench example](XROUTEBENCH_EXPERIMENT.md) uses published model outcomes
and BGE-M3 query embeddings. There are 3,588 training queries and 897 validation
queries. The following are means across seeds 42, 43 and 44; reward variability
is the sample standard deviation across those seeds.

| Relative-cost budget | RACER expected reward (mean ± SD) | Expected relative cost (mean) | Validation-feasible seeds |
|---|---:|---:|---:|
| 3.6890 | 0.638452 ± 0.002626 | 3.614651 | 3/3 |
| 7.7226 | 0.686178 ± 0.000653 | 7.680474 | 3/3 |
| 11.7561 | 0.719356 ± 0.001315 | 11.488979 | 3/3 |

All rows use `use_robust: true`, `tau_r: 30`, `tau_g: 500`,
`entropy_coef: 0`, `lr: 0.0003`, `dual_lr: 0.001`, `epochs: 100`, and
`batch_size: 128`. These are tuned example settings, not the algorithm defaults.
Each seed selects its checkpoint by validation reward subject to the budget.
Checkpoint reloads reproduced the saved per-query probabilities exactly in the
local validation environment.

These settings were selected after an exploratory validation search containing
369 RACER/ACER runs, with additional temperature trials for RACER. The table
reports implementation validation, not independent test performance or a fair
method-ranking benchmark. At these large temperatures the robust weights are
nearly uniform. It does not establish a robustness advantage or superiority
over ACER. The 3,729-query official test split remains unevaluated.

## GPU embedding and training follow-up

A single fixed-configuration RACER run completed on an NVIDIA RTX A6000 using
Python 3.11.4, PyTorch 2.0.1+cu118 (CUDA 11.8) and transformers 4.40.2.
BGE-M3 encoding used FP32, TF32 disabled, batch size 4 and the same feature recipe
as the CPU experiment. All 8,214 embeddings were regenerated in a separate
directory; encoding took 125.4 seconds, with zero truncations and finite,
unit-normalized vectors. Compared with the saved CPU embeddings, minimum cosine
similarity was 0.99999982 and maximum component difference was 7.33e-5 (rounded
up). Both met the preselected checks of cosine > 0.9999 and difference < 1e-4.

The standard CLI trained on GPU for 100 epochs with the same tuned settings
listed above, budget 7.7226 and seed 42. Validation selected epoch 52:

| Validation queries | Expected reward | Expected relative cost | Budget | Feasible |
|---|---:|---:|---:|---|
| 897 | 0.684875 | 7.708195 | 7.7226 | Yes |

The saved policy was loaded on CPU and moved to GPU for inference on the same
897 validation embeddings. Maximum probability difference was 5.97e-7 (rounded
up), passing `rtol=1e-5, atol=1e-6`. Saving it again from GPU and reloading on CPU
reproduced the original CPU probabilities exactly. Training history values were
finite throughout. The full automated suite passed in the CUDA environment:
39 passed, with the same three dependency warnings as the CPU run.

This checks GPU execution and checkpoint portability; it does not require CPU
and GPU optimization trajectories to select the same epoch. No additional tuning
was performed, and test metrics remain unevaluated. Reproduction instructions
are in the [GPU example](XROUTEBENCH_EXPERIMENT.md#gpu-validation).

## Scope and limits

The synthetic demo checks the training/save/load/evaluation workflow. Its default
20 epochs can return a validation-infeasible policy, which is reported with a
warning; its synthetic scores are not presented as a performance benchmark.
Real-data feasibility above concerns expected validation cost, not a guarantee
for sampled traffic or an unseen distribution.

Live candidate services and the downloaded Longformer inference path were not
exercised; their integration is covered with mocks. The entire repository's
training/inference suite, which requires other model artifacts and services,
was not run. Generated datasets, encoder weights and trained policies are
excluded from the contribution.
