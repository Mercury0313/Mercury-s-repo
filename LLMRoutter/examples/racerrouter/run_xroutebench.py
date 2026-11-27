"""Run reproducible RACER/ACER validation experiments; never evaluate test."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmrouter.models.racerrouter import RACERRouter, RACERTrainer
from llmrouter.models.racerrouter.data import load_pairs
from llmrouter.models.racerrouter.evaluation import evaluate_probabilities
from llmrouter.models.racerrouter.objective import objective_terms
from llmrouter.models.racerrouter.trainer import DEFAULTS


def configure_threads(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def run_job(job):
    return run(*job)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def predict(model, data):
    with torch.no_grad():
        return torch.cat([model(x).reshape(-1) for x in data.x.split(256)])


def metrics(p, data, budget, seed):
    result = evaluate_probabilities(torch.stack((1 - p, p), dim=1), data, budget, seed)
    result.update(
        probability_std=p.std().item(),
        fraction_p_below_001=(p < 0.01).float().mean().item(),
        fraction_p_above_099=(p > 0.99).float().mean().item(),
        entropy=(-(p * (p + 1e-8).log() + (1 - p) * (1 - p + 1e-8).log()))
        .mean()
        .item(),
    )
    return result


def diagnostics(model, data, hp, dual):
    generator = torch.Generator().manual_seed(hp["seed"])
    order = torch.randperm(len(data.keys), generator=generator)
    rows = []
    for indexes in order.split(hp["batch_size"]):
        with torch.no_grad():
            p = model(data.x[indexes])
            terms = objective_terms(
                p,
                data.r0[indexes],
                data.r1[indexes],
                data.ratio[indexes],
                torch.tensor(dual),
                **{
                    k: hp[k]
                    for k in ("budget", "entropy_coef", "use_robust", "tau_r", "tau_g")
                },
            )
            n = len(indexes)
            rows.append(
                dict(
                    reward_ess_fraction=(
                        1 / terms["reward_weights"].square().sum() / n
                    ).item(),
                    cost_ess_fraction=(
                        1 / terms["cost_weights"].square().sum() / n
                    ).item(),
                    robust_cost=terms["robust_cost"].item(),
                    cost=terms["cost"].mean().item(),
                    robust_reward=terms["robust_reward"].item(),
                    reward=terms["reward"].mean().item(),
                )
            )
    result = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    # Compare the objectives at the SAME selected policy and dual value.
    indexes = order[: hp["batch_size"]]
    gradients = []
    for robust in (False, True):
        model.zero_grad(set_to_none=True)
        terms = objective_terms(
            model(data.x[indexes]),
            data.r0[indexes],
            data.r1[indexes],
            data.ratio[indexes],
            torch.tensor(dual),
            budget=hp["budget"],
            entropy_coef=hp["entropy_coef"],
            use_robust=robust,
            tau_r=hp["tau_r"],
            tau_g=hp["tau_g"],
        )
        gradients.append(
            torch.cat(
                [
                    g.reshape(-1)
                    for g in torch.autograd.grad(terms["loss"], model.parameters())
                ]
            )
        )
    result["acer_racer_gradient_cosine"] = torch.nn.functional.cosine_similarity(
        *gradients, dim=0
    ).item()
    result["acer_gradient_norm"] = gradients[0].norm().item()
    result["racer_gradient_norm"] = gradients[1].norm().item()
    return result


def run(
    args, label, overrides, budget_index, seed, method, manifest, prepared, embeddings
):
    budget = manifest["budgets"][budget_index]
    dest = args.data / "runs" / label / f"b{budget_index}" / f"s{seed}" / method.lower()
    dest.mkdir(parents=True, exist_ok=True)
    hp = dict(DEFAULTS, **overrides)
    hp.update(budget=budget, seed=seed, use_robust=method == "RACER")
    config = dict(
        data_path=dict(
            routing_data_train=str(prepared / "train.jsonl"),
            query_embedding_data=str(embeddings),
        ),
        model_path=dict(save_model_path=str(dest / "policy.pt")),
        racer=dict(
            candidate_models=manifest["candidate_models"],
            token_column="output_tokens",
            cost_mode="relative_output_tokens",
            inference_seed=seed,
            validation_routing_path=str(prepared / "validation.jsonl"),
            validation_embedding_path=str(embeddings),
        ),
        embedding=dict(
            backend="precomputed", metadata_path=str(prepared / "embedding.json")
        ),
        hparam=hp,
    )
    config_path = dest / "config.yaml"
    result_path = dest / "result.json"
    if result_path.exists():
        assert yaml.safe_load(config_path.read_text()) == config, (
            "Refusing to reuse changed run configuration"
        )
        print(
            f"Verified completed {label}/b{budget_index}/s{seed}/{method}", flush=True
        )
        return json.loads(result_path.read_text())
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.monotonic()
    router = RACERRouter(str(config_path))
    trainer = RACERTrainer(router, device="cpu")
    trainer.train()
    selected_epoch = router.training_summary["best_epoch"]
    # The saved lambda is AFTER the epoch update; policy gradients used the
    # preceding epoch's lambda (zero in epoch 1).
    training_lambda = (
        0.0 if selected_epoch == 1 else trainer.history[selected_epoch - 2]["lambda"]
    )
    all_metrics = {}
    for split in ("train", "validation"):
        data = load_pairs(
            prepared / f"{split}.jsonl",
            embeddings,
            manifest["candidate_models"],
            "output_tokens",
        )
        p = predict(router.model, data)
        all_metrics[split] = metrics(p, data, budget, seed)
        np.savez_compressed(
            dest / f"{split}_predictions.npz",
            p=p.numpy(),
            keys=np.array(data.keys),
            r0=data.r0.numpy(),
            r1=data.r1.numpy(),
            ratio=data.ratio.numpy(),
        )
        if split == "train":
            diag = diagnostics(router.model, data, hp, training_lambda)
        else:
            records = [
                json.loads(line)
                for line in (prepared / "validation.jsonl").read_text().splitlines()
            ]
            task_by_id = {r["query_id"]: r["task_name"] for r in records}
            tasks = [task_by_id[json.loads(key)[1]] for key in data.keys]
            reward = (1 - p) * data.r0 + p * data.r1
            cost = (1 - p) + p * data.ratio
            all_metrics["tasks"] = {}
            for task in sorted(set(tasks)):
                mask = torch.tensor([t == task for t in tasks])
                all_metrics["tasks"][task] = dict(
                    n=int(mask.sum()),
                    reward=reward[mask].mean().item(),
                    cost=cost[mask].mean().item(),
                    probability_b=p[mask].mean().item(),
                )
    result = dict(
        label=label,
        method=method,
        seed=seed,
        budget_index=budget_index,
        hparam=hp,
        summary=router.training_summary,
        metrics=all_metrics,
        diagnostics=diag,
        training_trace=dict(
            selected_epoch_training_lambda=training_lambda,
            feasible_epochs=sum(r["val_cost"] <= budget for r in trainer.history),
            zero_dual_epochs=sum(r["lambda"] == 0 for r in trainer.history),
            min_validation_cost=min(r["val_cost"] for r in trainer.history),
            max_validation_cost=max(r["val_cost"] for r in trainer.history),
        ),
        elapsed_seconds=time.monotonic() - started,
        evaluation_split="validation",
        test_evaluated=False,
    )
    write_json(result_path, result)
    print(
        json.dumps(
            dict(
                run=f"{label}/b{budget_index}/s{seed}/{method}",
                **{
                    k: all_metrics["validation"][k]
                    for k in (
                        "expected_reward",
                        "expected_relative_cost",
                        "budget_feasible",
                        "mean_probability_b",
                    )
                },
                seconds=round(result["elapsed_seconds"], 1),
            )
        ),
        flush=True,
    )
    return result


def baselines(args, manifest, prepared, embeddings):
    data = load_pairs(
        prepared / "validation.jsonl",
        embeddings,
        manifest["candidate_models"],
        "output_tokens",
    )
    rows = []
    for index, budget in enumerate(manifest["budgets"]):
        fixed_p = min(
            1.0,
            max(
                0.0, (budget - 1) / (manifest["training_statistics"]["mean_ratio"] - 1)
            ),
        )
        for method, p in (
            ("Always A", 0.0),
            ("Always B", 1.0),
            ("Fixed random", fixed_p),
        ):
            rows.append(
                dict(
                    method=method,
                    budget_index=index,
                    fixed_p=p,
                    metrics=metrics(torch.full((len(data.keys),), p), data, budget, 42),
                )
            )
    write_json(args.data / "baselines_validation.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "data/xroutebench"
    )
    parser.add_argument("--phase", choices=["smoke", "full"], default="full")
    parser.add_argument("--label", default="reference")
    parser.add_argument("--overrides", type=json.loads, default={})
    parser.add_argument(
        "--grid",
        type=Path,
        help="JSON list of {label, overrides}; all attempts are retained",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--budgets", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--methods", nargs="+", choices=["RACER", "ACER"], default=["ACER", "RACER"]
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.data = args.data.resolve()
    configure_threads(args.threads)
    prepared = args.data / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text())
    # The immutable train-ready tensor is usable while held-out inputs encode.
    embeddings = prepared / "embeddings_train_ready.pt"
    if not embeddings.exists():
        raise FileNotFoundError("Train/validation embeddings are not ready yet")
    baselines(args, manifest, prepared, embeddings)
    plans = (
        json.loads(args.grid.read_text())
        if args.grid
        else [dict(label=args.label, overrides=args.overrides)]
    )
    budgets, seeds = (
        ([1], [42]) if args.phase == "smoke" else (args.budgets, args.seeds)
    )
    jobs = [
        (
            args,
            plan["label"],
            plan["overrides"],
            index,
            seed,
            method,
            manifest,
            prepared,
            embeddings,
        )
        for plan in plans
        for index in budgets
        for seed in seeds
        for method in args.methods
    ]
    if args.workers == 1:
        for job in jobs:
            run_job(job)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=configure_threads,
            initargs=(args.threads,),
        ) as executor:
            list(executor.map(run_job, jobs))


if __name__ == "__main__":
    main()
