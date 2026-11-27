"""Attach explicit candidate/embedding metadata to an upstream MLP checkpoint.

Use a training-style YAML with racer.candidate_models, token_column and embedding
metadata, without model_path.load_model_path. No training data is read.
"""

import argparse

import torch

from llmrouter.models.racerrouter import RACERRouter


def convert(source, config, output):
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("policy_type", "mlp") != "mlp":
        raise ValueError("Only the public RACER MLP architecture is supported")
    router = RACERRouter(str(config))
    if router.ready:
        raise ValueError("Conversion config must not specify load_model_path")
    state = checkpoint["state_dict"]
    router.initialize_policy(state["net.0.weight"].shape[1])
    router.model.load_state_dict(state, strict=True)
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("Source checkpoint contains non-finite weights")
    parameters = {k: checkpoint[k] for k in ("budget", "use_robust", "tau_r", "tau_g")}
    for key, value in parameters.items():
        configured = router.cfg.get("hparam", {}).get(key)
        if configured is not None and configured != value:
            raise ValueError(f"Configured {key} differs from source checkpoint")
    # Preserve known metadata; original checkpoints omit optimizer settings.
    router.training_parameters = parameters
    cost = float(checkpoint["best_val_cost"])
    budget = float(checkpoint["budget"])
    router.training_summary = dict(
        method="RACER" if parameters["use_robust"] else "ACER",
        best_epoch=checkpoint["best_epoch"],
        best_val_reward=checkpoint["best_val_reward"],
        best_val_cost=cost,
        budget=budget,
        budget_feasible=cost <= budget,
        budget_violation=max(0.0, cost - budget),
        selected_epoch_lambda=None,
        final_lambda=checkpoint.get("lambda_dual"),
        imported_upstream_weights=True,
    )
    router.ready = True
    router.save_router(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.source, args.config, args.output)


if __name__ == "__main__":
    main()
