"""Budget-constrained trainer matching onepounchman/RACER at 9d80ecc."""

import json
import math
import warnings

import torch
from torch.utils.data import DataLoader, random_split

from llmrouter.models.base_trainer import BaseTrainer
from .data import load_pairs, resolve_path
from .evaluation import evaluate_policy
from .objective import objective_terms, update_dual


DEFAULTS = dict(
    use_robust=True,
    tau_r=1.0,
    tau_g=50.0,
    entropy_coef=0.005,
    lr=1e-3,
    dual_lr=1e-2,
    epochs=20,
    batch_size=128,
    seed=42,
)


def validate_hparams(parameters):
    hp = dict(DEFAULTS, **parameters)
    if not isinstance(hp["use_robust"], bool):
        raise ValueError("use_robust must be a boolean")
    for key in ("budget", "lr", "dual_lr", "entropy_coef") + (
        ("tau_r", "tau_g") if hp["use_robust"] else ()
    ):
        value = hp.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (key != "entropy_coef" and value == 0)
        ):
            raise ValueError(
                f"hparam.{key} must be a finite {'nonnegative' if key == 'entropy_coef' else 'positive'} number"
            )
    for key in ("epochs", "batch_size", "seed"):
        value = hp[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"hparam.{key} must be a valid integer")
    return hp


class RACERTrainer(BaseTrainer):
    def __init__(self, router, optimizer=None, device="cpu", **kwargs):
        if optimizer is not None:
            raise ValueError("RACER uses the reference AdamW optimizer")
        super().__init__(router=router, device=device, **kwargs)
        self.hp = validate_hparams(router.cfg.get("hparam", {}))
        config = router.racer_config
        if (
            config.get("cost_mode", "relative_output_tokens")
            != "relative_output_tokens"
        ):
            raise ValueError("Only relative_output_tokens cost is supported")
        weights = router.cfg.get("metric", {}).get("weights", {})
        if any(
            value != (1 if key == "performance" else 0)
            for key, value in weights.items()
        ):
            raise ValueError(
                "RACER needs raw performance; use budget rather than metric.weights"
            )
        self.history = []
        self.split_manifest = {}

    def _loaders(self):
        router, hp = self.router, self.hp
        paths = router.cfg.get("data_path", {})
        token_column = router.racer_config.get("token_column")
        if not isinstance(token_column, str) or not token_column:
            raise ValueError("racer.token_column must name the output-token column")
        data = load_pairs(
            paths["routing_data_train"],
            paths["query_embedding_data"],
            router.candidate_models,
            token_column,
        )
        validation_path = router.racer_config.get("validation_routing_path")
        validation_embeddings = router.racer_config.get("validation_embedding_path")
        if bool(validation_path) != bool(validation_embeddings):
            raise ValueError(
                "Provide both validation_routing_path and validation_embedding_path"
            )
        if validation_path:
            validation = load_pairs(
                validation_path,
                validation_embeddings,
                router.candidate_models,
                token_column,
            )
            if set(data.keys) & set(validation.keys):
                raise ValueError("Training and validation queries overlap")
            if data.x.shape[1] != validation.x.shape[1]:
                raise ValueError("Training/validation embedding dimensions differ")
            training_set, validation_set = data.dataset(), validation.dataset()
            training_keys, validation_keys = data.keys, validation.keys
        else:
            fraction = router.racer_config.get("validation_fraction", 0.2)
            if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
                raise ValueError("validation_fraction must be between 0 and 1")
            n_val = max(1, int(len(data.keys) * fraction))
            if n_val >= len(data.keys):
                raise ValueError("At least two complete query pairs are needed")
            training_set, validation_set = random_split(
                data.dataset(),
                [len(data.keys) - n_val, n_val],
                generator=torch.Generator().manual_seed(hp["seed"]),
            )
            training_keys = [data.keys[i] for i in training_set.indices]
            validation_keys = [data.keys[i] for i in validation_set.indices]
        self.split_manifest = dict(
            train=training_keys,
            validation=validation_keys,
            seed=hp["seed"],
            filtered_training_rows=data.filtered_rows,
        )
        router.initialize_policy(data.x.shape[1], hp["seed"])
        return (
            DataLoader(
                training_set,
                batch_size=hp["batch_size"],
                shuffle=True,
                generator=torch.Generator().manual_seed(hp["seed"]),
            ),
            DataLoader(validation_set, batch_size=256, shuffle=False),
        )

    def loss_func(self, outputs, batch):
        _, r0, r1, ratio = batch
        return self._terms(outputs, r0, r1, ratio)["loss"]

    def _terms(self, p, r0, r1, ratio):
        return objective_terms(
            p,
            r0,
            r1,
            ratio,
            self.dual,
            **{
                k: self.hp[k]
                for k in ("budget", "entropy_coef", "use_robust", "tau_r", "tau_g")
            },
        )

    def train(self, dataloader=None):
        """Train from configuration, or explicit (train, validation) loaders.

        Explicit loaders require an initialized policy and are useful when
        reproducing upstream batch order. Training always starts a fresh optimizer.
        """
        hp, router = self.hp, self.router
        train_loader, val_loader = self._loaders() if dataloader is None else dataloader
        if router.input_dim is None:
            raise ValueError("Initialize the policy before passing explicit loaders")
        if not len(train_loader) or not len(val_loader):
            raise ValueError("Training and validation loaders must be nonempty")
        router.ready = False
        router.model.to(self.device)
        self.optimizer = torch.optim.AdamW(router.model.parameters(), lr=hp["lr"])
        self.dual = torch.tensor(0.0, device=self.device)
        self.history = []
        best_feasible = closest = None
        for epoch in range(hp["epochs"]):
            router.model.train()
            totals = dict(reward=0.0, cost=0.0, entropy=0.0)
            for batch in train_loader:
                x, r0, r1, ratio = (t.to(self.device) for t in batch)
                terms = self._terms(router.model(x), r0, r1, ratio)
                if not torch.isfinite(terms["loss"]):
                    raise ValueError("Non-finite RACER training loss")
                self.optimizer.zero_grad()
                terms["loss"].backward()
                self.optimizer.step()
                for key in totals:
                    totals[key] += terms[key].mean().item()
            averages = {key: value / len(train_loader) for key, value in totals.items()}
            self.dual = update_dual(
                self.dual,
                averages["cost"],
                hp["budget"],
                hp["dual_lr"],
                hp["entropy_coef"],
            )
            val_reward, val_cost, val_entropy = evaluate_policy(
                router.model, val_loader, self.device
            )
            row = dict(
                epoch=epoch + 1,
                train_reward=averages["reward"],
                train_cost=averages["cost"],
                train_entropy=averages["entropy"],
                val_reward=val_reward,
                val_cost=val_cost,
                val_entropy=val_entropy,
                **{"lambda": self.dual.item()},
            )
            self.history.append(row)
            feasible = val_cost <= hp["budget"]
            improve_feasible = feasible and (
                best_feasible is None or val_reward > best_feasible[0]["val_reward"]
            )
            improve_closest = closest is None or abs(val_cost - hp["budget"]) < abs(
                closest[0]["val_cost"] - hp["budget"]
            )
            if improve_feasible or improve_closest:
                saved = (
                    row.copy(),
                    {
                        k: v.detach().cpu().clone()
                        for k, v in router.model.state_dict().items()
                    },
                )
                if improve_feasible:
                    best_feasible = saved
                if improve_closest:
                    closest = saved
        best, state = best_feasible or closest
        router.model.load_state_dict(state)
        router.model.eval()
        router.training_parameters = hp.copy()
        router.training_summary = dict(
            method="RACER" if hp["use_robust"] else "ACER",
            best_epoch=best["epoch"],
            best_val_reward=best["val_reward"],
            best_val_cost=best["val_cost"],
            budget=hp["budget"],
            budget_feasible=best_feasible is not None,
            budget_violation=max(0.0, best["val_cost"] - hp["budget"]),
            selected_epoch_lambda=best["lambda"],
            final_lambda=self.dual.item(),
        )
        router.ready = True
        if best_feasible is None:
            warnings.warn(
                "No checkpoint met the validation budget; returning the closest-cost epoch",
                UserWarning,
            )
        path = router.cfg.get("model_path", {}).get("save_model_path")
        if path:
            router.save_router(path)
            for suffix, value in (
                ("history", self.history),
                ("summary", router.training_summary),
                ("splits", self.split_manifest),
            ):
                with (
                    resolve_path(path)
                    .with_suffix(f".{suffix}.json")
                    .open("w") as handle
                ):
                    json.dump(value, handle, indent=2, allow_nan=False)
        print(
            f"[{router.training_summary['method']}] epoch={best['epoch']} "
            f"val_reward={best['val_reward']:.4f} val_cost={best['val_cost']:.4f} "
            f"budget={hp['budget']} feasible={best_feasible is not None}"
        )
        return router.training_summary
