"""Train an MF router from local pairwise JSONL data and prompt embeddings."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel
from routellm.routers.matrix_factorization.train_matrix_factorization import (
    MFModel_Train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-jsonl", type=Path, required=True)
    parser.add_argument("--embeddings-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--noise-alpha", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pairwise_data(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add_quality_targets(data: list[dict]) -> None:
    for sample in data:
        quality_a = float(sample["quality_score_a"])
        quality_b = float(sample["quality_score_b"])
        if not math.isfinite(quality_a) or not math.isfinite(quality_b):
            raise ValueError("Quality scores must be finite.")
        sample["quality_target"] = (
            1.0 if quality_a > quality_b else 0.0 if quality_a < quality_b else 0.5
        )


def validate_inputs(data: list[dict], embeddings: np.ndarray) -> None:
    if not data:
        raise ValueError("Pairwise data is empty.")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(data):
        raise ValueError(
            "Embedding rows must equal the number of pairwise samples: "
            f"{embeddings.shape} vs {len(data)}."
        )
    if embeddings.shape[1] != 1536:
        raise ValueError(
            f"Expected 1536-dimensional embeddings, got {embeddings.shape[1]}."
        )

    expected_indices = set(range(len(data)))
    actual_indices = {sample.get("idx") for sample in data}
    if actual_indices != expected_indices:
        raise ValueError("Sample idx values must be unique and span 0..N-1.")

    for sample in data:
        if sample["quality_target"] not in {0.0, 0.5, 1.0}:
            raise ValueError(f"Invalid quality_target: {sample['quality_target']!r}")
        for model_key in ("model_a", "model_b"):
            model_name = sample.get(model_key)
            if model_name not in MODEL_IDS:
                raise ValueError(f"Model is missing from MODEL_IDS: {model_name!r}")


class FixedOrderPairwiseDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.models_a = torch.tensor(
            [MODEL_IDS[sample["model_a"]] for sample in data], dtype=torch.int64
        )
        self.models_b = torch.tensor(
            [MODEL_IDS[sample["model_b"]] for sample in data], dtype=torch.int64
        )
        self.prompt_ids = torch.tensor(
            [sample["idx"] for sample in data], dtype=torch.int64
        )
        self.quality_targets = torch.tensor(
            [sample["quality_target"] for sample in data], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.models_a)

    def __getitem__(self, index: int):
        return (
            self.models_a[index],
            self.models_b[index],
            self.prompt_ids[index],
            self.quality_targets[index],
        )


def stratified_split(
    data: list[dict], validation_fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in data:
        grouped[str(sample["quality_target"])].append(sample)

    rng = random.Random(seed)
    train_data, validation_data = [], []
    for samples in grouped.values():
        rng.shuffle(samples)
        validation_count = max(1, round(len(samples) * validation_fraction))
        if validation_count == len(samples):
            raise ValueError("Each winner class needs at least two samples.")
        validation_data.extend(samples[:validation_count])
        train_data.extend(samples[validation_count:])
    rng.shuffle(train_data)
    rng.shuffle(validation_data)
    return train_data, validation_data


@torch.no_grad()
def evaluate(model: MFModel_Train, data_loader: DataLoader, device: str) -> dict:
    model.eval()
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    decisive_correct = 0
    decisive_samples = 0
    tie_deviation = 0.0
    tie_samples = 0

    for models_a, models_b, prompt_ids, quality_targets in data_loader:
        quality_targets = quality_targets.to(device)
        logits = model(models_a, models_b, prompt_ids, test=True)
        total_loss += loss_function(logits, quality_targets).item()
        total_samples += len(quality_targets)

        decisive_mask = quality_targets != 0.5
        if decisive_mask.any():
            predictions = (logits[decisive_mask] > 0).float()
            decisive_correct += (
                predictions == quality_targets[decisive_mask]
            ).sum().item()
            decisive_samples += decisive_mask.sum().item()

        tie_mask = quality_targets == 0.5
        if tie_mask.any():
            tie_deviation += torch.abs(
                torch.sigmoid(logits[tie_mask]) - 0.5
            ).sum().item()
            tie_samples += tie_mask.sum().item()

    model.train()
    return {
        "loss": total_loss / total_samples,
        "decisive_accuracy": (
            decisive_correct / decisive_samples if decisive_samples else None
        ),
        "tie_probability_deviation": tie_deviation / tie_samples if tie_samples else None,
    }


def train_model(
    model: MFModel_Train,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: str,
) -> dict:
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss()
    validation_metrics = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for models_a, models_b, prompt_ids, quality_targets in train_loader:
            quality_targets = quality_targets.to(device)
            logits = model(
                models_a, models_b, prompt_ids, alpha=args.noise_alpha
            )
            loss = loss_function(logits, quality_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(quality_targets)
            total_samples += len(quality_targets)

        validation_metrics = evaluate(model, validation_loader, device)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / total_samples,
                    "validation": validation_metrics,
                }
            )
        )

    return validation_metrics


def export_inference_checkpoint(
    trained_model: MFModel_Train, output_dir: Path, text_dim: int
) -> None:
    inference_model = MFModel(
        dim=trained_model.P.embedding_dim,
        num_models=len(MODEL_IDS),
        text_dim=text_dim,
        num_classes=1,
        use_proj=True,
    )
    with torch.no_grad():
        inference_model.P.weight.copy_(trained_model.P.weight.detach().cpu())
        inference_model.text_proj[0].weight.copy_(
            trained_model.text_proj.weight.detach().cpu()
        )
        inference_model.classifier[0].weight.copy_(
            trained_model.classifier.weight.detach().cpu()
        )
    inference_model.save_pretrained(str(output_dir))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_pairwise_data(args.pairwise_jsonl)
    add_quality_targets(data)
    embeddings = np.load(args.embeddings_npy)
    validate_inputs(data, embeddings)
    train_data, validation_data = stratified_split(
        data, args.validation_fraction, args.seed
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MFModel_Train(
        dim=128,
        num_models=len(MODEL_IDS),
        num_prompts=len(data),
        text_dim=embeddings.shape[1],
        use_proj=True,
        npy_path=args.embeddings_npy,
    ).to(device)
    train_loader = DataLoader(
        FixedOrderPairwiseDataset(train_data), args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        FixedOrderPairwiseDataset(validation_data), args.batch_size, shuffle=False
    )
    validation_metrics = train_model(
        model,
        train_loader,
        validation_loader,
        args,
        device,
    )

    args.output_dir.mkdir(parents=True)
    export_inference_checkpoint(model, args.output_dir, embeddings.shape[1])
    reloaded = MFModel.from_pretrained(
        str(args.output_dir),
        dim=128,
        num_models=len(MODEL_IDS),
        text_dim=embeddings.shape[1],
        num_classes=1,
        use_proj=True,
    )
    if not torch.allclose(reloaded.P.weight, model.P.weight.detach().cpu()):
        raise RuntimeError("Checkpoint reload verification failed.")

    summary = {
        "device": device,
        "embedding_shape": list(embeddings.shape),
        "num_models": len(MODEL_IDS),
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "quality_target_counts": Counter(
            str(sample["quality_target"]) for sample in data
        ),
        "routing_preference_counts": Counter(sample["winner"] for sample in data),
        "validation": validation_metrics,
        "seed": args.seed,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Checkpoint written to: {args.output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
