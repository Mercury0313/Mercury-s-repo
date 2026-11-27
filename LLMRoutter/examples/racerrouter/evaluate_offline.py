"""Replay a trained RACER/ACER policy on paired standard JSONL observations."""

import argparse
import json
from pathlib import Path

from llmrouter.models.racerrouter import RACERRouter
from llmrouter.models.racerrouter.data import load_pairs
from llmrouter.models.racerrouter.evaluation import evaluate_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--routing-data", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument(
        "--token-column",
        help="Defaults to the training checkpoint's output-token column",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    router = RACERRouter(args.config)
    token_column = args.token_column or router.token_column
    if not token_column:
        parser.error(
            "Specify --token-column for a checkpoint without token-column metadata"
        )
    data = load_pairs(
        args.routing_data, args.embeddings, router.candidate_models, token_column
    )
    probabilities = router.predict_proba([dict(embedding=x) for x in data.x])
    result = evaluate_probabilities(
        probabilities, data, router.training_parameters["budget"], args.seed
    )
    result.update(
        method=router.training_summary["method"],
        candidate_models=router.candidate_models,
        embedding=router.embedding_config,
        training_parameters=router.training_parameters,
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
