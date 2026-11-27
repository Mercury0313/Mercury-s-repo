import argparse
import os
import torch
from llmrouter.models import TSRouter
from llmrouter.models import TSRouterTrainer


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_yaml = os.path.join(project_root, "configs", "model_config_train", "tsrouter.yaml")

    parser = argparse.ArgumentParser(
        description="Train and test TSRouter with a YAML configuration file."
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=default_yaml,
        help=f"Path to the YAML config file (default: {default_yaml})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML file not found: {args.yaml_path}")

    print(f"Using YAML file: {args.yaml_path}")
    router = TSRouter(args.yaml_path)
    print("TSRouter wrapper initialized.")

    # Train (builds graph + trains GNN)
    trainer = TSRouterTrainer(router=router, device=args.device)
    trainer.train()

    # Quick sanity check: route the internal test set
    results = router.route_batch()
    print(f"Routed {len(results)} test queries.")
    if results:
        print(f"Example result: {results[0]}")


if __name__ == "__main__":
    main()
