"""Developer-only: regenerate fixtures from a pinned local upstream checkout.

Usage: python tests/fixtures/racerrouter/generate_reference.py /path/to/RACER
No network access. Normal tests read the generated JSON/NPZ files directly.
"""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch


def main():
    root = Path(sys.argv[1]).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    assert commit == "9d80eccb5cae14684e372bce35929c6c8a13cead", commit
    # Load unmodified source modules without importing the upstream plotting
    # and data-generation convenience exports. No algorithm is substituted.
    package = types.ModuleType("racer")
    package.__path__ = [str(root / "racer")]
    sys.modules["racer"] = package
    PolicyNet = importlib.import_module("racer.policy").PolicyNet
    train = importlib.import_module("racer.trainer").train_dual_ascent
    torch.set_num_threads(1)
    hp = dict(
        budget=1.6,
        entropy_coef=0.005,
        epochs=4,
        lr=0.002,
        dual_lr=0.1,
        tau_r=0.3,
        tau_g=0.5,
    )
    x = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.5, 1.0, 0.5],
            [1.0, -1.0, 0.0],
        ]
    )
    values = (
        x,
        torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
        torch.ones(5),
        torch.tensor([3.0, 2.0, 5.0, 4.0, 7.0]),
    )
    batches = [
        tuple(v[start : start + 2].clone() for v in values) for start in (0, 2, 4)
    ]
    validation = [tuple(v.flip(0).clone() for v in values)]
    model = PolicyNet(3)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(
                (((torch.arange(parameter.numel()) + index) % 17 - 8) / 50).reshape(
                    parameter.shape
                )
            )
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    arrays = {f"initial/{k}": v.numpy() for k, v in initial.items()}
    for i, batch in enumerate(batches):
        for j, value in enumerate(batch):
            arrays[f"train/{i}/{j}"] = value.numpy()
    for j, value in enumerate(validation[0]):
        arrays[f"val/{j}"] = value.numpy()
    reference = dict(
        source_commit=commit,
        torch_version=str(torch.__version__),
        dtype="float32",
        hparam=hp,
        initialization="((arange + parameter_index) % 17 - 8) / 50",
    )
    for robust in (False, True):
        model = PolicyNet(3)
        model.load_state_dict(initial)
        policy, dual, history, best = train(
            model,
            (batches, validation),
            use_robust=robust,
            device="cpu",
            log_interval=0,
            **hp,
        )
        case = "racer" if robust else "acer"
        reference[case] = dict(
            best_epoch=best["epoch"],
            final_lambda=dual.item(),
            feasible=best["val_cost"] <= hp["budget"],
            history=[
                {k: v for k, v in row.items() if k != "state_dict"} for row in history
            ],
        )
        arrays.update(
            {f"{case}/{k}": v.numpy() for k, v in policy.state_dict().items()}
        )
    target = Path(__file__).parent
    (target / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    np.savez_compressed(target / "reference.npz", **arrays)


if __name__ == "__main__":
    main()
