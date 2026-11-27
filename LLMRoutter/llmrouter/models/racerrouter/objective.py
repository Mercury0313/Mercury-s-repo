"""RACER/ACER objective; preserves the public 9d80ecc training semantics."""

import torch


def objective_terms(
    p,
    r0,
    r1,
    ratio,
    dual,
    *,
    budget,
    entropy_coef,
    use_robust=True,
    tau_r=1.0,
    tau_g=50.0,
):
    p = p.reshape(-1)
    reward = (1 - p) * r0.reshape(-1) + p * r1.reshape(-1)
    cost = (1 - p) + p * ratio.reshape(-1)
    entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
    if use_robust:
        wr = torch.softmax(
            (reward.mean().detach() - reward.detach()) / max(tau_r, 1e-6), dim=0
        )
        wg = torch.softmax(
            (cost.detach() - cost.mean().detach()) / max(tau_g, 1e-6), dim=0
        )
    else:
        wr = torch.full_like(reward, 1 / reward.numel())
        wg = torch.full_like(cost, 1 / cost.numel())
    # Use mean() in ACER, as in the reference implementation.
    robust_reward = (wr * reward).sum() if use_robust else reward.mean()
    robust_cost = (wg * cost).sum() if use_robust else cost.mean()
    loss = -(
        robust_reward
        - dual * (robust_cost - budget)
        + entropy_coef * entropy
        + 0.5 * entropy_coef * dual**2
    )
    return dict(
        loss=loss,
        reward=reward,
        cost=cost,
        entropy=entropy,
        reward_weights=wr,
        cost_weights=wg,
        robust_reward=robust_reward,
        robust_cost=robust_cost,
    )


def update_dual(dual, mean_batch_cost, budget, dual_lr, entropy_coef):
    """Epoch update uses the unweighted mean of minibatch mean costs."""
    with torch.no_grad():
        return torch.clamp(
            dual + dual_lr * (mean_batch_cost - budget) - dual_lr * entropy_coef * dual,
            min=0.0,
        )
