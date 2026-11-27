"""Offline metrics for the probability policy, without LLM calls."""

import torch


def evaluate_policy(policy, loader, device):
    policy.eval()
    reward_sum = cost_sum = entropy_sum = 0.0
    count = batches = 0
    with torch.no_grad():
        for x, r0, r1, ratio in loader:
            x, r0, r1, ratio = (t.to(device) for t in (x, r0, r1, ratio))
            p = policy(x).squeeze(-1)
            reward_sum += ((1 - p) * r0 + p * r1).sum().item()
            cost_sum += ((1 - p) + p * ratio).sum().item()
            entropy_sum += (
                -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
            ).item()
            count += len(x)
            batches += 1
    if not count:
        raise ValueError("Validation loader is empty")
    return reward_sum / count, cost_sum / count, entropy_sum / batches


def evaluate_probabilities(probabilities, data, budget, seed=42):
    """Report exact policy expectations separately from a sampled replay."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32).cpu()
    if (
        probabilities.shape != (len(data.keys), 2)
        or not torch.isfinite(probabilities).all()
    ):
        raise ValueError("Expected finite probabilities with shape [N, 2]")
    if (probabilities < 0).any() or not torch.allclose(
        probabilities.sum(1), torch.ones(len(data.keys))
    ):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    p = probabilities[:, 1]
    reward = ((1 - p) * data.r0 + p * data.r1).mean().item()
    cost = ((1 - p) + p * data.ratio).mean().item()
    action = torch.bernoulli(p, generator=torch.Generator().manual_seed(seed))
    return {
        "num_queries": len(p),
        "expected_reward": reward,
        "expected_relative_cost": cost,
        "budget": budget,
        "budget_feasible": cost <= budget,
        "budget_violation": max(0.0, cost - budget),
        "mean_probability_b": p.mean().item(),
        "sampled_reward": ((1 - action) * data.r0 + action * data.r1).mean().item(),
        "sampled_relative_cost": ((1 - action) + action * data.ratio).mean().item(),
        "sampling_seed": seed,
        "always_a_reward": data.r0.mean().item(),
        "always_a_relative_cost": 1.0,
        "always_b_reward": data.r1.mean().item(),
        "always_b_relative_cost": data.ratio.mean().item(),
    }
