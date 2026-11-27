import pytest
import torch

from llmrouter.models.racerrouter.objective import objective_terms, update_dual


def test_hand_computed_expectations():
    result = objective_terms(
        torch.tensor([0.25]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor([5.0]),
        torch.tensor(0.0),
        budget=2,
        entropy_coef=0,
        use_robust=False,
    )
    assert result["reward"].item() == pytest.approx(0.25)
    assert result["cost"].item() == pytest.approx(2)


def test_detached_robust_weights_and_gradient():
    p = torch.tensor([0.2, 0.8], requires_grad=True)
    result = objective_terms(
        p,
        torch.zeros(2),
        torch.ones(2),
        torch.tensor([2.0, 5.0]),
        torch.tensor(0.7),
        budget=2,
        entropy_coef=0,
        tau_r=0.5,
        tau_g=0.4,
    )
    # Independently computed from r=[0.2,0.8], c=[1.2,4.2].
    wr = torch.softmax(torch.tensor([-0.2, -0.8]) / 0.5, 0)
    wg = torch.softmax(torch.tensor([1.2, 4.2]) / 0.4, 0)
    torch.testing.assert_close(result["reward_weights"], wr)
    torch.testing.assert_close(result["cost_weights"], wg)
    assert not result["cost_weights"].requires_grad
    result["loss"].backward()
    torch.testing.assert_close(p.grad, -wr + 0.7 * wg * torch.tensor([1.0, 4.0]))


def test_acer_ignores_temperatures():
    args = (
        torch.tensor([0.2, 0.8]),
        torch.zeros(2),
        torch.ones(2),
        torch.tensor([2.0, 5.0]),
        torch.tensor(0.7),
    )
    first = objective_terms(
        *args, budget=2, entropy_coef=0.005, use_robust=False, tau_r=None, tau_g=None
    )
    second = objective_terms(
        *args, budget=2, entropy_coef=0.005, use_robust=False, tau_r=1e-3, tau_g=1e3
    )
    torch.testing.assert_close(first["loss"], second["loss"])
    assert first["robust_reward"].item() == pytest.approx(0.5)
    assert first["robust_cost"].item() == pytest.approx(2.7)


def test_dual_shrinkage_and_projection():
    assert update_dual(torch.tensor(1.0), 4, 2, 0.1, 0.2).item() == pytest.approx(1.18)
    assert update_dual(torch.tensor(0.1), 0, 2, 1, 0.2).item() == 0
