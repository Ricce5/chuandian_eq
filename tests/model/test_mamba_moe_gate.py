import pytest
import torch


pytest.importorskip("mamba_ssm")

from src.models.bg.mamba_moe import MambaMoEBGModel


def _make_model(*, normalize_expert_gates: bool) -> MambaMoEBGModel:
    return MambaMoEBGModel(
        d_feature=1,
        scale_init=1.0,
        model_type="mamba",
        d_model=4,
        d_state=4,
        gate_activation="sigmoid",
        kernel_types=("gamma", "exp"),
        kernel_size=8,
        normalize_expert_gates=normalize_expert_gates,
        device=torch.device("cpu"),
    )


def test_normalized_expert_gate_weights_sum_to_one():
    model = _make_model(normalize_expert_gates=True)
    with torch.no_grad():
        model.gate_head.weight.zero_()
        model.gate_head.bias.copy_(torch.tensor([3.0, 1.0]))

    gate_hidden = torch.zeros(2, 5, model.d_model)
    gate_weights = model._gate_weights(gate_hidden)

    expected = torch.softmax(torch.tensor([3.0, 1.0]), dim=-1)
    assert torch.all(gate_weights >= 0.0)
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(2, 5))
    assert torch.allclose(gate_weights[0, 0], expected)


def test_legacy_sigmoid_expert_gate_weights_are_independent():
    model = _make_model(normalize_expert_gates=False)
    with torch.no_grad():
        model.gate_head.weight.zero_()
        model.gate_head.bias.copy_(torch.tensor([10.0, 10.0]))

    gate_hidden = torch.zeros(1, 3, model.d_model)
    gate_weights = model._gate_weights(gate_hidden)

    assert torch.all(gate_weights > 0.99)
    assert torch.all(gate_weights.sum(dim=-1) > 1.98)
