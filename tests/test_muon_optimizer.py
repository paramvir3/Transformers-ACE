import torch

from flashace.optim import (
    MuonWithAuxAdamW,
    adam_update,
    get_muon_param_groups,
    muon_update,
    zeropower_via_newtonschulz5,
)


def _reference_zeropower(matrix, steps):
    assert matrix.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    update = matrix
    if matrix.size(-2) > matrix.size(-1):
        update = update.mT
    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = update @ update.mT
        update = a * update + (b * gram + c * gram @ gram) @ update
    if matrix.size(-2) > matrix.size(-1):
        update = update.mT
    return update


def _reference_muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = _reference_zeropower(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update.reshape_as(grad)


def _reference_adam_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    exp_avg_corrected = exp_avg / (1 - betas[0] ** step)
    exp_avg_sq_corrected = exp_avg_sq / (1 - betas[1] ** step)
    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.norm = torch.nn.LayerNorm(3)
        self.vector_weight = torch.nn.Parameter(torch.ones(3))

    def forward(self, x):
        y = self.norm(self.linear(x))
        return (y * self.vector_weight).sum()


def test_muon_param_groups_cover_every_trainable_parameter_once():
    model = TinyModel()
    groups = get_muon_param_groups(
        model,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        parameter_mode="all_matrices",
    )

    grouped = [param for group in groups for param in group["params"]]
    assert len(grouped) == len({id(param) for param in grouped})
    assert {id(param) for param in grouped} == {
        id(param) for param in model.parameters() if param.requires_grad
    }

    muon_groups = [group for group in groups if group["use_muon"]]
    aux_groups = [group for group in groups if not group["use_muon"]]
    assert len(muon_groups) == 1
    assert aux_groups
    assert any(param is model.linear.weight for param in muon_groups[0]["params"])
    assert not any(param is model.linear.bias for param in muon_groups[0]["params"])


def test_newton_schulz_matches_nequix_reference_update():
    torch.manual_seed(1)
    matrix = torch.randn(7, 5)
    expected = _reference_zeropower(matrix, steps=5)
    actual = zeropower_via_newtonschulz5(matrix, steps=5)

    torch.testing.assert_close(actual, expected)


def test_muon_update_matches_nequix_reference_update():
    torch.manual_seed(2)
    grad = torch.randn(8, 6)
    momentum = torch.randn_like(grad) * 0.1
    ref_momentum = momentum.clone()

    expected = _reference_muon_update(grad.clone(), ref_momentum, beta=0.95, ns_steps=5)
    actual = muon_update(grad.clone(), momentum, beta=0.95, ns_steps=5)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(momentum, ref_momentum)


def test_aux_adam_update_matches_nequix_reference_update():
    torch.manual_seed(3)
    grad = torch.randn(5)
    exp_avg = torch.randn(5) * 0.1
    exp_avg_sq = torch.rand(5) * 0.01
    ref_exp_avg = exp_avg.clone()
    ref_exp_avg_sq = exp_avg_sq.clone()

    expected = _reference_adam_update(
        grad,
        ref_exp_avg,
        ref_exp_avg_sq,
        step=4,
        betas=(0.9, 0.95),
        eps=1.0e-10,
    )
    actual = adam_update(
        grad,
        exp_avg,
        exp_avg_sq,
        step=4,
        betas=(0.9, 0.95),
        eps=1.0e-10,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(exp_avg, ref_exp_avg)
    torch.testing.assert_close(exp_avg_sq, ref_exp_avg_sq)


def test_muon_optimizer_steps_matrix_and_auxiliary_parameters():
    torch.manual_seed(7)
    model = TinyModel()
    groups = get_muon_param_groups(
        model,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        parameter_mode="all_matrices",
    )
    optimizer = MuonWithAuxAdamW(groups)

    matrix_before = model.linear.weight.detach().clone()
    vector_before = model.vector_weight.detach().clone()

    loss = model(torch.randn(5, 4))
    loss.backward()
    optimizer.step()

    assert not torch.allclose(model.linear.weight, matrix_before)
    assert not torch.allclose(model.vector_weight, vector_before)


def test_default_muon_grouping_uses_only_transformer_hidden_matrices():
    from flashace.model import TransformersACE

    model = TransformersACE(
        r_max=3.0,
        l_max=1,
        num_radial=3,
        hidden_dim=8,
        num_layers=1,
        correlation_channels=4,
        attention_num_heads=2,
    )
    groups = get_muon_param_groups(model, learning_rate=1.0e-3, weight_decay=1.0e-4)
    muon_group = next(group for group in groups if group["use_muon"])
    by_id = {id(param): name for name, param in model.named_parameters()}
    muon_names = {by_id[id(param)] for param in muon_group["params"]}

    assert "emb.weight" not in muon_names
    assert "readout.0.weight" not in muon_names
    assert "readout.2.weight" not in muon_names
    assert "ace.radial_net.layer0.weight" not in muon_names
    assert "ace.radial_net.layer1.weight" not in muon_names
    assert "ace.center_proj.weight" not in muon_names
    assert "layers.0.q_proj.weight" in muon_names
    assert "layers.0.k_proj.weight" in muon_names
    assert "layers.0.scalar_ffn.0.weight" in muon_names
    assert "layers.0.scalar_ffn.3.weight" in muon_names
