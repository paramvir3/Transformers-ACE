import torch

from flashace.model import TransformersACE
from transformers_ace.deploy import LAMMPSEnergyModel


def test_lammps_energy_export_supports_position_and_strain_gradients():
    model = TransformersACE(
        r_max=4.0,
        l_max=1,
        num_radial=4,
        hidden_dim=8,
        num_layers=1,
        correlation_channels=4,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    deploy_model = LAMMPSEnergyModel(model).eval()
    z = torch.tensor([55, 82, 53, 53], dtype=torch.long)
    pos = torch.randn(4, 3)
    cell = torch.eye(3) * 10.0
    edge_index = torch.tensor([[1, 2, 3, 0], [0, 0, 0, 1]], dtype=torch.long)
    edge_shift = torch.zeros(edge_index.shape[1], 3)
    strain = torch.zeros(6)
    local_mask = torch.ones(4)

    traced = torch.jit.trace(
        deploy_model,
        (z, pos, cell, edge_index, edge_shift, strain, local_mask),
        check_trace=False,
    )
    pos = pos.clone().requires_grad_(True)
    strain = torch.zeros(6, requires_grad=True)

    energy = traced(z, pos, cell, edge_index, edge_shift, strain, local_mask)
    grad_pos, grad_strain = torch.autograd.grad(energy, (pos, strain))
    forces = -grad_pos
    virial = torch.stack(
        (
            -grad_strain[0],
            -grad_strain[1],
            -grad_strain[2],
            -0.5 * grad_strain[3],
            -0.5 * grad_strain[4],
            -0.5 * grad_strain[5],
        )
    )

    assert energy.ndim == 0
    assert forces.shape == pos.shape
    assert virial.shape == (6,)
    assert torch.isfinite(energy)
    assert torch.isfinite(forces).all()
    assert torch.isfinite(virial).all()
