import tempfile
from pathlib import Path

import numpy as np
import torch

from flashace.calculator import TransformersACECalculator
from flashace.model import LegacyTransformersACE, TransformersACE
from flashace.physics import ACEV2Descriptor, SmoothPolynomialCutoff


def _directed_edges(positions, cutoff):
    senders = []
    receivers = []
    for receiver in range(len(positions)):
        for sender in range(len(positions)):
            if sender == receiver:
                continue
            if torch.linalg.norm(positions[sender] - positions[receiver]) < cutoff:
                senders.append(sender)
                receivers.append(receiver)
    return torch.tensor([senders, receivers], dtype=torch.long)


def _model(cutoff=3.0):
    torch.manual_seed(11)
    model = TransformersACE(
        r_max=cutoff,
        l_max=2,
        num_radial=4,
        hidden_dim=16,
        num_layers=1,
        correlation_order=4,
        correlation_channels=8,
        attention_num_heads=2,
        attention_dropout=0.0,
    )
    model.eval()
    return model


def _data(positions, cutoff=3.0, numbers=None):
    return {
        "z": torch.tensor(numbers or [55, 82, 53, 53], dtype=torch.long),
        "pos": positions.clone(),
        "edge_index": _directed_edges(positions, cutoff),
        "volume": torch.tensor(1.0),
    }


def test_center_species_is_retained_by_descriptor():
    torch.manual_seed(3)
    descriptor = ACEV2Descriptor(
        r_max=3.0,
        l_max=2,
        num_radial=4,
        hidden_dim=16,
        correlation_channels=8,
    )
    attrs = torch.randn(3, 16)
    edge_index = torch.tensor([[1, 2], [0, 0]])
    edge_vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.2, 0.0]])
    edge_len = torch.linalg.norm(edge_vec, dim=1)

    first = descriptor(attrs, edge_index, edge_vec, edge_len)[0]
    changed = attrs.clone()
    changed[0] = changed[0] + 1.0
    second = descriptor(changed, edge_index, edge_vec, edge_len)[0]

    assert not torch.allclose(first, second)


def test_clebsch_gordan_multiplicity_channels_do_not_collapse():
    torch.manual_seed(4)
    descriptor = ACEV2Descriptor(
        r_max=3.0,
        l_max=2,
        num_radial=4,
        hidden_dim=16,
        correlation_channels=8,
    )
    density = torch.randn(32, descriptor.irreps_correlation.dim)
    contracted = descriptor.contractions[0](density, density)
    scalar_channels = contracted[:, : descriptor.irreps_correlation[0].mul]

    assert descriptor.contractions[0].weight.requires_grad
    assert torch.linalg.matrix_rank(scalar_channels).item() > 1


def test_quintic_cutoff_is_c2_at_boundary():
    cutoff = SmoothPolynomialCutoff(3.0).double()
    radius = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    value = cutoff(radius)
    first = torch.autograd.grad(value, radius, create_graph=True)[0]
    second = torch.autograd.grad(first, radius)[0]

    np.testing.assert_allclose(float(value), 0.0, atol=1e-14)
    np.testing.assert_allclose(float(first), 0.0, atol=1e-14)
    np.testing.assert_allclose(float(second), 0.0, atol=1e-14)


def test_full_model_is_continuous_when_an_edge_leaves_the_neighbor_list():
    model = _model(cutoff=3.0)
    energies = []
    forces = []
    for distance in (2.99, 3.01):
        positions = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        energy, force, _, _ = model(
            _data(positions, cutoff=3.0, numbers=[55, 53]),
            training=False,
            compute_stress=False,
        )
        energies.append(float(energy))
        forces.append(force.detach())

    np.testing.assert_allclose(energies[0], energies[1], atol=2e-6)
    assert float(forces[0].abs().max()) < 2e-6
    assert float(forces[1].abs().max()) == 0.0


def test_energy_and_forces_are_rotation_and_permutation_equivariant():
    model = _model()
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.1, 0.4, 0.5], [0.4, 1.3, 0.8], [0.8, 0.7, 1.6]],
        dtype=torch.float32,
    )
    energy, forces, _, _ = model(_data(positions), training=False, compute_stress=False)

    axis = torch.tensor([0.3, -0.4, 0.5])
    axis = axis / torch.linalg.norm(axis)
    angle = torch.tensor(0.73)
    cross = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    rotation = (
        torch.eye(3) * torch.cos(angle)
        + (1.0 - torch.cos(angle)) * torch.outer(axis, axis)
        + torch.sin(angle) * cross
    )
    rotated_positions = positions @ rotation.T
    rotated_energy, rotated_forces, _, _ = model(
        _data(rotated_positions), training=False, compute_stress=False
    )
    np.testing.assert_allclose(float(rotated_energy), float(energy), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        rotated_forces.detach().numpy(),
        (forces @ rotation.T).detach().numpy(),
        rtol=2e-4,
        atol=2e-4,
    )

    permutation = torch.tensor([2, 0, 3, 1])
    numbers = [55, 82, 53, 53]
    permuted_numbers = [numbers[i] for i in permutation.tolist()]
    permuted_energy, permuted_forces, _, _ = model(
        _data(positions[permutation], numbers=permuted_numbers),
        training=False,
        compute_stress=False,
    )
    np.testing.assert_allclose(float(permuted_energy), float(energy), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        permuted_forces.detach().numpy(),
        forces[permutation].detach().numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_attention_does_not_read_updated_sender_hidden_states():
    model = _model()
    layer = model.layers[0]
    node_features = torch.randn(2, model.attention_irreps.dim)
    edge_features = torch.randn(1, model.ace.irreps_correlation.dim)
    receiver = torch.tensor([0], dtype=torch.long)
    edge_len = torch.tensor([1.2])
    cutoff = model.ace.cutoff(edge_len)

    first = layer(node_features, edge_features, receiver, edge_len, cutoff)
    changed = node_features.clone()
    changed[1] = changed[1] + 100.0
    second = layer(changed, edge_features, receiver, edge_len, cutoff)

    torch.testing.assert_close(first[0], second[0])


def test_new_model_is_compact_and_legacy_checkpoints_remain_versioned():
    model = TransformersACE(
        r_max=6.0,
        l_max=2,
        num_radial=12,
        hidden_dim=64,
        num_layers=1,
        correlation_order=4,
        correlation_channels=16,
        attention_num_heads=2,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) < 200_000

    legacy = LegacyTransformersACE(
        r_max=3.0,
        l_max=1,
        num_radial=3,
        hidden_dim=8,
        num_layers=1,
        attention_num_heads=1,
    )
    checkpoint = {
        "config": {
            "r_max": 3.0,
            "l_max": 1,
            "num_radial": 3,
            "hidden_dim": 8,
            "num_layers": 1,
            "attention_num_heads": 1,
        },
        "model_state_dict": legacy.state_dict(),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "legacy.pt"
        torch.save(checkpoint, path)
        calculator = TransformersACECalculator(str(path), device="cpu")
        assert isinstance(calculator.model, LegacyTransformersACE)

        v2_path = Path(directory) / "v2.pt"
        torch.save(
            {
                "config": {
                    "architecture_version": 2,
                    "r_max": 6.0,
                    "l_max": 2,
                    "num_radial": 12,
                    "hidden_dim": 64,
                    "num_layers": 1,
                    "correlation_order": 4,
                    "correlation_channels": 16,
                    "radial_mlp_hidden": 32,
                    "attention_num_heads": 2,
                },
                "model_state_dict": model.state_dict(),
            },
            v2_path,
        )
        calculator = TransformersACECalculator(str(v2_path), device="cpu")
        assert isinstance(calculator.model, TransformersACE)
