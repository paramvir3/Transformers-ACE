import numpy as np
import torch

from flashace.model import FlashACE


def _data(model, deformation=None):
    pos = torch.tensor(
        [[0.2, 0.3, 0.4], [1.4, 0.8, 0.7], [0.7, 1.6, 1.2]],
        dtype=torch.float32,
    )
    cell = torch.tensor(
        [[3.1, 0.0, 0.0], [0.2, 3.3, 0.0], [0.1, 0.3, 3.5]],
        dtype=torch.float32,
    )
    if deformation is not None:
        pos = pos @ deformation
        cell = cell @ deformation

    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]],
        dtype=torch.long,
    )
    return {
        "z": torch.tensor([55, 53, 82]),
        "pos": pos,
        "cell": cell,
        "edge_index": edge_index,
        "edge_shift": torch.zeros((edge_index.shape[1], 3)),
        "volume": torch.abs(torch.det(cell)),
    }


def test_stress_matches_symmetric_strain_finite_difference():
    torch.manual_seed(7)
    model = FlashACE(
        r_max=5.0,
        l_max=1,
        num_radial=3,
        hidden_dim=8,
        num_layers=1,
        attention_num_heads=1,
    )
    model.eval()

    base = _data(model)
    energy, _, stress, _ = model(base, training=False, compute_stress=True)
    volume = float(base["volume"])
    assert np.isfinite(float(energy.detach()))

    step = 2.0e-3
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for a, b in components:
        plus = torch.eye(3)
        minus = torch.eye(3)
        plus[a, b] += step
        minus[a, b] -= step
        if a != b:
            plus[b, a] += step
            minus[b, a] -= step

        e_plus = model(_data(model, plus), training=False, compute_stress=False)[0]
        e_minus = model(_data(model, minus), training=False, compute_stress=False)[0]
        derivative = float(((e_plus - e_minus) / (2.0 * step * volume)).detach())
        component = float(stress[a, b].detach())
        expected = component if a == b else 2.0 * component
        np.testing.assert_allclose(derivative, expected, rtol=2e-2, atol=2e-4)
