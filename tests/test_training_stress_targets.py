import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from train import MetricTracker, stress_target_from_atoms, stress_to_voigt


def _periodic_atom():
    return Atoms("Cs", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 2.0, pbc=True)


def test_ase_voigt_stress_order_and_zero_label_are_preserved():
    atoms = _periodic_atom()
    stress_voigt = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
    atoms.calc = SinglePointCalculator(atoms, stress=stress_voigt)

    stress, has_stress = stress_target_from_atoms(atoms)

    assert has_stress
    np.testing.assert_allclose(
        stress,
        [[1.0, 0.6, 0.5], [0.6, 2.0, 0.4], [0.5, 0.4, 3.0]],
    )

    atoms.calc = SinglePointCalculator(atoms, stress=np.zeros(6))
    stress, has_stress = stress_target_from_atoms(atoms)
    assert has_stress
    np.testing.assert_allclose(stress, np.zeros((3, 3)))


def test_virial_uses_atomistic_sign_convention():
    atoms = _periodic_atom()
    atoms.info["virial"] = np.eye(3) * -4.0

    stress, has_stress = stress_target_from_atoms(atoms)

    assert has_stress
    np.testing.assert_allclose(stress, np.eye(3) * 0.5)


def test_stress_metric_uses_six_independent_components():
    target = torch.zeros((3, 3))
    prediction = torch.tensor(
        [[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]
    )
    tracker = MetricTracker()
    tracker.update(
        torch.tensor(0.0),
        torch.zeros((1, 3)),
        prediction,
        torch.tensor(0.0),
        torch.zeros((1, 3)),
        target,
        True,
        1,
    )

    _, _, stress_rmse, _, _ = tracker.get_metrics()
    expected = torch.mean(stress_to_voigt(prediction) ** 2).sqrt().item()
    np.testing.assert_allclose(stress_rmse, expected)
