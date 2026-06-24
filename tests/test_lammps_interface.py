import numpy as np

from transformers_ace.lammps.external import ase_stress_to_lammps_virial, lammps_box_to_ase_cell


def test_lammps_box_to_ase_cell_restricted_triclinic():
    cell = lammps_box_to_ase_cell(
        boxlo=[1.0, 2.0, 3.0],
        boxhi=[5.0, 8.0, 10.0],
        xy=0.2,
        yz=0.3,
        xz=-0.4,
    )

    expected = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.2, 6.0, 0.0],
            [-0.4, 0.3, 7.0],
        ]
    )
    np.testing.assert_allclose(cell, expected)


def test_ase_stress_to_lammps_virial_sign_and_order():
    # ASE order is xx, yy, zz, yz, xz, xy. LAMMPS virial order is
    # xx, yy, zz, xy, xz, yz and has the opposite sign times volume.
    virial = ase_stress_to_lammps_virial([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], volume=10.0)

    np.testing.assert_allclose(virial, [-10.0, -20.0, -30.0, -60.0, -50.0, -40.0])
