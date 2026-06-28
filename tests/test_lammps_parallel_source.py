from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_native_pair_style_allows_mpi_with_newton_guard():
    source = (ROOT / "lammps/pair_style/pair_transformers_ace.cpp").read_text()

    assert "comm->nprocs != 1" not in source
    assert "force->newton_pair" in source
    assert "reverse-communicated" in source
    assert "local_rank() % count" in source


def test_lammps_examples_use_newton_on_for_parallel_force_comm():
    inputs = [
        ROOT / "lammps/pair_style/in.transformers_ace",
        ROOT / "tests/run_lammps/test_lammps_cspbi3/in.transformers_ace",
        ROOT / "tests/run_lammps/test_plumed_cspbi3/in.transformers_ace",
        ROOT / "tests/run_lammps/test_plumed_cspbi3_2/in.transformers_ace",
    ]

    for path in inputs:
        text = path.read_text()
        assert "newton          on" in text or "newton on" in text
        assert "newton off" not in text
