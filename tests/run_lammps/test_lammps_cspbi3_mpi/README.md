# CsPbI3 MPI LAMMPS Validation

This folder records a local MPI validation run for the native
`pair_style transformers_ace` interface after enabling domain-decomposed
multi-rank execution.

The run used the same 640-atom CsPbI3 input from
`../test_lammps_cspbi3`, with `newton on` and the exported
`../model.transformers_ace.pt` model. Generated LAMMPS files such as
trajectories, restarts, logs, and copied model files are intentionally ignored.

Run pattern:

```bash
mpirun -np 8 /path/to/lammps/build/lmp -in ../test_lammps_cspbi3/in.transformers_ace
```

The local validation completed:

- 1000 NVT steps;
- 3000 NPT steps;
- 8 MPI ranks on a `4 by 1 by 2` processor grid;
- 640 atoms;
- 0 dangerous neighbor-list builds.

See `mpi_validation_summary.csv` for the compact recorded metrics.
