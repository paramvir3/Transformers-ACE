# Tests

`test_stress_finite_difference.py` checks the energy-derived periodic stress
against finite differences for normal and symmetric shear strain.

`test_architecture_v2.py` checks central-species retention, independent
Clebsch-Gordan channels, C2 cutoff behavior, neighbor-list continuity,
rotation/permutation equivariance, model size, and legacy checkpoint loading.

`test_data_splitting.py` checks deterministic blocked trajectory validation and
the optional boundary gap. `test_training_stress_targets.py` preserves ASE
stress/virial conventions and six-component stress metrics.

`cspbi3/` contains the reproducible phase-energy and structure-relaxation
workflow. Reference structures and reusable scripts are tracked; local models,
relaxation directories, logs, plots, and generated results remain ignored.

Run the automated regression from the repository root:

```bash
pytest -q tests
```

See [`../docs/CSPBI3_TEST.md`](../docs/CSPBI3_TEST.md) for the phase workflow.
