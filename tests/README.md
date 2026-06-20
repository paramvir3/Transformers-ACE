# Tests

`test_stress_finite_difference.py` checks the energy-derived periodic stress
against finite differences for normal and symmetric shear strain.

`cspbi3/` contains the reproducible phase-energy and structure-relaxation
workflow. Reference structures and reusable scripts are tracked; local models,
relaxation directories, logs, plots, and generated results remain ignored.

Run the automated regression from the repository root:

```bash
pytest -q tests/test_stress_finite_difference.py
```

See [`../docs/CSPBI3_TEST.md`](../docs/CSPBI3_TEST.md) for the phase workflow.
