# CsPbI3 Phase Test

This directory contains five periodic CsPbI3 structures and reusable scripts
for phase-energy evaluation and geometry optimization with Transformers-ACE.

See [../../docs/CSPBI3_TEST.md](../../docs/CSPBI3_TEST.md) for installation,
commands, units, reference conventions, and scientific validation guidance.

Quick single-point evaluation from the repository root:

```bash
python tests/cspbi3/evaluate_phases.py \
  --model training/model.pt \
  --device cpu \
  --reference minimum
```

The committed files under `results/` were generated in one run with this exact
checkpoint. Per-phase scratch directories and copied checkpoints remain ignored
because mixing checkpoints between phases does not define a valid relative
energy comparison.
