# CsPbI3 Phase Test

This directory contains five periodic CsPbI3 structures and reusable scripts
for phase-energy evaluation and geometry optimization with Transformers-ACE.

See [../../docs/CSPBI3_TEST.md](../../docs/CSPBI3_TEST.md) for installation,
commands, units, reference conventions, and scientific validation guidance.

Quick single-point evaluation from the repository root:

```bash
python phase_tests/cspbi3/evaluate_phases.py \
  --model /absolute/path/to/model.pt \
  --device cpu \
  --reference minimum
```
