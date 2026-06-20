# CsPbI3 Phase Test

This test evaluates a Transformers-ACE checkpoint on five periodic CsPbI3
polymorphs:

- cubic alpha;
- tetragonal beta;
- orthorhombic gamma;
- edge-sharing delta;
- face-sharing delta.

The 20-atom source cells are taken from the
[CsPbI3 NEP r2SCAN+rVV10 structure-optimization example](https://github.com/paramvir3/Machine-Learning-Potentials-Halide-Perovskites-solar-cells-LEDs/tree/main/CsPbI3/NEP-r2SCAN_rVV10/1-structures_optimization).
Each included cell contains four CsPbI3 formula units.

Run all commands from the repository root with the Python environment activated.

## Single-Point Phase Energies

```bash
python tests/cspbi3/evaluate_phases.py \
  --model training/model.pt \
  --device cpu
```

The default reference is edge-sharing delta, matching the source comparison.
Use the checkpoint's predicted minimum as zero instead:

```bash
python tests/cspbi3/evaluate_phases.py \
  --model training/model.pt \
  --device cpu \
  --reference minimum
```

Results are written to:

```text
tests/cspbi3/results/phase_energies.csv
```

The table contains total energy, energy per formula unit, relative meV per
formula unit, relative kJ/mol, volume, and maximum atomic force.

Plot the resulting relative energies:

```bash
python tests/cspbi3/plot_rl.py \
  --csv tests/cspbi3/results/phase_energies.csv \
  --output-dir tests/cspbi3/results
```

## Relax All Phases

Relax positions while keeping the supplied cells fixed:

```bash
python tests/cspbi3/evaluate_phases.py \
  --model training/model.pt \
  --device cpu \
  --relax --fmax 0.05 --steps 500
```

Add `--relax-cell` only after validating checkpoint stress against finite
energy-strain differences:

```bash
python tests/cspbi3/evaluate_phases.py \
  --model training/model.pt \
  --device cpu \
  --relax-cell --fmax 0.05 --steps 500
```

Relaxed structures and optimization logs are saved under
`tests/cspbi3/results/relaxed/`.

## Relax One Structure

```bash
python tests/cspbi3/relax_single.py \
  --structure tests/cspbi3/structures/orthorhombic_gamma_phase.vasp \
  --model training/model.pt \
  --device cpu \
  --repeat 1 1 1 \
  --fixed-cell \
  --fmax 0.05 \
  --steps 500
```

Remove `--fixed-cell` to relax the periodic cell. Optional controls include
`--hydrostatic` and `--fix-symmetry`. Symmetry constraints require `spglib`.

## Scientific Checks

Before trusting relative energies or variable-cell results:

1. Confirm that all phases are represented in independent validation data.
2. Compare phase-resolved energy, force, and stress errors.
3. Verify stress against finite differences of energy under all six independent
   symmetric strains.
4. Compare energy-volume curves and elastic constants with the reference method.
5. Use the same composition normalization for every phase.
6. Reject relaxations that do not converge or leave the training distribution.

An apparently successful high-symmetry cubic relaxation does not by itself prove
transferability to tilted or edge-sharing phases.
