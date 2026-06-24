# Transformers-ACE Architecture

The full paper-style derivation is available as
[LaTeX source](TRANSFORMERS_ACE_METHODS.tex) and a
[rendered PDF](../output/pdf/transformers_ace_methods.pdf).

## Local periodic geometry

For a directed neighbor edge from atom `j` to center `i`, including the ASE
periodic image shift `S_ij`, the displacement is

```text
r_ij = r_j - r_i + S_ij h,       d_ij = |r_ij|,
```

where `h` is the cell matrix. Only edges with `d_ij < r_cut` are used. The v2
model uses the compact quintic envelope

```text
f_cut(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5,   x = d / r_cut,
```

inside the cutoff and zero outside. Its value, first derivative, and second
derivative vanish at the boundary.

## ACE density correlations

Radial functions, sender-species embeddings, and spherical harmonics form the
equivariant edge density features

```text
a_ij^(l,m,c) = f_cut(d_ij) R_c(d_ij, z_j) Y_lm(r_ij / d_ij).
```

Their sum is permutation invariant with respect to the neighbors of center `i`:

```text
A_i = sum_j a_ij.
```

`A_i` is body order two. Learned Clebsch-Gordan tensor products recursively
construct higher body-order correlations:

```text
C_i^(2) = A_i,
C_i^(nu+1) = CG(C_i^(nu), A_i),       2 <= nu < correlation_order.
```

Independent learned multiplicity weights are retained in every contraction.
The old all-ones contraction, which made copies of an irrep rank deficient, is
not used in v2. Each body order is linearly projected to the node irreps and
added to an explicit central-species embedding. Changing `z_i` therefore changes
the representation of center `i`, even when its neighbor density is unchanged.

This is a compressed, learned ACE density-correlation basis with configurable
angular, channel, and body-order truncation. The default is body order four,
while the implementation permits orders two through six for larger data sets.
It does not claim formal completeness of an untruncated sparse ACE basis.

ACEsuit/EquivariantTensors.jl builds a more explicit sparse symmetric ACE basis:
it enumerates one-particle `(n, l, m)` specifications, forms symmetric products,
then applies sparse permutation-invariant Clebsch-Gordan symmetrization maps.
Transformers-ACE follows the same symmetry requirements through e3nn tensor
products and compact density correlations, but keeps a learned low-rank basis for
scalability and local attention. Regression tests check O(3) equivariance,
inversion parity, central-species dependence, independent multiplicity channels,
and higher body-order availability.

## Strictly local equivariant attention

Attention is performed independently in each center's fixed neighbor set. The
query is built from the current center scalar state. Keys and equivariant values
come from the fixed geometric edge features `a_ij`; they do not use or transmit
the updated hidden state of atom `j`.

For attention head `p`, the invariant query, key, and score are

```text
q_ip = WQ_p LN(h_i,scalar)
k_ijp = WK_p LN(a_ij,scalar)

s_ijp = q_ip . k_ijp / sqrt(d_key)
        + b_p(B(d_ij)) - softplus(lambda_p) d_ij
```

The cutoff-normalized weight is

```text
alpha_ijp = f_cut(d_ij) exp(s_ijp)
             / [1 + sum_k f_cut(d_ik) exp(s_ikp)].
```

The `1` is a local null/self channel. It prevents softmax normalization from
cancelling the cutoff when the last neighbor leaves the list. Directional values
are equivariant linear maps `v_ijp = LinearV_p(a_ij)`. Their local update is

```text
u_i = LinearO[(1 / H) sum_p sum_j alpha_ijp v_ijp].
```

Atom-local squared norms of the resulting non-scalar channels,
`n_i,c,l = sum_m |h_i,c,l,m|^2`, feed the scalar update. This allows angular
interference among neighbors to affect energy while preserving rotational
invariance.

This model is strictly cutoff local and contains no learned-state message
passing between centers. Architecture v2 also contains no explicit long-range
electrostatic term.

## Conservative observables

The total energy is a sum of invariant atomic scalar readouts:

```text
E = sum_i E_i,
F_i = -dE / dr_i,
sigma_ab = (1 / V) dE / d epsilon_ab.
```

The six strain variables parameterize a symmetric strain tensor. Shear
derivatives are divided by two when reconstructed as a symmetric stress tensor.
The stress target conversion, ASE Voigt order, energy-derived stress loss, and
stress-weight ramp are unchanged from the validated v1 training path.

## Default version-2 dimensions

```text
cutoff:               6.0 Angstrom
radial basis:         12 Bessel functions
node irreps:          64x0e + 32x1o + 16x2e
correlation irreps:   16x0e + 8x1o + 4x2e
maximum body order:   4
allowed body order:   2 through 6
attention:            1 strictly local block, 2 heads
readout:              64 -> 64 -> 1
trainable parameters: 132,005
```

Training uses energy-per-atom MSE, Cartesian force MSE, and six-component stress
MSE. The supplied CsPbI3 configuration uses a blocked trajectory split, light
dropout and weight decay, a stress-weight ramp, validation-based early stopping,
and separate best/final checkpoints.

## Checkpoint versions

New checkpoints store `architecture_version: 2`. Checkpoints without this field
are loaded with `LegacyTransformersACE`, preserving existing v1 results instead
of interpreting old weights with the new equations.
