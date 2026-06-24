import math

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet


class PolynomialCutoff(nn.Module):
    """Smooth polynomial cutoff for local ACE radial features.

    The expression ``1 - (p + 1) * x**p + p * x**(p + 1)`` forces both the value and
    first derivative to vanish at the cutoff.
    """

    def __init__(self, r_max: float, p: int = 5):
        super().__init__()
        self.r_max = float(r_max)
        self.p = p

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(distances / self.r_max, max=1.0)
        x_p = torch.pow(x, self.p)
        return 1 - (self.p + 1) * x_p + self.p * x_p * x


class BesselBasis(nn.Module):
    """Bessel radial basis for cutoff-local ACE descriptors.

    Setting ``trainable=True`` makes the frequencies learnable, similar to the
    adaptive radial grids explored in GRACE and NequIP variants that refine the
    basis where the data needs more resolution.
    """

    def __init__(self, r_max: float, num_radial: int, trainable: bool = False):
        super().__init__()
        self.r_max = float(r_max)
        freq = torch.arange(1, num_radial + 1, dtype=torch.get_default_dtype()) * math.pi
        if trainable:
            self.freq = nn.Parameter(freq)
        else:
            self.register_buffer("freq", freq)
        self.norm = math.sqrt(2 / self.r_max)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = torch.clamp(distances, min=0.0)
        scaled = distances.unsqueeze(-1) * self.freq / self.r_max

        # Use the analytical limit for r -> 0 to avoid NaNs.
        safe_dist = distances.unsqueeze(-1).clamp(min=1e-12)
        bessel = torch.sin(scaled) / safe_dist
        bessel = torch.where(
            distances.unsqueeze(-1) == 0,
            self.freq / self.r_max,
            bessel,
        )
        return self.norm * bessel


class GaussianBasis(nn.Module):
    """Gaussian radial basis with optional learnable centers/widths.

    The centered Gaussians mirror the smoothly decaying descriptors used in
    modern equivariant potentials (e.g., PaiNN, SpookyNet) and can reduce the
    Gibbs-like ringing that Bessel bases sometimes exhibit near the cutoff.
    """

    def __init__(
        self,
        r_max: float,
        num_radial: int,
        width_factor: float = 0.5,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.r_max = float(r_max)

        centers = torch.linspace(0.0, self.r_max, num_radial + 2, dtype=torch.get_default_dtype())[1:-1]
        delta = centers[1] - centers[0]
        widths = torch.full_like(centers, width_factor * delta)

        if trainable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = torch.clamp(distances, min=0.0)
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff / self.widths).pow(2))


class ACERadialBasis(nn.Module):
    """Cutoff-masked radial basis with Bessel or Gaussian options."""

    def __init__(
        self,
        r_max: float,
        num_radial: int,
        envelope_exponent: int = 5,
        basis_type: str = "bessel",
        trainable: bool = False,
        gaussian_width: float = 0.5,
    ):
        super().__init__()
        self.cutoff = PolynomialCutoff(r_max, p=envelope_exponent)

        basis_type = basis_type.lower()
        if basis_type == "bessel":
            self.basis = BesselBasis(r_max, num_radial, trainable=trainable)
        elif basis_type == "gaussian":
            self.basis = GaussianBasis(r_max, num_radial, width_factor=gaussian_width, trainable=trainable)
        else:
            raise ValueError(f"Unsupported radial basis type: {basis_type}")

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        cutoff = self.cutoff(distances)
        return cutoff.unsqueeze(-1) * self.basis(distances)

class ACE_Descriptor(nn.Module):
    def __init__(
        self,
        r_max,
        l_max,
        num_radial,
        hidden_dim,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        radial_mlp_hidden: int = 64,
        radial_mlp_layers: int = 2,
    ):
        super().__init__()
        self.r_max = r_max
        self.num_radial = num_radial
        
        # 1. Local ACE angular channels
        # ------------------------------------------------------------------
        # Instead of giving 'hidden_dim' to everything, we taper it down.
        # Scalars (L=0) get full resolution (Chemistry).
        # Vectors/Tensors (L>0) get reduced resolution (Geometry).
        # This saves massive amounts of memory.
        
        irreps_list = []
        for l in range(l_max + 1):
            if l == 0:
                dim = hidden_dim          # e.g. 128
            elif l == 1:
                dim = hidden_dim // 2     # e.g. 64
            else:
                dim = hidden_dim // 4     # e.g. 32
            
            # Parity (-1)**l is standard
            irreps_list.append((dim, (l, (-1)**l)))
            
        self.irreps_out = o3.Irreps(irreps_list)
        
        # Spherical Harmonics (Geometry Input)
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        
        # Node Features (Scalar Input)
        self.irreps_node = o3.Irreps(f"{hidden_dim}x0e")
        # ------------------------------------------------------------------

        # 2. A-Basis Components
        self.radial_basis = ACERadialBasis(
            r_max,
            num_radial,
            envelope_exponent=envelope_exponent,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        self.sh = o3.SphericalHarmonics(self.irreps_sh, normalize=True, normalization="component")

        # Radial functions provide edge-dependent tensor-product weights that
        # mix chemical scalar channels with spherical harmonics.
        self.tp_a = o3.FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        radial_mlp_hidden = max(1, int(radial_mlp_hidden))
        radial_mlp_layers = max(1, int(radial_mlp_layers))
        mlp_sizes = [num_radial] + [radial_mlp_hidden] * (radial_mlp_layers - 1) + [self.tp_a.weight_numel]
        self.radial_net = FullyConnectedNet(mlp_sizes, torch.nn.functional.silu)

        # 3. B-Basis (Symmetric Contraction)
        # The B-basis is a Clebsch-Gordan contraction of local A-basis channels;
        # learned mixing happens after the contraction.
        self.tp_b = o3.FullyConnectedTensorProduct(
            self.irreps_out,
            self.irreps_out,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        # The B-basis in ACE is a pure Clebsch–Gordan contraction without
        # learnable weights. ``FullyConnectedTensorProduct`` requires explicit
        # weights when ``internal_weights=False``, so we register a fixed buffer
        # of ones to recover the unweighted contraction behavior.
        self.register_buffer("tp_b_weights", torch.ones(self.tp_b.weight_numel))

        # Linear Mixing to recover full feature interactions
        self.mix = o3.Linear(self.tp_b.irreps_out, self.irreps_out)

    def forward(self, node_attrs, edge_index, edge_vec, edge_len):
        sender = edge_index[0]
        receiver = edge_index[1]

        # Stage 1: Projection
        radial_emb = self.radial_basis(edge_len)
        Y_lm = self.sh(edge_vec)
        tp_weights = self.radial_net(radial_emb)

        # The radial basis already vanishes at ``r_max`` but masking protects the
        # downstream tensor products from spurious numerical noise when edges
        # are padded or left over from a larger neighbor list.
        edge_mask = (edge_len <= self.r_max).unsqueeze(-1)

        # Incorporate atomic attributes on the sending atoms and gate the
        # tensor product weights with the learned radial functions before
        # combining with the spherical harmonics. This mirrors the ACE
        # construction where chemical identity enters the A-basis through the
        # tensor-product weights.
        node_feats = node_attrs[sender]
        edge_feats = self.tp_a(node_feats, Y_lm, tp_weights)
        edge_feats = edge_feats * edge_mask

        # Sum Neighbors
        A_basis = torch.zeros(
            node_attrs.shape[0], edge_feats.shape[1],
            device=node_attrs.device, dtype=edge_feats.dtype
        )
        A_basis.index_add_(0, receiver, edge_feats)
        
        # Stage 2: Contraction (Linear scaling with N)
        # B = A ⊗ A with learned Clebsch–Gordan mixing
        # ``FullyConnectedTensorProduct`` expects a batch dimension on the
        # weights when ``shared_weights`` is False. We want the same fixed
        # Clebsch–Gordan contraction for every atom, so broadcast a single
        # vector of ones across the batch dimension of ``A_basis``.
        tp_b_weights = self.tp_b_weights.expand(A_basis.shape[0], -1)
        B_basis = self.tp_b(A_basis, A_basis, tp_b_weights)

        # Mix and Add Residual
        return self.mix(B_basis) + A_basis


class SmoothPolynomialCutoff(nn.Module):
    """Compact C2 cutoff used by the corrected descriptor and attention.

    The quintic smoothstep and its first two derivatives are zero at ``r_max``.
    This matters for stable forces, stress, phonons, and variable-cell dynamics.
    """

    def __init__(self, r_max: float):
        super().__init__()
        self.r_max = float(r_max)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(distances / self.r_max, min=0.0, max=1.0)
        envelope = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
        return torch.where(distances < self.r_max, envelope, torch.zeros_like(envelope))


class SmoothACERadialBasis(nn.Module):
    """Radial basis multiplied by a compact C2 cutoff envelope."""

    def __init__(
        self,
        r_max: float,
        num_radial: int,
        basis_type: str = "bessel",
        trainable: bool = False,
        gaussian_width: float = 0.5,
    ):
        super().__init__()
        self.cutoff = SmoothPolynomialCutoff(r_max)
        basis_type = basis_type.lower()
        if basis_type == "bessel":
            self.basis = BesselBasis(r_max, num_radial, trainable=trainable)
        elif basis_type == "gaussian":
            self.basis = GaussianBasis(
                r_max,
                num_radial,
                width_factor=gaussian_width,
                trainable=trainable,
            )
        else:
            raise ValueError(f"Unsupported radial basis type: {basis_type}")

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return self.cutoff(distances).unsqueeze(-1) * self.basis(distances)


class ACEV2Descriptor(nn.Module):
    """Local equivariant ACE-style density correlations.

    Neighbor species and geometry first form an equivariant density ``A``. Learned
    Clebsch-Gordan products recursively contract that same permutation-invariant
    density, retaining independent multiplicity channels instead of collapsing
    them with a shared vector of ones. The central species remains an explicit
    part of every atomic representation.
    """

    __constants__ = [
        "r_max",
        "hidden_dim",
        "correlation_order",
        "irreps_out_dim",
        "irreps_correlation_dim",
    ]

    def __init__(
        self,
        r_max: float,
        l_max: int,
        num_radial: int,
        hidden_dim: int,
        correlation_order: int = 4,
        correlation_channels: int = 16,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        gaussian_width: float = 0.5,
        radial_mlp_hidden: int = 32,
        radial_mlp_layers: int = 2,
    ):
        super().__init__()
        if correlation_order < 2 or correlation_order > 6:
            raise ValueError("correlation_order must be between 2 and 6")

        self.r_max = float(r_max)
        self.hidden_dim = int(hidden_dim)
        self.correlation_order = int(correlation_order)

        output_irreps = []
        correlation_irreps = []
        for l in range(l_max + 1):
            output_mul = hidden_dim if l == 0 else hidden_dim // (2 if l == 1 else 4)
            corr_mul = correlation_channels if l == 0 else correlation_channels // (2 if l == 1 else 4)
            output_irreps.append((max(1, output_mul), (l, (-1) ** l)))
            correlation_irreps.append((max(1, corr_mul), (l, (-1) ** l)))

        self.irreps_out = o3.Irreps(output_irreps)
        self.irreps_correlation = o3.Irreps(correlation_irreps)
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        self.irreps_node = o3.Irreps(f"{hidden_dim}x0e")
        self.irreps_out_dim = int(self.irreps_out.dim)
        self.irreps_correlation_dim = int(self.irreps_correlation.dim)

        self.cutoff = SmoothPolynomialCutoff(r_max)
        self.radial_basis = SmoothACERadialBasis(
            r_max,
            num_radial,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_sh,
            normalize=True,
            normalization="component",
        )
        self.tp_density = o3.FullyConnectedTensorProduct(
            self.irreps_node,
            self.irreps_sh,
            self.irreps_correlation,
            internal_weights=False,
            shared_weights=False,
        )

        radial_mlp_hidden = max(1, int(radial_mlp_hidden))
        radial_mlp_layers = max(1, int(radial_mlp_layers))
        mlp_sizes = (
            [num_radial]
            + [radial_mlp_hidden] * (radial_mlp_layers - 1)
            + [self.tp_density.weight_numel]
        )
        self.radial_net = FullyConnectedNet(mlp_sizes, torch.nn.functional.silu)

        # A is body order two. Each recursive A-product adds one body order.
        self.contractions = nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    self.irreps_correlation,
                    self.irreps_correlation,
                    self.irreps_correlation,
                    internal_weights=True,
                    shared_weights=True,
                )
                for _ in range(self.correlation_order - 2)
            ]
        )
        self.order_mix = nn.ModuleList(
            [
                o3.Linear(self.irreps_correlation, self.irreps_out)
                for _ in range(self.correlation_order - 1)
            ]
        )
        self.center_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _density(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
    ):
        sender = edge_index[0]
        receiver = edge_index[1]
        radial = self.radial_basis(edge_len)
        harmonics = self.sh(edge_vec)
        weights = self.radial_net(radial)
        edge_features = self.tp_density(node_attrs[sender], harmonics, weights)

        density = torch.zeros(
            node_attrs.shape[0],
            self.irreps_correlation_dim,
            device=node_attrs.device,
            dtype=edge_features.dtype,
        )
        density.index_add_(0, receiver, edge_features)
        return density, edge_features, self.cutoff(edge_len)

    def forward(
        self,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
        return_edge_features: bool = False,
    ):
        density, edge_features, cutoff = self._density(
            node_attrs,
            edge_index,
            edge_vec,
            edge_len,
        )

        non_scalar_dim = self.irreps_out_dim - self.hidden_dim
        center = torch.cat(
            (
                self.center_proj(node_attrs),
                node_attrs.new_zeros((node_attrs.shape[0], non_scalar_dim)),
            ),
            dim=-1,
        )

        correlation = density
        output = center + self.order_mix[0](correlation)
        if self.correlation_order >= 3:
            correlation = self.contractions[0](correlation, density)
            output = output + self.order_mix[1](correlation)
        if self.correlation_order >= 4:
            correlation = self.contractions[1](correlation, density)
            output = output + self.order_mix[2](correlation)
        if self.correlation_order >= 5:
            correlation = self.contractions[2](correlation, density)
            output = output + self.order_mix[3](correlation)
        if self.correlation_order >= 6:
            correlation = self.contractions[3](correlation, density)
            output = output + self.order_mix[4](correlation)

        if return_edge_features:
            return output, edge_features, cutoff
        return output
