import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from .physics import ACE_Descriptor

class ScalarMessagePassing(nn.Module):
    """Lightweight, scalar-only message passing to mimic NequIP-style updates."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        scalars, rest = h[..., : self.hidden_dim], h[..., self.hidden_dim :]
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h

        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_len.unsqueeze(-1)], dim=-1)
        msgs = self.mlp(msg_in)
        if msgs.dtype != scalars.dtype:
            msgs = msgs.to(scalars.dtype)
        agg = torch.zeros_like(scalars)
        agg.index_add_(0, receiver, msgs)
        scalars = scalars + agg
        return torch.cat([scalars, rest], dim=-1)

class EdgeUpdate(nn.Module):
    """Per-layer scalar edge update that refreshes node scalars from current states."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        scalars, rest = h[..., : self.hidden_dim], h[..., self.hidden_dim :]
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h

        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_len.unsqueeze(-1)], dim=-1)
        msgs = self.mlp(msg_in)
        if msgs.dtype != scalars.dtype:
            msgs = msgs.to(scalars.dtype)
        agg = torch.zeros_like(scalars)
        agg.index_add_(0, receiver, msgs)
        scalars = scalars + agg
        return torch.cat([scalars, rest], dim=-1)

class EdgeStateInit(nn.Module):
    """Initialize per-edge embeddings from current node scalars and distances."""
    def __init__(self, node_dim: int, edge_state_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, edge_state_dim),
            nn.SiLU(),
            nn.Linear(edge_state_dim, edge_state_dim),
        )

    def forward(self, scalars: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index
        if sender.numel() == 0:
            return torch.zeros((0, self.mlp[-1].out_features), device=scalars.device, dtype=scalars.dtype)
        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_len.unsqueeze(-1)], dim=-1)
        return self.mlp(msg_in)

class EdgeStateUpdate(nn.Module):
    """Update edge embeddings from current node scalars and previous edge state."""
    def __init__(self, node_dim: int, edge_state_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_state_dim + 1, edge_state_dim),
            nn.SiLU(),
            nn.Linear(edge_state_dim, edge_state_dim),
        )

    def forward(self, scalars: torch.Tensor, edge_index: torch.Tensor, edge_len: torch.Tensor, edge_state: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index
        if sender.numel() == 0:
            return edge_state
        msg_in = torch.cat([scalars[sender], scalars[receiver], edge_state, edge_len.unsqueeze(-1)], dim=-1)
        return self.mlp(msg_in)

class NodeUpdateMLP(nn.Module):
    """Irrep-aware node update on scalars only (post-aggregation)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scalars, rest = h[..., : self.mlp[0].in_features], h[..., self.mlp[0].in_features :]
        scalars = scalars + self.mlp(scalars)
        return torch.cat([scalars, rest], dim=-1)

class EquivariantMixBlock(nn.Module):
    """Equivariant tensor-product mixing with scalar gating per layer."""
    def __init__(self, irreps: o3.Irreps, hidden_dim: int, l_max: int, radial_mlp_hidden: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.irreps = irreps
        self.sh_irreps = o3.Irreps.spherical_harmonics(l_max)
        self.sh = o3.SphericalHarmonics(self.sh_irreps, normalize=True, normalization="component")
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.sh_irreps,
            self.irreps,
            internal_weights=False,
            shared_weights=False,
        )
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, radial_mlp_hidden),
            nn.SiLU(),
            nn.Linear(radial_mlp_hidden, self.tp.weight_numel),
        )
        non_scalar_dim = self.irreps.dim - hidden_dim
        self.gate = None
        if non_scalar_dim > 0:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, non_scalar_dim),
                nn.Sigmoid(),
            )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h
        sh = self.sh(edge_vec)
        weights = self.radial_mlp(edge_len.unsqueeze(-1))
        msg = self.tp(h[sender], sh, weights)
        agg = torch.zeros_like(h)
        agg.index_add_(0, receiver, msg)
        scalars, rest = agg[..., : self.hidden_dim], agg[..., self.hidden_dim :]
        if self.gate is not None and rest.numel() > 0:
            gate = self.gate(h[..., : self.hidden_dim])
            rest = rest * gate
        agg = torch.cat([scalars, rest], dim=-1)
        return h + agg

class IrrepRMSNorm(nn.Module):
    """Equivariant RMS normalization (per irreps block) with learnable gain."""
    def __init__(self, irreps: o3.Irreps, eps: float = 1e-8):
        super().__init__()
        self.irreps = irreps
        self.eps = eps
        self.slices = []
        gains = []
        cursor = 0
        for mul, ir in self.irreps:
            block_dim = mul * ir.dim
            self.slices.append((cursor, cursor + block_dim))
            gains.append(nn.Parameter(torch.ones(block_dim)))
            cursor += block_dim
        self.gains = nn.ParameterList(gains)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for (start, end), gain in zip(self.slices, self.gains):
            block = x[..., start:end]
            rms = block.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            outputs.append(block / rms * gain)
        return torch.cat(outputs, dim=-1)

def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int) -> nn.Sequential:
    if depth <= 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

class EquiformerV2Block(nn.Module):
    """EquiformerV2-style equivariant mixing block with RMS norm and gating."""
    def __init__(
        self,
        irreps: o3.Irreps,
        hidden_dim: int,
        l_max: int,
        radial_mlp_hidden: int,
        radial_mlp_layers: int,
        rms_eps: float,
        scalar_mlp_hidden: int,
        scalar_mlp_layers: int,
        use_scalar_mlp: bool,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.irreps = irreps
        self.sh_irreps = o3.Irreps.spherical_harmonics(l_max)
        self.sh = o3.SphericalHarmonics(self.sh_irreps, normalize=True, normalization="component")
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.sh_irreps,
            self.irreps,
            internal_weights=False,
            shared_weights=False,
        )
        self.radial_mlp = _make_mlp(
            in_dim=1,
            hidden_dim=radial_mlp_hidden,
            out_dim=self.tp.weight_numel,
            depth=radial_mlp_layers,
        )
        self.norm = IrrepRMSNorm(self.irreps, eps=rms_eps)
        non_scalar_dim = self.irreps.dim - hidden_dim
        self.gate = None
        if non_scalar_dim > 0:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim, non_scalar_dim),
                nn.Sigmoid(),
            )
        self.use_scalar_mlp = use_scalar_mlp
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.scalar_mlp = None
        if self.use_scalar_mlp:
            self.scalar_mlp = _make_mlp(
                in_dim=hidden_dim,
                hidden_dim=scalar_mlp_hidden,
                out_dim=hidden_dim,
                depth=scalar_mlp_layers,
            )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_len: torch.Tensor,
    ) -> torch.Tensor:
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h
        h = self.norm(h)
        sh = self.sh(edge_vec)
        weights = self.radial_mlp(edge_len.unsqueeze(-1))
        msg = self.tp(h[sender], sh, weights)
        agg = torch.zeros_like(h)
        agg.index_add_(0, receiver, msg)
        scalars, rest = agg[..., : self.hidden_dim], agg[..., self.hidden_dim :]
        if self.gate is not None and rest.numel() > 0:
            gate = self.gate(h[..., : self.hidden_dim])
            rest = rest * gate
        h = h + torch.cat([scalars, rest], dim=-1)
        if self.use_scalar_mlp and self.scalar_mlp is not None:
            scalars, rest = h[..., : self.hidden_dim], h[..., self.hidden_dim :]
            scalars = scalars + self.scalar_mlp(self.scalar_norm(scalars))
            h = torch.cat([scalars, rest], dim=-1)
        return h

class TransformerBlock(nn.Module):
    """Full transformer block with pre-norm attention and FFN on all channels."""
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        ffn_hidden: int,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        ffn_gated: bool = False,
        layer_scale_init: float | None = None,
        attention_chunk_size: int | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn_gated = ffn_gated
        if ffn_gated:
            self.ffn_in = nn.Linear(feature_dim, ffn_hidden * 2)
            self.ffn_out = nn.Linear(ffn_hidden, feature_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, ffn_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, feature_dim),
            )
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale_attn = (
            nn.Parameter(torch.full((feature_dim,), layer_scale_init))
            if layer_scale_init is not None
            else None
        )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((feature_dim,), layer_scale_init))
            if layer_scale_init is not None
            else None
        )
        self.attention_chunk_size = attention_chunk_size

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Treat nodes as the sequence dimension; single batch.
        residual = x
        x_norm = self.norm1(x)
        chunk_size = self.attention_chunk_size
        if chunk_size is None or x_norm.shape[0] <= chunk_size:
            attn_out, _ = self.attn(
                x_norm.unsqueeze(0),
                x_norm.unsqueeze(0),
                x_norm.unsqueeze(0),
                attn_mask=attn_mask,
                need_weights=False,
            )
            attn_out = attn_out.squeeze(0)
        else:
            outputs = []
            for start in range(0, x_norm.shape[0], chunk_size):
                end = start + chunk_size
                mask_slice = attn_mask[start:end, :] if attn_mask is not None else None
                chunk_out, _ = self.attn(
                    x_norm[start:end].unsqueeze(0),
                    x_norm.unsqueeze(0),
                    x_norm.unsqueeze(0),
                    attn_mask=mask_slice,
                    need_weights=False,
                )
                outputs.append(chunk_out.squeeze(0))
            attn_out = torch.cat(outputs, dim=0)
        if self.layer_scale_attn is not None:
            attn_out = attn_out * self.layer_scale_attn
        x = residual + self.residual_dropout(attn_out)
        x_norm = self.norm2(x)
        if self.ffn_gated:
            gate, value = self.ffn_in(x_norm).chunk(2, dim=-1)
            ffn_out = self.ffn_out(F.silu(gate) * value)
        else:
            ffn_out = self.ffn(x_norm)
        ffn_out = self.dropout(ffn_out)
        if self.layer_scale_ffn is not None:
            ffn_out = ffn_out * self.layer_scale_ffn
        return x + self.residual_dropout(ffn_out)


class ScalarTransformerBlock(nn.Module):
    """Transformer block applied only to scalar channels."""
    def __init__(
        self,
        scalar_dim: int,
        num_heads: int,
        ffn_hidden: int,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        ffn_gated: bool = False,
        layer_scale_init: float | None = None,
        attention_chunk_size: int | None = None,
    ):
        super().__init__()
        self.block = TransformerBlock(
            scalar_dim,
            num_heads,
            ffn_hidden,
            dropout=dropout,
            residual_dropout=residual_dropout,
            ffn_gated=ffn_gated,
            layer_scale_init=layer_scale_init,
            attention_chunk_size=attention_chunk_size,
        )

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        scalars, rest = h[..., : self.block.norm1.normalized_shape[0]], h[..., self.block.norm1.normalized_shape[0] :]
        scalars = self.block(scalars, attn_mask=attn_mask)
        return torch.cat([scalars, rest], dim=-1)

def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per = torch.full(
        (num_nodes, logits.shape[-1]),
        float("-inf"),
        device=logits.device,
        dtype=logits.dtype,
    )
    max_per = max_per.scatter_reduce(0, expanded_index, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - max_per[index])
    sum_per = torch.zeros_like(max_per)
    sum_per = sum_per.scatter_reduce(0, expanded_index, exp, reduce="sum", include_self=True)
    return exp / (sum_per[index] + 1e-9)


class PointTransformerBlock(nn.Module):
    """Vector attention block from Point Transformer (Zhao et al., ICCV 2021)."""
    def __init__(
        self,
        scalar_dim: int,
        ffn_hidden: int,
        pos_hidden: int,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        ffn_gated: bool = False,
        layer_scale_init: float | None = None,
        rpe_bins: int = 0,
        rpe_scale: float = 1.0,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.norm1 = nn.LayerNorm(scalar_dim)
        self.phi = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.psi = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.alpha = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.delta = nn.Sequential(
            nn.Linear(3, pos_hidden),
            nn.ReLU(),
            nn.Linear(pos_hidden, scalar_dim),
        )
        self.delta_val = nn.Sequential(
            nn.Linear(3, pos_hidden),
            nn.ReLU(),
            nn.Linear(pos_hidden, scalar_dim),
        )
        self.gamma = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim),
            nn.ReLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.rpe_bins = int(rpe_bins)
        self.rpe_scale = float(rpe_scale)
        self.rpe_x = None
        self.rpe_y = None
        self.rpe_z = None
        if self.rpe_bins > 0:
            table_size = 2 * self.rpe_bins + 1
            self.rpe_x = nn.Embedding(table_size, scalar_dim)
            self.rpe_y = nn.Embedding(table_size, scalar_dim)
            self.rpe_z = nn.Embedding(table_size, scalar_dim)
        self.attn_out = nn.Linear(scalar_dim, scalar_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.layer_scale_attn = (
            nn.Parameter(torch.full((scalar_dim,), layer_scale_init))
            if layer_scale_init is not None
            else None
        )
        self.norm2 = nn.LayerNorm(scalar_dim)
        self.ffn_gated = ffn_gated
        if ffn_gated:
            self.ffn_in = nn.Linear(scalar_dim, ffn_hidden * 2)
            self.ffn_out = nn.Linear(ffn_hidden, scalar_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(scalar_dim, ffn_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, scalar_dim),
            )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((scalar_dim,), layer_scale_init))
            if layer_scale_init is not None
            else None
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(h)
        sender, receiver = edge_index
        rel_pos = pos[receiver] - pos[sender]
        rel = self.delta(rel_pos)
        if self.rpe_bins > 0:
            scaled = torch.round(rel_pos / self.rpe_scale).clamp(-self.rpe_bins, self.rpe_bins).to(torch.long)
            offset = self.rpe_bins
            rpe = (
                self.rpe_x(scaled[:, 0] + offset)
                + self.rpe_y(scaled[:, 1] + offset)
                + self.rpe_z(scaled[:, 2] + offset)
            )
            rel = rel + rpe
        q = self.phi(x)[receiver]
        k = self.psi(x)[sender]
        v = self.alpha(x)[sender]
        relation = q - k + rel
        attn = self.gamma(relation)
        attn = _segment_softmax(attn, receiver, x.shape[0])
        value = v + self.delta_val(rel_pos)
        out = torch.zeros_like(x)
        out.index_add_(0, receiver, attn * value)
        out = self.attn_out(out)
        if self.layer_scale_attn is not None:
            out = out * self.layer_scale_attn
        h = h + self.residual_dropout(out)

        x = self.norm2(h)
        if self.ffn_gated:
            gate, value = self.ffn_in(x).chunk(2, dim=-1)
            ffn_out = self.ffn_out(F.silu(gate) * value)
        else:
            ffn_out = self.ffn(x)
        ffn_out = self.dropout(ffn_out)
        if self.layer_scale_ffn is not None:
            ffn_out = ffn_out * self.layer_scale_ffn
        return h + self.residual_dropout(ffn_out)

class FlashACE(nn.Module):
    def __init__(
        self,
        r_max=5.0,
        l_max=0,
        num_radial=8,
        hidden_dim=128,
        num_layers=2,
        radial_basis_type: str = "bessel",
        radial_trainable: bool = False,
        envelope_exponent: int = 5,
        gaussian_width: float = 0.5,
        descriptor_passes: int = 1,
        descriptor_residual: bool = True,
        radial_mlp_hidden: int = 64,
        radial_mlp_layers: int = 2,
        pt_pos_hidden: int = 64,
        pt_ffn_hidden: int | None = None,
        pt_dropout: float = 0.0,
        pt_residual_dropout: float = 0.0,
        pt_ffn_gated: bool = False,
        pt_layer_scale_init: float | None = None,
        pt_rpe_bins: int = 0,
        pt_rpe_scale: float = 1.0,
        readout_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if l_max != 0:
            raise ValueError("IPA-only mode requires l_max=0 for invariant scalars.")
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.descriptor_passes = max(1, int(descriptor_passes))
        self.descriptor_residual = bool(descriptor_residual)

        self.emb = nn.Embedding(118, hidden_dim)
        self.ace = ACE_Descriptor(
            r_max,
            l_max,
            num_radial,
            hidden_dim,
            radial_basis_type=radial_basis_type,
            radial_trainable=radial_trainable,
            envelope_exponent=envelope_exponent,
            gaussian_width=gaussian_width,
            radial_mlp_hidden=radial_mlp_hidden,
            radial_mlp_layers=radial_mlp_layers,
        )
        scalar_ffn_hidden = pt_ffn_hidden or hidden_dim * 4
        self.pt_layers = nn.ModuleList(
            [
                PointTransformerBlock(
                    hidden_dim,
                    scalar_ffn_hidden,
                    pos_hidden=pt_pos_hidden,
                    dropout=pt_dropout,
                    residual_dropout=pt_residual_dropout,
                    ffn_gated=pt_ffn_gated,
                    layer_scale_init=pt_layer_scale_init,
                    rpe_bins=pt_rpe_bins,
                    rpe_scale=pt_rpe_scale,
                )
                for _ in range(num_layers)
            ]
        )

        readout_layers = []
        readout_dims = [hidden_dim]
        if readout_hidden_dims:
            readout_dims.extend([int(d) for d in readout_hidden_dims])
        for in_dim, out_dim in zip(readout_dims, readout_dims[1:]):
            readout_layers.append(nn.Linear(in_dim, out_dim))
            readout_layers.append(nn.SiLU())
        readout_layers.append(nn.Linear(readout_dims[-1], 1))
        self.readout = nn.Sequential(*readout_layers)

    def _ensure_self_edges(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            idx = torch.arange(num_nodes, device=device)
            return torch.stack([idx, idx], dim=0)
        self_edges = torch.arange(num_nodes, device=device)
        self_edges = torch.stack([self_edges, self_edges], dim=0)
        return torch.cat([edge_index, self_edges], dim=1)

    def forward(
        self,
        data,
        training: bool = False,
        temperature_scale: float = 1.0,
        detach_pos: bool = True,
        compute_stress: bool | None = None,
    ):
        z, pos, edge_index = data['z'], data['pos'], data['edge_index']
        cell_volume = data.get('volume', None)

        if compute_stress is None:
            compute_stress = training

        # We always need gradients w.r.t. atomic positions to compute forces.
        # Detach to ensure we work with a leaf tensor before enabling grads.
        if detach_pos:
            pos = pos.detach()
        pos.requires_grad_(True)

        if compute_stress and cell_volume is not None:
            # Parameterize the small-strain tensor symmetrically so the stress
            # we backpropagate through corresponds to the symmetric Cauchy
            # stress and does not pick up spurious rotational components. This
            # matches how ACE/MACE form stresses by differentiating with
            # respect to symmetric lattice strains.
            strain_params = torch.zeros(6, device=pos.device, requires_grad=True)

            epsilon = torch.zeros(3, 3, device=pos.device)
            epsilon[0, 0] = strain_params[0]
            epsilon[1, 1] = strain_params[1]
            epsilon[2, 2] = strain_params[2]
            epsilon[0, 1] = epsilon[1, 0] = strain_params[3]
            epsilon[0, 2] = epsilon[2, 0] = strain_params[4]
            epsilon[1, 2] = epsilon[2, 1] = strain_params[5]

            deformation = torch.eye(3, device=pos.device) + epsilon
            pos = pos @ deformation
        else:
            strain_params = None
            epsilon = None

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_len = torch.norm(edge_vec, dim=1)

        # 1. Descriptor iterations (optionally residual) before message passing / attention.
        h = self.emb(z)
        for i in range(self.descriptor_passes):
            scalars = h[..., : self.hidden_dim]
            desc = self.ace(scalars, edge_index, edge_vec, edge_len)
            if i == 0 or not self.descriptor_residual:
                h = desc
            else:
                h = h + desc

        edge_index = self._ensure_self_edges(edge_index, h.shape[0], h.device)
        scalars = h
        for layer in self.pt_layers:
            scalars = layer(scalars, pos, edge_index=edge_index)
        h = scalars
            
        # 2. Readout
        # Note: We extract only the scalar (L=0) features for energy
        # The optimized physics.py puts scalars first, so this slice is correct.
        scalars = h[:, :self.hidden_dim] 
        E = torch.sum(self.readout(scalars))

        aux = {}

        # 3. Derivatives
        # Avoid building second-order graphs during evaluation to reduce memory.
        grad_opts = {
            'create_graph': training,  # only keep graph for higher-order grads when training
            # Retain the graph whenever we also need stress so we can take an
            # additional derivative with respect to strain after forces.
            'retain_graph': epsilon is not None,
            'allow_unused': True,
        }

        grads = torch.autograd.grad(E, pos, **grad_opts)[0]
        F = -grads if grads is not None else torch.zeros_like(pos)
        
        S = torch.zeros(3, 3, device=pos.device)
        if compute_stress and epsilon is not None:
            # Retain the graph so the outer loss.backward() can still traverse
            # the computation graph built when taking the strain derivative.
            g_eps = torch.autograd.grad(
                E,
                strain_params,
                create_graph=training,
                retain_graph=training,
                allow_unused=True,
            )[0]
            if g_eps is not None:
                # Map the 6 unique components back to a symmetric stress tensor
                # and normalize by the deformed volume to avoid overestimating
                # stress under volumetric strain.
                stress = torch.zeros(3, 3, device=pos.device)
                stress[0, 0] = g_eps[0]
                stress[1, 1] = g_eps[1]
                stress[2, 2] = g_eps[2]
                stress[0, 1] = stress[1, 0] = g_eps[3]
                stress[0, 2] = stress[2, 0] = g_eps[4]
                stress[1, 2] = stress[2, 1] = g_eps[5]

                volume = cell_volume * torch.det(deformation)
                S = -stress / volume

        return E, F, S, aux
