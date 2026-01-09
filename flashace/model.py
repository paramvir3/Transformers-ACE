import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from .physics import ACE_Descriptor, ACERadialBasis, TACE_Descriptor

try:
    import torch_scatter
except Exception:
    torch_scatter = None

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

def _segment_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
    use_torch_scatter: bool,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    if use_torch_scatter and torch_scatter is not None:
        return torch_scatter.scatter_softmax(logits, index, dim=0)
    expanded_index = index[:, None].expand(-1, logits.shape[-1])
    max_per = torch.full(
        (num_nodes, logits.shape[-1]),
        float("-inf"),
        device=logits.device,
        dtype=logits.dtype,
    )
    max_per = max_per.scatter_reduce(0, expanded_index, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - max_per[index])
    sum_per = torch.zeros_like(max_per, dtype=exp.dtype)
    sum_per = sum_per.scatter_reduce(0, expanded_index, exp, reduce="sum", include_self=True)
    return exp / (sum_per[index] + 1e-9)


class PointTransformerBlock(nn.Module):
    """Vector attention block from Point Transformer (Zhao et al., ICCV 2021)."""
    def __init__(
        self,
        scalar_dim: int,
        ffn_hidden: int,
        pos_hidden: int,
        r_max: float,
        num_radial: int,
        l_max: int,
        radial_basis_type: str,
        radial_trainable: bool,
        envelope_exponent: int,
        gaussian_width: float,
        radial_mlp_hidden: int,
        radial_mlp_layers: int,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        ffn_gated: bool = False,
        layer_scale_init: float | None = None,
        rpe_bins: int = 0,
        rpe_scale: float = 1.0,
        use_torch_scatter: bool = False,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.norm1 = nn.LayerNorm(scalar_dim)
        self.phi = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.psi = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.alpha = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.radial_basis = ACERadialBasis(
            r_max=r_max,
            num_radial=num_radial,
            envelope_exponent=envelope_exponent,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        radial_mlp_hidden = max(1, int(radial_mlp_hidden))
        radial_mlp_layers = max(1, int(radial_mlp_layers))
        self.radial_gate = _make_mlp(
            in_dim=num_radial,
            hidden_dim=radial_mlp_hidden,
            out_dim=scalar_dim,
            depth=radial_mlp_layers,
        )
        self.sh_irreps = o3.Irreps.spherical_harmonics(l_max)
        self.sh = o3.SphericalHarmonics(self.sh_irreps, normalize=True, normalization="component")
        sh_dim = self.sh_irreps.dim
        self.sh_key = nn.Linear(sh_dim, scalar_dim, bias=False)
        self.sh_val = nn.Linear(sh_dim, scalar_dim, bias=False)
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
            nn.Linear(scalar_dim * 3 + num_radial, scalar_dim),
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
        self.use_torch_scatter = bool(use_torch_scatter)
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
        edge_len = torch.norm(rel_pos, dim=1)
        edge_dir = rel_pos / (edge_len.unsqueeze(-1) + 1e-9)
        sh = self.sh(edge_dir)
        radial_emb = self.radial_basis(edge_len)
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
        sh_k = self.sh_key(sh)
        sh_v = self.sh_val(sh)
        relation = torch.cat([q - k, rel, sh_k, radial_emb], dim=-1)
        attn = self.gamma(relation)
        attn = _segment_softmax(attn, receiver, x.shape[0], self.use_torch_scatter)
        gate = torch.sigmoid(self.radial_gate(radial_emb))
        value = (v + self.delta_val(rel_pos) + sh_v) * gate
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


def _build_radius_graph(pos: torch.Tensor, r_max: float) -> torch.Tensor:
    if pos.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=pos.device)
    dist = torch.cdist(pos, pos)
    mask = (dist <= r_max) & (dist > 0)
    sender, receiver = torch.where(mask)
    return torch.stack([sender, receiver], dim=0)


def _voxel_pool(pos: torch.Tensor, h: torch.Tensor, voxel_size: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pos.numel() == 0:
        return pos, h, torch.zeros((0,), dtype=torch.long, device=pos.device)
    coords = torch.floor(pos / voxel_size).to(torch.long)
    unique, inverse = torch.unique(coords, return_inverse=True, dim=0)
    pooled_pos = torch.zeros((unique.size(0), 3), device=pos.device, dtype=pos.dtype)
    pooled_h = torch.zeros((unique.size(0), h.size(1)), device=h.device, dtype=h.dtype)
    pooled_pos = pooled_pos.index_add(0, inverse, pos)
    pooled_h = pooled_h.index_add(0, inverse, h)
    counts = torch.bincount(inverse, minlength=unique.size(0)).to(pos.dtype).unsqueeze(-1)
    pooled_pos = pooled_pos / counts
    pooled_h = pooled_h / counts
    return pooled_pos, pooled_h, inverse


class EquivariantPointTransformerBlock(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        r_max: float,
        num_radial: int,
        l_max: int,
        radial_basis_type: str,
        radial_trainable: bool,
        envelope_exponent: int,
        gaussian_width: float,
        radial_mlp_hidden: int,
        radial_mlp_layers: int,
        attn_hidden: int,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        use_torch_scatter: bool = False,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.r_max = r_max
        self.irreps = o3.Irreps(f"{scalar_dim}x0e + {vector_dim}x1o")
        self.norm = IrrepRMSNorm(self.irreps)
        self.sh_irreps = o3.Irreps.spherical_harmonics(l_max)
        self.sh = o3.SphericalHarmonics(self.sh_irreps, normalize=True, normalization="component")
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.sh_irreps,
            self.irreps,
            internal_weights=False,
            shared_weights=False,
        )
        radial_mlp_hidden = max(1, int(radial_mlp_hidden))
        radial_mlp_layers = max(1, int(radial_mlp_layers))
        self.radial_basis = ACERadialBasis(
            r_max=r_max,
            num_radial=num_radial,
            envelope_exponent=envelope_exponent,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        self.radial_mlp = _make_mlp(
            in_dim=num_radial,
            hidden_dim=radial_mlp_hidden,
            out_dim=self.tp.weight_numel,
            depth=radial_mlp_layers,
        )
        self.radial_gate = _make_mlp(
            in_dim=num_radial,
            hidden_dim=radial_mlp_hidden,
            out_dim=self.irreps.dim,
            depth=radial_mlp_layers,
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(scalar_dim + num_radial, attn_hidden),
            nn.SiLU(),
            nn.Linear(attn_hidden, 1),
        )
        self.use_torch_scatter = bool(use_torch_scatter)
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h
        h_norm = self.norm(h)
        rel_pos = pos[receiver] - pos[sender]
        edge_len = torch.norm(rel_pos, dim=1)
        edge_dir = rel_pos / (edge_len.unsqueeze(-1) + 1e-9)
        sh = self.sh(edge_dir)
        radial_emb = self.radial_basis(edge_len)
        weights = self.radial_mlp(radial_emb)
        msg = self.tp(h_norm[sender], sh, weights)
        gate = torch.sigmoid(self.radial_gate(radial_emb))
        msg = msg * gate
        scalars = h_norm[:, : self.scalar_dim]
        attn_in = torch.cat([scalars[receiver] - scalars[sender], radial_emb], dim=-1)
        attn_logits = self.attn_mlp(attn_in)
        attn = _segment_softmax(attn_logits, receiver, h.shape[0], self.use_torch_scatter)
        out = torch.zeros_like(h)
        out.index_add_(0, receiver, attn * msg)
        out = self.dropout(out)
        return h + self.residual_dropout(out)


def _irrep_squared_norm(irreps: o3.Irreps, x: torch.Tensor) -> torch.Tensor:
    start = 0
    total = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    for mul, ir in irreps:
        dim = mul * ir.dim
        block = x[:, start : start + dim]
        total = total + block.pow(2).sum(dim=-1, keepdim=True)
        start += dim
    return total


class FactorizedPointTransformerBlock(nn.Module):
    """Point Transformer block with explicit radial/angular separation."""
    def __init__(
        self,
        scalar_dim: int,
        r_max: float,
        num_radial: int,
        l_max: int,
        radial_basis_type: str,
        radial_trainable: bool,
        envelope_exponent: int,
        gaussian_width: float,
        radial_mlp_hidden: int,
        radial_mlp_layers: int,
        score_hidden: int,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        ffn_gated: bool = False,
        layer_scale_init: float | None = None,
        use_torch_scatter: bool = False,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.norm1 = nn.LayerNorm(scalar_dim)
        self.q_proj = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.k_proj = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.v_proj = nn.Linear(scalar_dim, scalar_dim, bias=False)

        self.irreps_node = o3.Irreps(f"{scalar_dim}x0e")
        self.sh_irreps = o3.Irreps.spherical_harmonics(l_max)
        self.edge_irreps = o3.Irreps([(scalar_dim, ir) for _, ir in self.sh_irreps])
        self.sh = o3.SphericalHarmonics(self.sh_irreps, normalize=True, normalization="component")

        self.tp_k = o3.FullyConnectedTensorProduct(self.irreps_node, self.sh_irreps, self.edge_irreps)
        self.tp_v = o3.FullyConnectedTensorProduct(self.irreps_node, self.sh_irreps, self.edge_irreps)

        self.radial_basis = ACERadialBasis(
            r_max,
            num_radial,
            envelope_exponent=envelope_exponent,
            basis_type=radial_basis_type,
            trainable=radial_trainable,
            gaussian_width=gaussian_width,
        )
        radial_mlp_hidden = max(1, int(radial_mlp_hidden))
        radial_mlp_layers = max(1, int(radial_mlp_layers))
        self.radial_gate_k = _make_mlp(
            in_dim=num_radial,
            hidden_dim=radial_mlp_hidden,
            out_dim=self.edge_irreps.dim,
            depth=radial_mlp_layers,
        )
        self.radial_gate_v = _make_mlp(
            in_dim=num_radial,
            hidden_dim=radial_mlp_hidden,
            out_dim=self.edge_irreps.dim,
            depth=radial_mlp_layers,
        )
        self.radial_bias = _make_mlp(
            in_dim=num_radial,
            hidden_dim=radial_mlp_hidden,
            out_dim=1,
            depth=radial_mlp_layers,
        )

        self.score_mlp = nn.Sequential(
            nn.Linear(1, score_hidden),
            nn.SiLU(),
            nn.Linear(score_hidden, 1),
        )
        self.use_torch_scatter = bool(use_torch_scatter)
        self.key_proj = o3.Linear(self.edge_irreps, self.irreps_node)
        self.out_proj = o3.Linear(self.edge_irreps, self.irreps_node)
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
            self.ffn_in = nn.Linear(scalar_dim, scalar_dim * 2)
            self.ffn_out = nn.Linear(scalar_dim, scalar_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(scalar_dim * 4, scalar_dim),
            )
        self.layer_scale_ffn = (
            nn.Parameter(torch.full((scalar_dim,), layer_scale_init))
            if layer_scale_init is not None
            else None
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.norm1(h)
        sender, receiver = edge_index
        if sender.numel() == 0:
            return h

        rel_pos = pos[receiver] - pos[sender]
        edge_len = torch.norm(rel_pos, dim=1)
        edge_dir = rel_pos / (edge_len.unsqueeze(-1) + 1e-9)
        sh = self.sh(edge_dir)

        q_nodes = self.q_proj(x)
        k_nodes = self.k_proj(x)
        v_nodes = self.v_proj(x)

        q = q_nodes[receiver]
        k_edge = self.tp_k(k_nodes[sender], sh)
        v_edge = self.tp_v(v_nodes[sender], sh)

        radial_emb = self.radial_basis(edge_len)
        k_gate = self.radial_gate_k(radial_emb)
        v_gate = self.radial_gate_v(radial_emb)
        k_edge = k_edge * k_gate
        v_edge = v_edge * v_gate

        k_proj = self.key_proj(k_edge)
        delta = q - k_proj
        delta_norm = _irrep_squared_norm(self.irreps_node, delta)
        attn_logits = self.score_mlp(delta_norm) + self.radial_bias(radial_emb)
        attn = _segment_softmax(attn_logits, receiver, x.shape[0], self.use_torch_scatter)

        out = torch.zeros((x.shape[0], self.edge_irreps.dim), device=x.device, dtype=x.dtype)
        out.index_add_(0, receiver, attn * v_edge)
        out = self.out_proj(out)
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


def _expand_list(value: int | list[int], length: int, name: str) -> list[int]:
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"{name} must have length {length}, got {len(value)}")
        return [int(v) for v in value]
    return [int(value)] * length


class TACEBackbone(nn.Module):
    def __init__(
        self,
        r_max: float,
        hidden_dim: int,
        atomic_numbers: list[int],
        num_radial: int,
        num_layers: int,
        Lmax: int | list[int],
        lmax: int | list[int],
        avg_num_neighbors: int = 64,
        num_channel: int | list[int] = 64,
        num_channel_hidden: int | list[int] = 64,
        radial_basis: dict | None = None,
        angular_basis: dict | None = None,
        radial_mlp: dict | None = None,
        inter: dict | None = None,
        prod: dict | None = None,
        universal_embedding: dict | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        self.atomic_number_to_index = {int(z): i for i, z in enumerate(self.atomic_numbers)}
        self.num_layers = int(num_layers)
        Lmax_list = _expand_list(Lmax, self.num_layers, "tace_Lmax")
        lmax_list = _expand_list(lmax, self.num_layers, "tace_lmax")
        num_channel_list = _expand_list(num_channel, self.num_layers, "tace_num_channel")
        num_channel_hidden_list = _expand_list(num_channel_hidden, self.num_layers, "tace_num_channel_hidden")
        radial_basis = radial_basis or {}
        radial_mlp = radial_mlp or {}
        self.descriptor = TACE_Descriptor(
            r_max=r_max,
            num_radial=num_radial,
            lmax=max(lmax_list),
            node_dim=num_channel_list[-1],
            hidden_dim=num_channel_hidden_list[-1],
            radial_basis_type=radial_basis.get("radial_basis", "bessel"),
            radial_trainable=radial_basis.get("trainable", False),
            envelope_exponent=radial_basis.get("polynomial_cutoff", 5),
            gaussian_width=radial_basis.get("gaussian_width", 0.5),
            radial_mlp_hidden=radial_mlp.get("hidden_dim", 64),
            radial_mlp_layers=radial_mlp.get("num_layers", 2),
        )
        self.proj = nn.Linear(num_channel_hidden_list[-1], hidden_dim)

    def _one_hot(self, z: torch.Tensor) -> torch.Tensor:
        num_atoms = z.shape[0]
        out = torch.zeros(
            (num_atoms, len(self.atomic_numbers)),
            dtype=torch.get_default_dtype(),
            device=z.device,
        )
        for idx, zi in enumerate(z.tolist()):
            if zi not in self.atomic_number_to_index:
                raise ValueError(f"Unknown atomic number {zi} for TACE backbone")
            out[idx, self.atomic_number_to_index[zi]] = 1.0
        return out

    def forward(self, z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_attrs = self._one_hot(z)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_len = torch.norm(edge_vec, dim=1)
        desc = self.descriptor(node_attrs, edge_index, edge_vec, edge_len)
        return self.proj(desc)

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
        pt_use_torch_scatter: bool = False,
        pt_factorized: bool = False,
        pt_l_max: int | None = None,
        pt_radial_mlp_hidden: int = 64,
        pt_radial_mlp_layers: int = 2,
        readout_hidden_dims: list[int] | None = None,
        descriptor_backend: str = "tace",
        tace_atomic_numbers: list[int] | None = None,
        tace_num_layers: int = 2,
        tace_Lmax: int | list[int] = 1,
        tace_lmax: int | list[int] = 1,
        tace_avg_num_neighbors: int = 64,
        tace_num_channel: int | list[int] = 64,
        tace_num_channel_hidden: int | list[int] = 64,
        tace_radial_basis: dict | None = None,
        tace_angular_basis: dict | None = None,
        tace_radial_mlp: dict | None = None,
        tace_inter: dict | None = None,
        tace_prod: dict | None = None,
        tace_universal_embedding: dict | None = None,
        pt_equivariant: bool = False,
        pt_vector_dim: int = 32,
        pt_hierarchical: bool = False,
        pt_voxel_size: float = 2.0,
        pt_long_range: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.descriptor_passes = max(1, int(descriptor_passes))
        self.descriptor_residual = bool(descriptor_residual)
        self.descriptor_backend = descriptor_backend
        self.pt_equivariant = pt_equivariant
        self.pt_hierarchical = pt_hierarchical
        self.pt_voxel_size = float(pt_voxel_size)
        self.pt_long_range = pt_long_range
        self.pt_vector_dim = int(pt_vector_dim)

        if self.descriptor_backend == "tace":
            if not tace_atomic_numbers:
                raise ValueError("tace_atomic_numbers must be provided when using descriptor_backend='tace'.")
            self.tace_backbone = TACEBackbone(
                r_max=r_max,
                hidden_dim=hidden_dim,
                num_radial=num_radial,
                atomic_numbers=tace_atomic_numbers,
                num_layers=tace_num_layers,
                Lmax=tace_Lmax,
                lmax=tace_lmax,
                avg_num_neighbors=tace_avg_num_neighbors,
                num_channel=tace_num_channel,
                num_channel_hidden=tace_num_channel_hidden,
                radial_basis=tace_radial_basis,
                angular_basis=tace_angular_basis,
                radial_mlp=tace_radial_mlp,
                inter=tace_inter,
                prod=tace_prod,
                universal_embedding=tace_universal_embedding,
            )
            self.emb = None
            self.ace = None
        elif self.descriptor_backend == "pt":
            self.emb = nn.Embedding(118, hidden_dim)
            self.ace = None
        else:
            raise ValueError("descriptor_backend must be 'tace' or 'pt'.")
        scalar_ffn_hidden = pt_ffn_hidden or hidden_dim * 4
        self.pt_layers = nn.ModuleList([])
        pt_l_max = l_max if pt_l_max is None else int(pt_l_max)
        for _ in range(num_layers):
            if pt_equivariant:
                self.pt_layers.append(
                    EquivariantPointTransformerBlock(
                        scalar_dim=hidden_dim,
                        vector_dim=pt_vector_dim,
                        r_max=r_max,
                        num_radial=num_radial,
                        l_max=pt_l_max,
                        radial_basis_type=radial_basis_type,
                        radial_trainable=radial_trainable,
                        envelope_exponent=envelope_exponent,
                        gaussian_width=gaussian_width,
                        radial_mlp_hidden=pt_radial_mlp_hidden,
                        radial_mlp_layers=pt_radial_mlp_layers,
                        attn_hidden=scalar_ffn_hidden,
                        dropout=pt_dropout,
                        residual_dropout=pt_residual_dropout,
                        use_torch_scatter=pt_use_torch_scatter,
                    )
                )
            elif pt_factorized:
                self.pt_layers.append(
                    FactorizedPointTransformerBlock(
                        scalar_dim=hidden_dim,
                        r_max=r_max,
                        num_radial=num_radial,
                        l_max=pt_l_max,
                        radial_basis_type=radial_basis_type,
                        radial_trainable=radial_trainable,
                        envelope_exponent=envelope_exponent,
                        gaussian_width=gaussian_width,
                        radial_mlp_hidden=pt_radial_mlp_hidden,
                        radial_mlp_layers=pt_radial_mlp_layers,
                        score_hidden=scalar_ffn_hidden,
                        dropout=pt_dropout,
                        residual_dropout=pt_residual_dropout,
                        ffn_gated=pt_ffn_gated,
                        layer_scale_init=pt_layer_scale_init,
                        use_torch_scatter=pt_use_torch_scatter,
                    )
                )
            else:
                self.pt_layers.append(
                    PointTransformerBlock(
                        hidden_dim,
                        scalar_ffn_hidden,
                        pos_hidden=pt_pos_hidden,
                        r_max=r_max,
                        num_radial=num_radial,
                        l_max=pt_l_max,
                        radial_basis_type=radial_basis_type,
                        radial_trainable=radial_trainable,
                        envelope_exponent=envelope_exponent,
                        gaussian_width=gaussian_width,
                        radial_mlp_hidden=pt_radial_mlp_hidden,
                        radial_mlp_layers=pt_radial_mlp_layers,
                        dropout=pt_dropout,
                        residual_dropout=pt_residual_dropout,
                        ffn_gated=pt_ffn_gated,
                        layer_scale_init=pt_layer_scale_init,
                        rpe_bins=pt_rpe_bins,
                        rpe_scale=pt_rpe_scale,
                        use_torch_scatter=pt_use_torch_scatter,
                    )
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

        # 1. Descriptor + PT stack.
        if self.descriptor_backend == "tace":
            h = self.tace_backbone(z, pos, edge_index)
        else:
            h = self.emb(z)
        if self.pt_equivariant:
            zeros = torch.zeros((h.shape[0], self.pt_vector_dim * 3), device=h.device, dtype=h.dtype)
            h = torch.cat([h, zeros], dim=-1)

        edge_index = self._ensure_self_edges(edge_index, h.shape[0], h.device)
        if self.pt_hierarchical:
            pooled_pos, pooled_h, inverse = _voxel_pool(pos, h, self.pt_voxel_size)
            coarse_edges = _build_radius_graph(pooled_pos, self.r_max)
            coarse_edges = self._ensure_self_edges(coarse_edges, pooled_h.shape[0], pooled_h.device)
            for layer in self.pt_layers:
                pooled_h = layer(pooled_h, pooled_pos, edge_index=coarse_edges)
            if self.pt_long_range:
                pooled_h = pooled_h + pooled_h.mean(dim=0, keepdim=True)
            h = h + pooled_h[inverse]
        for layer in self.pt_layers:
            h = layer(h, pos, edge_index=edge_index)
        if self.pt_long_range:
            h = h + h.mean(dim=0, keepdim=True)
            
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
