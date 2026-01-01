import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from .physics import ACE_Descriptor

try:
    from flash_ipa import flash_ipa as _flash_ipa_kernel
except Exception:
    _flash_ipa_kernel = None

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

class InvariantPointAttentionBlock(nn.Module):
    """IPA-style attention on scalar channels with point projections."""
    def __init__(
        self,
        scalar_dim: int,
        num_heads: int,
        num_points: int,
        ffn_hidden: int,
        dropout: float = 0.0,
        residual_dropout: float = 0.0,
        ffn_gated: bool = False,
        layer_scale_init: float | None = None,
        point_weight_init: float = 1.0,
        point_scale: float = 1.0,
        point_dropout: float = 0.0,
        bias_norm: bool = True,
        logit_clip: float | None = None,
        use_flash_ipa: bool = False,
        attn_logit_scale_init: float = 1.0,
    ):
        super().__init__()
        if scalar_dim % num_heads != 0:
            raise ValueError(
                "num_heads must divide scalar_dim for IPA "
                f"(scalar_dim={scalar_dim}, heads={num_heads})."
            )
        self.scalar_dim = scalar_dim
        self.num_heads = num_heads
        self.head_dim = scalar_dim // num_heads
        self.num_points = num_points
        self.point_scale = float(point_scale)
        self.point_dropout = nn.Dropout(point_dropout)
        self.bias_norm = bool(bias_norm)
        self.logit_clip = logit_clip
        self.use_flash_ipa = bool(use_flash_ipa) and _flash_ipa_kernel is not None
        self.flash_ipa = _flash_ipa_kernel if self.use_flash_ipa else None
        self.attn_logit_scale = nn.Parameter(
            torch.full((num_heads,), float(attn_logit_scale_init))
        )
        self.norm1 = nn.LayerNorm(scalar_dim)
        self.q = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.k = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.v = nn.Linear(scalar_dim, scalar_dim, bias=False)
        self.q_pts = nn.Linear(scalar_dim, num_heads * num_points * 3, bias=False)
        self.k_pts = nn.Linear(scalar_dim, num_heads * num_points * 3, bias=False)
        self.v_pts = nn.Linear(scalar_dim, num_heads * num_points * 3, bias=False)
        self.point_weight = nn.Parameter(torch.full((num_heads,), float(point_weight_init)))
        self.attn_out = nn.Linear(scalar_dim, scalar_dim)
        self.point_out = nn.Linear(num_heads * num_points * 3, scalar_dim)
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
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(h)
        n_nodes = x.shape[0]
        q = self.q(x).reshape(n_nodes, self.num_heads, self.head_dim)
        k = self.k(x).reshape(n_nodes, self.num_heads, self.head_dim)
        v = self.v(x).reshape(n_nodes, self.num_heads, self.head_dim)
        q_pts = self.q_pts(x).reshape(n_nodes, self.num_heads, self.num_points, 3)
        k_pts = self.k_pts(x).reshape(n_nodes, self.num_heads, self.num_points, 3)
        v_pts = self.v_pts(x).reshape(n_nodes, self.num_heads, self.num_points, 3)
        v_pts = self.point_dropout(v_pts)

        if self.flash_ipa is not None:
            try:
                out, out_pts = self.flash_ipa(
                    q,
                    k,
                    v,
                    q_pts,
                    k_pts,
                    v_pts,
                    attn_mask=attn_mask,
                    point_weight=self.point_weight,
                    point_scale=self.point_scale,
                )
            except Exception:
                out = out_pts = None
        else:
            out = out_pts = None

        if out is None:
            attn_logits = torch.einsum("ihd,jhd->hij", q, k)
            attn_logits = attn_logits * self.attn_logit_scale[:, None, None]
            attn_logits = attn_logits / (self.head_dim ** 0.5)
            q_pts_h = q_pts.permute(1, 0, 2, 3)
            k_pts_h = k_pts.permute(1, 0, 2, 3)
            diff = q_pts_h[:, :, None, :, :] - k_pts_h[:, None, :, :, :]
            dist2 = (diff ** 2).sum(dim=(-1, -2))
            if self.bias_norm and self.num_points > 0:
                dist2 = dist2 / float(self.num_points)
            dist2 = dist2 * self.point_weight[:, None, None]
            attn_logits = attn_logits - 0.5 * self.point_scale * dist2
            if self.logit_clip is not None:
                attn_logits = attn_logits.clamp(-self.logit_clip, self.logit_clip)
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_logits = attn_logits.masked_fill(attn_mask[None, ...], float("-inf"))
                else:
                    attn_logits = attn_logits + attn_mask[None, ...]

            attn = torch.softmax(attn_logits, dim=-1)
            attn = self.dropout(attn)
            out = torch.einsum("hij,jhd->ihd", attn, v).reshape(n_nodes, self.scalar_dim)
            out_pts = torch.einsum("hij,jhpd->ihpd", attn, v_pts).reshape(
                n_nodes, self.num_heads * self.num_points * 3
            )
        out = self.attn_out(out) + self.point_out(out_pts)
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
        ipa_num_heads: int = 4,
        ipa_num_points: int = 4,
        ipa_point_weight_init: float = 1.0,
        ipa_point_scale: float = 1.0,
        ipa_point_dropout: float = 0.0,
        ipa_bias_norm: bool = True,
        ipa_logit_clip: float | None = None,
        ipa_use_flash: bool = False,
        ipa_attn_logit_scale_init: float = 1.0,
        ipa_ffn_hidden: int | None = None,
        ipa_dropout: float = 0.0,
        ipa_residual_dropout: float = 0.0,
        ipa_ffn_gated: bool = False,
        ipa_layer_scale_init: float | None = None,
        readout_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if l_max != 0:
            raise ValueError("IPA-only mode requires l_max=0 for invariant scalars.")
        if hidden_dim % ipa_num_heads != 0:
            raise ValueError(
                "ipa_num_heads must divide hidden_dim "
                f"(hidden_dim={hidden_dim}, heads={ipa_num_heads})."
            )
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
        scalar_ffn_hidden = ipa_ffn_hidden or hidden_dim * 4
        self.ipa_layers = nn.ModuleList(
            [
                InvariantPointAttentionBlock(
                    hidden_dim,
                    ipa_num_heads,
                    ipa_num_points,
                    scalar_ffn_hidden,
                    dropout=ipa_dropout,
                    residual_dropout=ipa_residual_dropout,
                    ffn_gated=ipa_ffn_gated,
                    layer_scale_init=ipa_layer_scale_init,
                    point_weight_init=ipa_point_weight_init,
                    point_scale=ipa_point_scale,
                    point_dropout=ipa_point_dropout,
                    bias_norm=ipa_bias_norm,
                    logit_clip=ipa_logit_clip,
                    use_flash_ipa=ipa_use_flash,
                    attn_logit_scale_init=ipa_attn_logit_scale_init,
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

    def _build_attention_mask(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        attn_mask = torch.ones((num_nodes, num_nodes), device=device, dtype=torch.bool)
        idx = torch.arange(num_nodes, device=device)
        attn_mask[idx, idx] = False
        if edge_index.numel() > 0:
            attn_mask[edge_index[0], edge_index[1]] = False
        return attn_mask

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

        attn_mask = self._build_attention_mask(h.shape[0], edge_index, h.device)
        scalars = h
        for layer in self.ipa_layers:
            scalars = layer(scalars, pos, attn_mask=attn_mask)
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
