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
        self.sh = o3.SphericalHarmonics(l_max, normalize=True, normalization="component")
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps,
            self.sh_irreps,
            self.irreps,
            internal_weights=False,
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

class FlashACE(nn.Module):
    def __init__(
        self,
        r_max=5.0,
        l_max=2,
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
        transformer_num_heads: int = 4,
        transformer_ffn_hidden: int | None = None,
        transformer_dropout: float = 0.0,
        transformer_residual_dropout: float = 0.0,
        transformer_ffn_gated: bool = False,
        transformer_layer_scale_init: float | None = None,
        transformer_attention_chunk_size: int | None = None,
        use_transformer: bool = True,
        transformer_scalar_only: bool = False,
        attention_neighbor_mask: bool = False,
        attention_short_range: bool = False,
        attention_short_range_ratio: float = 0.5,
        attention_short_range_gate: bool = True,
        use_aux_force_head: bool = True,
        use_aux_stress_head: bool = True,
        message_passing_layers: int = 0,
        interleave_descriptor: bool = False,
        edge_update_per_layer: bool = False,
        node_update_mlp: bool = False,
        equivariant_mix_per_layer: bool = False,
        edge_state_dim: int | None = None,
        edge_attention: bool = False,
        equivariant_rms_norm: bool = False,
        equivariant_rms_norm_eps: float = 1e-8,
        readout_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.l_max = l_max
        self.descriptor_passes = max(1, int(descriptor_passes))
        self.descriptor_residual = bool(descriptor_residual)
        self.transformer_num_heads = max(1, int(transformer_num_heads))
        self.transformer_ffn_hidden = transformer_ffn_hidden
        self.transformer_dropout = float(transformer_dropout)
        self.transformer_residual_dropout = float(transformer_residual_dropout)
        self.transformer_ffn_gated = bool(transformer_ffn_gated)
        self.transformer_layer_scale_init = transformer_layer_scale_init
        self.transformer_attention_chunk_size = transformer_attention_chunk_size
        self.use_transformer = bool(use_transformer)
        self.transformer_scalar_only = bool(transformer_scalar_only)
        self.attention_neighbor_mask = bool(attention_neighbor_mask)
        self.attention_short_range = bool(attention_short_range)
        self.attention_short_range_ratio = float(attention_short_range_ratio)
        self.attention_short_range_gate = bool(attention_short_range_gate)
        self.use_aux_force_head = use_aux_force_head
        self.use_aux_stress_head = use_aux_stress_head
        self.message_passing_layers = max(0, int(message_passing_layers))
        self.interleave_descriptor = bool(interleave_descriptor)
        self.edge_update_per_layer = bool(edge_update_per_layer)
        self.node_update_mlp = bool(node_update_mlp)
        self.equivariant_mix_per_layer = bool(equivariant_mix_per_layer)
        self.edge_attention = bool(edge_attention)
        self.edge_state_dim = edge_state_dim or hidden_dim
        self.equivariant_rms_norm = bool(equivariant_rms_norm)
        self.equivariant_rms_norm_eps = float(equivariant_rms_norm_eps)
        self.readout_hidden_dims = readout_hidden_dims
        self.node_scalar_irreps = o3.Irreps(f"{hidden_dim}x0e")

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
        self.attention_irreps = self.ace.irreps_out

        self.mp_layers = nn.ModuleList(
            [ScalarMessagePassing(hidden_dim) for _ in range(self.message_passing_layers)]
        )
        self.edge_updates = nn.ModuleList(
            [EdgeUpdate(hidden_dim) for _ in range(num_layers)] if self.edge_update_per_layer else []
        )
        self.node_updates = nn.ModuleList(
            [NodeUpdateMLP(hidden_dim) for _ in range(num_layers)] if self.node_update_mlp else []
        )
        self.eq_mixes = nn.ModuleList(
            [
                EquivariantMixBlock(
                    self.attention_irreps,
                    hidden_dim,
                    l_max,
                    radial_mlp_hidden,
                )
                for _ in range(num_layers)
            ]
            if self.equivariant_mix_per_layer
            else []
        )
        self.eq_norms = nn.ModuleList(
            [IrrepRMSNorm(self.attention_irreps, eps=self.equivariant_rms_norm_eps) for _ in range(num_layers)]
            if self.equivariant_rms_norm
            else []
        )
        self.edge_state_init = None
        self.edge_state_updates = nn.ModuleList()
        self.edge_bias_proj = None
        if self.edge_attention:
            self.edge_state_init = EdgeStateInit(hidden_dim, self.edge_state_dim)
            self.edge_state_updates = nn.ModuleList(
                [EdgeStateUpdate(hidden_dim, self.edge_state_dim) for _ in range(num_layers)]
            )
            self.edge_bias_proj = nn.Linear(self.edge_state_dim, 1)
        ffn_hidden = self.transformer_ffn_hidden or self.attention_irreps.dim * 4
        if self.use_transformer:
            if self.transformer_scalar_only:
                if self.hidden_dim % self.transformer_num_heads != 0:
                    raise ValueError(
                        "transformer_num_heads must divide hidden_dim when transformer_scalar_only=True "
                        f"(hidden_dim={self.hidden_dim}, heads={self.transformer_num_heads})."
                    )
                scalar_ffn_hidden = self.transformer_ffn_hidden or self.hidden_dim * 4
                self.layers = nn.ModuleList(
                    [
                        ScalarTransformerBlock(
                            self.hidden_dim,
                            self.transformer_num_heads,
                            scalar_ffn_hidden,
                            dropout=self.transformer_dropout,
                            residual_dropout=self.transformer_residual_dropout,
                            ffn_gated=self.transformer_ffn_gated,
                            layer_scale_init=self.transformer_layer_scale_init,
                            attention_chunk_size=self.transformer_attention_chunk_size,
                        )
                        for i in range(num_layers)
                    ]
                )
            else:
                if self.attention_irreps.dim % self.transformer_num_heads != 0:
                    raise ValueError(
                        "transformer_num_heads must divide attention_irreps.dim when transformer_scalar_only=False "
                        f"(attention_dim={self.attention_irreps.dim}, heads={self.transformer_num_heads}). "
                        "Adjust transformer_num_heads, l_max/hidden_dim, or enable transformer_scalar_only."
                    )
                self.layers = nn.ModuleList(
                    [
                        TransformerBlock(
                            self.attention_irreps.dim,
                            self.transformer_num_heads,
                            ffn_hidden,
                            dropout=self.transformer_dropout,
                            residual_dropout=self.transformer_residual_dropout,
                            ffn_gated=self.transformer_ffn_gated,
                            layer_scale_init=self.transformer_layer_scale_init,
                            attention_chunk_size=self.transformer_attention_chunk_size,
                        )
                        for i in range(num_layers)
                    ]
                )
        else:
            self.layers = nn.ModuleList()
        self.short_range_gate = None
        if self.use_transformer and self.attention_short_range and self.attention_short_range_gate:
            self.short_range_gate = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )
        
        readout_layers = []
        readout_dims = [hidden_dim]
        if self.readout_hidden_dims:
            readout_dims.extend([int(d) for d in self.readout_hidden_dims])
        for in_dim, out_dim in zip(readout_dims, readout_dims[1:]):
            readout_layers.append(nn.Linear(in_dim, out_dim))
            readout_layers.append(nn.SiLU())
        readout_layers.append(nn.Linear(readout_dims[-1], 1))
        self.readout = nn.Sequential(*readout_layers)
        self.aux_force_head = None
        self.aux_stress_head = None
        if self.use_aux_force_head:
            self.aux_force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )
        if self.use_aux_stress_head:
            self.aux_stress_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 6),
            )

    def _build_attention_bias(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_bias: torch.Tensor | None,
        base_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if base_mask is None and edge_bias is None:
            return None
        attn_bias = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
        if base_mask is not None:
            attn_bias = attn_bias.masked_fill(base_mask, float("-inf"))
        if edge_bias is not None and edge_bias.numel() > 0:
            attn_bias[edge_index[0], edge_index[1]] = edge_bias
        return attn_bias

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

        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_len)

        edge_state = None
        if self.edge_attention and self.edge_state_init is not None:
            edge_state = self.edge_state_init(h[..., : self.hidden_dim], edge_index, edge_len)

        attn_mask = None
        attn_mask_short = None
        if self.use_transformer and self.attention_neighbor_mask:
            num_nodes = h.shape[0]
            attn_mask = torch.ones((num_nodes, num_nodes), device=h.device, dtype=torch.bool)
            idx = torch.arange(num_nodes, device=h.device)
            attn_mask[idx, idx] = False
            attn_mask[edge_index[0], edge_index[1]] = False
            if self.attention_short_range:
                if not (0.0 < self.attention_short_range_ratio <= 1.0):
                    raise ValueError("attention_short_range_ratio must be in (0, 1]")
                cutoff = self.r_max * self.attention_short_range_ratio
                short_edges = edge_len <= cutoff
                attn_mask_short = torch.ones((num_nodes, num_nodes), device=h.device, dtype=torch.bool)
                attn_mask_short[idx, idx] = False
                if short_edges.any():
                    attn_mask_short[edge_index[0][short_edges], edge_index[1][short_edges]] = False

        for idx, layer in enumerate(self.layers):
            if self.equivariant_rms_norm and len(self.eq_norms) > 0:
                h = self.eq_norms[idx](h)
            if self.interleave_descriptor:
                scalars = h[..., : self.hidden_dim]
                desc = self.ace(scalars, edge_index, edge_vec, edge_len)
                h = h + desc if self.descriptor_residual else desc
            if self.edge_update_per_layer and len(self.edge_updates) > 0:
                h = self.edge_updates[idx](h, edge_index, edge_len)
            if self.equivariant_mix_per_layer and len(self.eq_mixes) > 0:
                h = self.eq_mixes[idx](h, edge_index, edge_vec, edge_len)
            edge_bias = None
            if self.edge_attention and self.edge_bias_proj is not None:
                if edge_state is None:
                    edge_state = self.edge_state_init(h[..., : self.hidden_dim], edge_index, edge_len)
                else:
                    edge_state = self.edge_state_updates[idx](
                        h[..., : self.hidden_dim], edge_index, edge_len, edge_state
                    )
                edge_bias = self.edge_bias_proj(edge_state).squeeze(-1)
                if edge_bias.dtype != h.dtype:
                    edge_bias = edge_bias.to(h.dtype)
            if attn_mask_short is not None:
                long_bias = self._build_attention_bias(
                    h.shape[0],
                    edge_index,
                    edge_bias,
                    attn_mask,
                    h.dtype,
                    h.device,
                )
                short_bias = self._build_attention_bias(
                    h.shape[0],
                    edge_index,
                    edge_bias,
                    attn_mask_short,
                    h.dtype,
                    h.device,
                )
                long_out = layer(h, attn_mask=long_bias)
                short_out = layer(h, attn_mask=short_bias)
                if self.short_range_gate is None:
                    h = 0.5 * (short_out + long_out)
                else:
                    gate = self.short_range_gate(h[..., : self.hidden_dim])
                    h = gate * short_out + (1.0 - gate) * long_out
            else:
                attn_bias = self._build_attention_bias(
                    h.shape[0],
                    edge_index,
                    edge_bias,
                    attn_mask,
                    h.dtype,
                    h.device,
                )
                h = layer(h, attn_mask=attn_bias)
            if self.node_update_mlp and len(self.node_updates) > 0:
                h = self.node_updates[idx](h)
            
        # 2. Readout
        # Note: We extract only the scalar (L=0) features for energy
        # The optimized physics.py puts scalars first, so this slice is correct.
        scalars = h[:, :self.hidden_dim] 
        E = torch.sum(self.readout(scalars))

        aux = {}
        if self.use_aux_force_head and self.aux_force_head is not None:
            aux['force'] = self.aux_force_head(scalars)
        if self.use_aux_stress_head and self.aux_stress_head is not None:
            pooled = scalars.mean(dim=0, keepdim=True)
            stress_voigt = self.aux_stress_head(pooled).view(-1)
            aux['stress'] = stress_voigt

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
