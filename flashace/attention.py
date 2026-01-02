import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

_torch_scatter_available = importlib.util.find_spec("torch_scatter") is not None
if _torch_scatter_available:
    from torch_scatter import scatter_add, scatter_softmax

class DenseFlashAttention(nn.Module):
    def __init__(
        self,
        irreps_in,
        hidden_dim,
        num_heads: int = 4,
        message_clip: float | None = None,
        use_conditioned_decay: bool = True,
        share_qkv_mode: str | bool = "none",
        scalar_pre_norm: bool = True,
        layer_scale_init_value: float | None = 1e-2,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = o3.Irreps(irreps_in).dim
        self.hidden_dim = hidden_dim
        self.message_clip = message_clip
        self.use_conditioned_decay = use_conditioned_decay
        self.scalar_pre_norm = scalar_pre_norm
        self.layer_scale_init_value = layer_scale_init_value
        if isinstance(share_qkv_mode, bool):
            share_qkv_mode = "all" if share_qkv_mode else "none"
        if share_qkv_mode not in {"none", "kv", "all"}:
            raise ValueError("share_qkv_mode must be one of {'none', 'kv', 'all'} or a boolean")
        self.share_qkv_mode = share_qkv_mode

        if self.share_qkv_mode == "all":
            self.w_proj_shared = o3.Linear(irreps_in, irreps_in)
            self.radial_update_shared = o3.Linear(irreps_in, irreps_in)
            self.tangential_update_shared = o3.Linear(irreps_in, irreps_in)
        elif self.share_qkv_mode == "kv":
            self.w_proj = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )
            self.radial_update_shared = o3.Linear(irreps_in, irreps_in)
            self.tangential_update_shared = o3.Linear(irreps_in, irreps_in)
        else:
            self.w_proj = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )
            self.radial_update = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )
            self.tangential_update = nn.ModuleList(
                [o3.Linear(irreps_in, irreps_in) for _ in range(num_heads)]
            )

        # Geometry-aware scoring vectors
        self.radial_score = nn.Parameter(
            torch.empty(num_heads, self.feature_dim)
        )
        self.tangential_score = nn.Parameter(
            torch.empty(num_heads, self.feature_dim)
        )
        # Use a positive scale so longer bonds are consistently penalized.
        self._radial_distance_log_scale = nn.Parameter(torch.zeros(num_heads))
        # Distance-dependent temperature sharpens radial logits for close
        # neighbors while keeping gradients stable on far bonds.
        self._radial_temp_bias = nn.Parameter(torch.zeros(num_heads))
        self._radial_temp_weight = nn.Parameter(torch.zeros(num_heads))
        # Distance-gated mixer that blends radial/tangential streams so
        # short bonds can emphasize radial updates while long bonds lean on
        # tangential cues without running separate aggregation passes.
        self._mix_bias = nn.Parameter(torch.zeros(num_heads))
        self._mix_scale = nn.Parameter(torch.zeros(num_heads))

        # Environment-conditioned decay/temperature to sharpen locality per site.
        hidden_mid = max(1, self.feature_dim // 2)
        self.radial_decay_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_mid),
                    nn.SiLU(),
                    nn.Linear(hidden_mid, 1),
                )
                for _ in range(num_heads)
            ]
        )
        self.radial_temp_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_mid),
                    nn.SiLU(),
                    nn.Linear(hidden_mid, 1),
                )
                for _ in range(num_heads)
            ]
        )

        self.w_out = o3.Linear(irreps_in, irreps_in)
        self.scalar_norm = nn.LayerNorm(hidden_dim) if scalar_pre_norm else None
        self.layer_scale = (
            nn.Parameter(torch.full((self.feature_dim,), layer_scale_init_value))
            if layer_scale_init_value is not None
            else None
        )
        self.drop_path_rate = drop_path_rate
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.radial_score)
        nn.init.xavier_uniform_(self.tangential_score)
        nn.init.zeros_(self._radial_distance_log_scale)
        nn.init.zeros_(self._radial_temp_bias)
        nn.init.zeros_(self._radial_temp_weight)
        nn.init.zeros_(self._mix_bias)
        nn.init.zeros_(self._mix_scale)
        def _maybe_reset(layer):
            reset_fn = getattr(layer, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()

        if self.share_qkv_mode == "all":
            for layer in [
                self.w_proj_shared,
                self.radial_update_shared,
                self.tangential_update_shared,
            ]:
                _maybe_reset(layer)
        elif self.share_qkv_mode == "kv":
            for layer in list(self.w_proj) + [
                self.radial_update_shared,
                self.tangential_update_shared,
            ]:
                _maybe_reset(layer)
        else:
            for layer in list(self.w_proj) + list(self.radial_update) + list(self.tangential_update):
                _maybe_reset(layer)
        for mlp in list(self.radial_decay_mlp) + list(self.radial_temp_mlp):
            for m in mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        if self.layer_scale is not None and self.layer_scale_init_value is not None:
            nn.init.constant_(self.layer_scale, self.layer_scale_init_value)
    def forward(self, x, edge_index, edge_vec, edge_len, temperature_scale: float = 1.0):
        sender, receiver = edge_index
        num_nodes = x.shape[0]
        if sender.numel() == 0:
            return x

        if self.scalar_norm is not None:
            scalars, rest = x[..., : self.hidden_dim], x[..., self.hidden_dim :]
            x = torch.cat((self.scalar_norm(scalars), rest), dim=-1)

        if self.share_qkv_mode == "all":
            energy_base = self.w_proj_shared(x)
            radial_base = self.radial_update_shared(x)
            tangential_base = self.tangential_update_shared(x)
            energy_proj = energy_base.unsqueeze(0).expand(self.num_heads, -1, -1)
            radial_proj = radial_base.unsqueeze(0).expand(self.num_heads, -1, -1)
            tangential_proj = tangential_base.unsqueeze(0).expand(self.num_heads, -1, -1)
        elif self.share_qkv_mode == "kv":
            energy_proj = torch.stack([layer(x) for layer in self.w_proj], dim=0)
            radial_base = self.radial_update_shared(x)
            tangential_base = self.tangential_update_shared(x)
            radial_proj = radial_base.unsqueeze(0).expand(self.num_heads, -1, -1)
            tangential_proj = tangential_base.unsqueeze(0).expand(self.num_heads, -1, -1)
        else:
            energy_proj = torch.stack([layer(x) for layer in self.w_proj], dim=0)
            radial_proj = torch.stack([layer(x) for layer in self.radial_update], dim=0)
            tangential_proj = torch.stack([layer(x) for layer in self.tangential_update], dim=0)

        out = self._geometric_decomposition_attention(
            energy_proj,
            radial_proj,
            tangential_proj,
            sender,
            receiver,
            edge_len,
            temperature_scale=temperature_scale,
        )

        out = torch.nan_to_num(out)
        out = self.w_out(out)
        if self.layer_scale is not None:
            out = out * self.layer_scale
        if self.drop_path_rate > 0.0 and self.training:
            keep_prob = 1 - self.drop_path_rate
            shape = (out.shape[0],) + (1,) * (out.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=out.dtype, device=out.device)
            random_tensor = torch.floor(random_tensor)
            out = out / keep_prob * random_tensor
        return x + out

    def _geometric_decomposition_attention(
        self,
        energy_proj,
        radial_proj,
        tangential_proj,
        sender,
        receiver,
        edge_len,
        temperature_scale: float,
    ):
        # *_proj are shaped (num_heads, num_nodes, feature_dim)
        num_heads = energy_proj.shape[0]
        num_nodes = energy_proj.shape[1]

        energy_delta = energy_proj[:, sender] - energy_proj[:, receiver]
        radial_delta = radial_proj[:, sender] - radial_proj[:, receiver]
        tangential_delta = tangential_proj[:, sender] - tangential_proj[:, receiver]


        # Radial energy penalizes long bonds, tangential is distance agnostic.
        radial_distance_scale = F.softplus(self._radial_distance_log_scale).to(edge_len.dtype)[:, None]

        receiver_feat = energy_proj[:, receiver]  # (heads, edges, feature_dim)
        if self.use_conditioned_decay:
            decay_offset = torch.stack(
                [mlp(receiver_feat[h]).squeeze(-1) for h, mlp in enumerate(self.radial_decay_mlp)],
                dim=0,
            )
            temp_offset = torch.stack(
                [mlp(receiver_feat[h]).squeeze(-1) for h, mlp in enumerate(self.radial_temp_mlp)],
                dim=0,
            )
        else:
            decay_offset = torch.zeros_like(energy_delta[..., 0])
            temp_offset = torch.zeros_like(energy_delta[..., 0])

        radial_logits = (
            (energy_delta * self.radial_score[:, None, :]).sum(dim=-1).float()
            - (radial_distance_scale + decay_offset).float() * edge_len.float()
        )
        radial_temp = F.softplus(
            self._radial_temp_bias[:, None]
            + self._radial_temp_weight[:, None] * edge_len
            + temp_offset
        )
        radial_temp = (radial_temp * temperature_scale).float()
        radial_logits = radial_logits / (radial_temp + 1e-4)
        tangential_logits = (
            energy_delta * self.tangential_score[:, None, :]
        ).sum(dim=-1).float()

        num_edges = sender.numel()
        feature_dim = energy_proj.shape[-1]
        out = torch.zeros_like(energy_proj)
        if num_edges == 0:
            return out.mean(dim=0)

        expanded_receiver = receiver.unsqueeze(0).expand(num_heads, -1)

        def _segment_softmax(logits):
            # logits: (heads, num_edges)
            if _torch_scatter_available:
                return scatter_softmax(
                    logits, expanded_receiver, dim=1, dim_size=num_nodes
                )

            max_init = torch.full(
                (num_heads, num_nodes),
                float("-inf"),
                device=logits.device,
                dtype=logits.dtype,
            )
            max_per_node = max_init.scatter_reduce(
                1,
                expanded_receiver,
                logits,
                reduce="amax",
                include_self=True,
            )
            max_per_node = torch.where(
                torch.isfinite(max_per_node), max_per_node, torch.zeros_like(max_per_node)
            )

            centered = logits - max_per_node.gather(1, expanded_receiver)
            exp_logits = torch.exp(centered)

            denom = torch.zeros_like(max_per_node).scatter_add(
                1, expanded_receiver, exp_logits
            )
            alpha = exp_logits / (denom.gather(1, expanded_receiver) + 1e-9)
            return torch.nan_to_num(alpha)

        radial_alpha = torch.nan_to_num(_segment_softmax(radial_logits))
        tangential_alpha = torch.nan_to_num(_segment_softmax(tangential_logits))

        mix_gate = torch.sigmoid(
            self._mix_bias[:, None]
            + self._mix_scale[:, None] * edge_len[None, :]
        )

        blended_alpha = mix_gate * radial_alpha + (1.0 - mix_gate) * tangential_alpha
        blended_delta = mix_gate[:, :, None] * radial_delta + (
            1.0 - mix_gate[:, :, None]
        ) * tangential_delta

        weighted_delta = blended_alpha[..., None].to(blended_delta.dtype) * blended_delta
        if self.message_clip is not None:
            clip = torch.tensor(self.message_clip, device=weighted_delta.device, dtype=weighted_delta.dtype)
            norms = weighted_delta.norm(dim=-1, keepdim=True)
            safe = norms + 1e-8
            scale = torch.tanh(norms / clip) * (clip / safe)
            weighted_delta = weighted_delta * scale
        if _torch_scatter_available:
            out = scatter_add(
                weighted_delta,
                expanded_receiver[:, :, None].expand(-1, -1, feature_dim),
                dim=1,
                dim_size=num_nodes,
            )
        else:
            out = out.scatter_add(
                1,
                expanded_receiver[:, :, None].expand(-1, -1, feature_dim),
                weighted_delta,
            )
        return out.mean(dim=0)
