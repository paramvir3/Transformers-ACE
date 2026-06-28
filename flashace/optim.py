"""Optimizers used by the Transformers-ACE training script."""

from __future__ import annotations

import math
from fnmatch import fnmatch
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch


def zeropower_via_newtonschulz5(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Return the Newton-Schulz zero-power approximation used by Muon.

    This mirrors the NequIX/KellerJordan quintic Newton-Schulz iteration and
    orthogonalizes along the last two dimensions.
    """

    if matrix.ndim < 2:
        raise ValueError("Muon zero-power update requires a matrix-shaped tensor")

    update = matrix
    transposed = update.size(-2) > update.size(-1)
    if transposed:
        update = update.mT

    update = update / (update.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(int(steps)):
        gram = update @ update.mT
        update = a * update + (b * gram + c * gram @ gram) @ update

    if transposed:
        update = update.mT
    return update


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute one Muon update for a matrix-like parameter gradient."""

    momentum.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update = update * math.sqrt(max(1.0, grad.size(-2) / grad.size(-1)))
    return update.reshape_as(grad)


def adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: Tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Return the auxiliary Adam update used by NequIX Muon."""

    exp_avg.lerp_(grad, 1.0 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1.0 - betas[1])
    exp_avg_corrected = exp_avg / (1.0 - betas[0] ** step)
    exp_avg_sq_corrected = exp_avg_sq / (1.0 - betas[1] ** step)
    return exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + eps)


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Single-device Muon optimizer with AdamW for non-matrix parameters.

    Muon is applied only to parameter groups marked ``use_muon=True``. Other
    groups use AdamW, which is appropriate for biases, 1D scale parameters, and
    normalization parameters.
    """

    def __init__(self, param_groups: Iterable[MutableMapping]):
        groups = list(param_groups)
        if not groups:
            raise ValueError("MuonWithAuxAdamW requires at least one parameter group")

        for group in groups:
            group.setdefault("use_muon", False)
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
            else:
                group.setdefault("lr", 3.0e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1.0e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group: Dict) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta = group["momentum"]
        ns_steps = group["ns_steps"]
        nesterov = group["nesterov"]

        for param in group["params"]:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if grad.ndim < 2:
                raise RuntimeError("Muon parameter groups must contain matrix-like tensors")

            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)

            update = muon_update(
                grad,
                state["momentum_buffer"],
                beta=beta,
                ns_steps=ns_steps,
                nesterov=nesterov,
            )
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(update, alpha=-lr)

    def _step_adamw_group(self, group: Dict) -> None:
        lr = group["lr"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for param in group["params"]:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            grad = param.grad
            if grad.is_sparse:
                raise RuntimeError("AdamW auxiliary groups do not support sparse gradients")

            state = self.state[param]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1

            update = adam_update(
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
                group["betas"],
                eps,
            )
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            param.add_(update, alpha=-lr)


SingleDeviceMuonWithAuxAdam = MuonWithAuxAdamW


def _is_no_decay_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.endswith(".bias")
        or "norm" in lowered
        or "layer_scale" in lowered
        or lowered.endswith("distance_log_scale")
    )


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def _is_hidden_muon_matrix(name: str, param: torch.nn.Parameter) -> bool:
    """Return whether a parameter is a hidden matrix suitable for Muon.

    This follows the original Muon recommendation: use Muon for hidden weight
    matrices, while embeddings, output heads, gains, and biases remain on AdamW.
    For Transformers-ACE, the safest default hidden matrices are the local
    attention query/key projections and scalar feed-forward layers.
    """

    if param.ndim < 2 or not name.endswith(".weight"):
        return False

    hidden_patterns = (
        "layers.*.q_proj.weight",
        "layers.*.k_proj.weight",
        "layers.*.scalar_ffn.*.weight",
    )
    excluded_patterns = (
        "emb.weight",
        "readout.*.weight",
        "ace.radial_net.*.weight",
        "ace.center_proj.weight",
        "layers.*.radial_bias.*.weight",
    )
    return _matches_any(name, hidden_patterns) and not _matches_any(name, excluded_patterns)


def get_muon_param_groups(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    muon_learning_rate: Optional[float] = None,
    aux_learning_rate: Optional[float] = None,
    momentum: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    aux_betas: Tuple[float, float] = (0.9, 0.95),
    aux_eps: float = 1.0e-10,
    parameter_mode: str = "hidden",
    include_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """Build Muon/AdamW parameter groups for a generic PyTorch model."""

    include_patterns = tuple(include_patterns or ())
    exclude_patterns = tuple(exclude_patterns or ())
    parameter_mode = str(parameter_mode).lower()
    if parameter_mode not in {"hidden", "all_matrices"}:
        raise ValueError("muon_parameter_mode must be 'hidden' or 'all_matrices'")

    muon_params = []
    aux_decay_params = []
    aux_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_muon = (
            _is_hidden_muon_matrix(name, param)
            if parameter_mode == "hidden"
            else param.ndim >= 2 and not _is_no_decay_name(name)
        )
        if include_patterns and _matches_any(name, include_patterns):
            use_muon = param.ndim >= 2
        if exclude_patterns and _matches_any(name, exclude_patterns):
            use_muon = False

        if use_muon:
            muon_params.append(param)
        elif _is_no_decay_name(name):
            aux_no_decay_params.append(param)
        else:
            aux_decay_params.append(param)

    assigned = {id(param) for param in muon_params + aux_decay_params + aux_no_decay_params}
    expected = {id(param) for param in model.parameters() if param.requires_grad}
    if assigned != expected:
        raise RuntimeError("Muon parameter grouping did not cover every trainable parameter")

    muon_lr = learning_rate if muon_learning_rate is None else muon_learning_rate
    aux_lr = learning_rate if aux_learning_rate is None else aux_learning_rate

    groups = []
    if muon_params:
        muon_params = sorted(muon_params, key=lambda p: p.numel(), reverse=True)
        groups.append(
            {
                "name": "muon_matrix",
                "params": muon_params,
                "use_muon": True,
                "lr": muon_lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "ns_steps": int(ns_steps),
                "nesterov": bool(nesterov),
            }
        )
    if aux_decay_params:
        groups.append(
            {
                "name": "adamw_aux_decay",
                "params": aux_decay_params,
                "use_muon": False,
                "lr": aux_lr,
                "betas": aux_betas,
                "eps": aux_eps,
                "weight_decay": weight_decay,
            }
        )
    if aux_no_decay_params:
        groups.append(
            {
                "name": "adamw_aux_no_decay",
                "params": aux_no_decay_params,
                "use_muon": False,
                "lr": aux_lr,
                "betas": aux_betas,
                "eps": aux_eps,
                "weight_decay": 0.0,
            }
        )
    return groups


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for group in optimizer.param_groups:
        n_tensors = len(group["params"])
        n_values = sum(param.numel() for param in group["params"])
        kind = "Muon" if group.get("use_muon", False) else "AdamW"
        name = group.get("name", kind.lower())
        parts.append(f"{name}: {kind}, tensors={n_tensors}, params={n_values}, lr={group['lr']}")
    return "; ".join(parts)


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


def _config_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create the configured optimizer while preserving Adam as the default."""

    optimizer_name = str(config.get("optimizer", "adam")).lower()
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.0))

    if optimizer_name == "muon":
        betas = tuple(float(v) for v in config.get("muon_aux_betas", (0.9, 0.95)))
        if len(betas) != 2:
            raise ValueError("muon_aux_betas must contain two values")
        param_groups = get_muon_param_groups(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            muon_learning_rate=_optional_float(config.get("muon_learning_rate")),
            aux_learning_rate=_optional_float(config.get("muon_aux_learning_rate")),
            momentum=float(config.get("muon_momentum", 0.95)),
            ns_steps=int(config.get("muon_ns_steps", 5)),
            nesterov=_config_bool(config.get("muon_nesterov", True)),
            aux_betas=(betas[0], betas[1]),
            aux_eps=float(config.get("muon_aux_eps", 1.0e-10)),
            parameter_mode=str(config.get("muon_parameter_mode", "hidden")),
            include_patterns=config.get("muon_include", ()),
            exclude_patterns=config.get("muon_exclude", ()),
        )
        return MuonWithAuxAdamW(param_groups)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=_config_bool(config.get("amsgrad", False)),
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            amsgrad=_config_bool(config.get("amsgrad", True)),
            weight_decay=weight_decay,
        )

    raise ValueError("optimizer must be one of: adam, adamw, muon")
