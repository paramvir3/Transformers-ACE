import argparse
import yaml
import os
from typing import Optional
import torch
import torch.optim as optim
import numpy as np
import time
from ase.io import read
from ase.data import atomic_numbers, chemical_symbols
from e3nn import o3
from flashace.model import FlashACE
from flashace.plotting import plot_training_metrics
from ase.neighborlist import neighbor_list
from torch.utils.data import DataLoader, Dataset, random_split

# --- STABILITY SETTINGS ---
# Disable TF32 to prevent potential TensorCore precision crashes in e3nn
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _load_config(config_arg: Optional[str]):
    """Load a YAML config, falling back to training/config.yaml if needed."""
    candidates = []
    if config_arg:
        candidates.append(config_arg)
    candidates.extend([
        "config.yaml",
        os.path.join("training", "config.yaml"),
    ])

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            with open(candidate, "r") as f:
                return yaml.safe_load(f), candidate

    raise FileNotFoundError(
        "No configuration file found. Provide --config or place config.yaml in the repo root or training/config.yaml."
    )


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, config, energy_shift_per_atom, atomic_energies):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'config': {
            'r_max': config['r_max'],
            'l_max': config['l_max'],
            'num_radial': config['num_radial'],
            'hidden_dim': config['hidden_dim'],
            'num_layers': config['num_layers'],
            'radial_basis_type': config.get('radial_basis_type', 'bessel'),
            'radial_trainable': config.get('radial_trainable', False),
            'envelope_exponent': config.get('envelope_exponent', 5),
            'gaussian_width': config.get('gaussian_width', 0.5),
            'energy_shift_per_atom': energy_shift_per_atom,
            'atomic_energies': atomic_energies or {},
            'amp_dtype': config.get('amp_dtype', 'float16'),
            'use_amp': config.get('use_amp', False),
            'grad_accum_steps': max(1, int(config.get('grad_accum_steps', 1))),
            'precompute_neighbors': config.get('precompute_neighbors', False),
            'descriptor_passes': config.get('descriptor_passes', 1),
            'descriptor_residual': config.get('descriptor_residual', True),
            'radial_mlp_hidden': config.get('radial_mlp_hidden', 64),
            'radial_mlp_layers': config.get('radial_mlp_layers', 2),
            'message_passing_layers': config.get('message_passing_layers', 0),
            'interleave_descriptor': config.get('interleave_descriptor', False),
            'edge_update_per_layer': config.get('edge_update_per_layer', False),
            'node_update_mlp': config.get('node_update_mlp', False),
            'transformer_num_heads': config.get('transformer_num_heads', 4),
            'transformer_ffn_hidden': config.get('transformer_ffn_hidden', None),
            'transformer_dropout': config.get('transformer_dropout', 0.0),
            'transformer_residual_dropout': config.get('transformer_residual_dropout', 0.0),
            'transformer_ffn_gated': config.get('transformer_ffn_gated', False),
            'transformer_layer_scale_init': config.get('transformer_layer_scale_init', None),
            'transformer_attention_chunk_size': config.get('transformer_attention_chunk_size', None),
            'use_transformer': config.get('use_transformer', True),
            'transformer_scalar_only': config.get('transformer_scalar_only', False),
            'attention_neighbor_mask': config.get('attention_neighbor_mask', False),
            'attention_short_range': config.get('attention_short_range', False),
            'attention_short_range_ratio': config.get('attention_short_range_ratio', 0.5),
            'attention_short_range_gate': config.get('attention_short_range_gate', True),
        }
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

class AtomisticDataset(Dataset):
    def __init__(
        self,
        atoms_list,
        r_max,
        random_rotation=False,
        precompute_neighbors=False,
        precompute_log_interval=0,
    ):
        self.atoms_list = atoms_list
        self.r_max = r_max
        self.random_rotation = random_rotation
        self.precompute_neighbors = precompute_neighbors
        self.precompute_log_interval = max(0, int(precompute_log_interval))

        self._edge_cache = None
        if precompute_neighbors:
            self._edge_cache = []
            total = len(atoms_list)
            if total:
                print(f"Precomputing neighbor lists for {total} structures...")
            for atoms in atoms_list:
                if (
                    self.precompute_log_interval > 0
                    and len(self._edge_cache) > 0
                    and len(self._edge_cache) % self.precompute_log_interval == 0
                ):
                    print(f"  precomputed {len(self._edge_cache)}/{total} neighbor lists")
                i, j = neighbor_list('ij', atoms, self.r_max)
                edge_index = torch.stack(
                    [torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)], dim=0
                )
                self._edge_cache.append(edge_index)
        
    def __len__(self): return len(self.atoms_list)
    
    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        
        # Geometry
        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        vol = torch.tensor(atoms.get_volume(), dtype=torch.float32)
        
        # Targets
        t_E = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        t_F = torch.tensor(atoms.get_forces(), dtype=torch.float32)
        
        # Stress (Robust Load)
        s_obj = None
        if atoms.calc is not None and 'stress' in atoms.calc.results:
            s_obj = atoms.calc.results['stress']
        elif 'stress' in atoms.info:
            s_obj = atoms.info['stress']
        elif 'virial' in atoms.info:
            s_obj = atoms.info['virial'] / atoms.get_volume()
            
        if s_obj is None:
            s_voigt = np.zeros((3,3))
        else:
            s_voigt = np.array(s_obj)

        if s_voigt.shape == (6,):
            s_mat = np.array([[s_voigt[0], s_voigt[5], s_voigt[4]],
                              [s_voigt[5], s_voigt[1], s_voigt[3]],
                              [s_voigt[4], s_voigt[3], s_voigt[2]]])
            t_S = torch.tensor(s_mat, dtype=torch.float32)
        else:
            t_S = torch.tensor(s_voigt, dtype=torch.float32)
            if len(t_S.shape) == 1: t_S = t_S.view(3,3)
        
        # Neighbors
        if self.random_rotation:
            # Random SO(3) rotation applied via Wigner matrices to encourage rotationally
            # equivariant learning even on small datasets. Energies remain invariant
            # while forces/stresses are rotated consistently.
            rot = o3.rand_matrix().to(dtype=pos.dtype)
            pos = pos @ rot.T
            t_F = t_F @ rot.T
            t_S = rot @ t_S @ rot.T

        if self._edge_cache is not None:
            edge_index = self._edge_cache[idx]
        else:
            i, j = neighbor_list('ij', atoms, self.r_max)
            edge_index = torch.stack(
                [torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)], dim=0
            )

        return {'z':z, 'pos':pos, 'edge_index':edge_index, 'volume':vol, 't_E':t_E, 't_F':t_F, 't_S':t_S}
    
    @staticmethod
    def collate_fn(batch): return batch

class MetricTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.sse_e = 0.0; self.sse_s = 0.0
        self.sum_force_mse = 0.0
        self.sum_force_mae = 0.0
        self.sum_force_sse = 0.0
        self.n_force_comp = 0
        self.n_atoms = 0; self.n_stress_comp = 0; self.n_struct = 0
    def update(self, p_E, p_F, p_S, t_E, t_F, t_S, n_ats):
        err_e = (p_E - t_E).item() / n_ats
        self.sse_e += err_e**2 * n_ats
        diff_f = p_F - t_F
        # Per-structure force MSE/MAE averaged over 3N components (NequIP-style).
        force_mse = diff_f.pow(2).mean().item()
        force_mae = diff_f.abs().mean().item()
        self.sum_force_mse += force_mse
        self.sum_force_mae += force_mae
        # Global per-atom force SSE for dataset-level RMSE.
        self.sum_force_sse += diff_f.pow(2).sum().item()
        self.n_force_comp += diff_f.numel()
        self.n_struct += 1
        if torch.norm(t_S) > 1e-6:
             self.sse_s += (p_S - t_S).pow(2).sum().item()
             self.n_stress_comp += 9
        self.n_atoms += n_ats
    def get_metrics(self):
        rmse_e = np.sqrt(self.sse_e / self.n_atoms) if self.n_atoms > 0 else 0.0
        rmse_s = np.sqrt(self.sse_s / self.n_stress_comp) if self.n_stress_comp > 0 else 0.0
        # Global per-atom force RMSE (preferred for tracking).
        force_mse = (self.sum_force_sse / self.n_force_comp) if self.n_force_comp > 0 else 0.0
        force_mae = (self.sum_force_mae / self.n_struct) if self.n_struct > 0 else 0.0
        force_rmse = np.sqrt(force_mse)
        return rmse_e * 1000, force_rmse, rmse_s, force_mse, force_mae

def compute_mean_energy_per_atom(atoms_seq):
    total_energy = 0.0
    total_atoms = 0
    for atoms in atoms_seq:
        total_energy += atoms.get_potential_energy()
        total_atoms += len(atoms)
    return (total_energy / total_atoms) if total_atoms > 0 else 0.0


def compute_atomic_energies_from_dataset(atoms_seq):
    """Solve for per-species reference energies via least squares.

    Builds the standard NequIP/MACE-style linear system where each structure's
    total energy is expressed as the sum of per-species reference energies plus
    a residual. The least-squares solution provides offsets that remove most of
    the composition-dependent baseline from the supervised loss.
    """

    species = sorted({int(z) for atoms in atoms_seq for z in atoms.numbers})
    if not species:
        return {}

    counts = []
    energies = []
    for atoms in atoms_seq:
        counts.append([np.count_nonzero(atoms.numbers == z) for z in species])
        energies.append(atoms.get_potential_energy())

    X = np.array(counts, dtype=float)
    y = np.array(energies, dtype=float)

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {z: float(e) for z, e in zip(species, coeffs)}


def parse_atomic_energy_table(raw_table):
    """Normalize an atomic energy mapping with atomic numbers as keys."""

    table = {}
    if raw_table is None:
        return table

    for key, value in raw_table.items():
        if isinstance(key, str):
            try:
                z = atomic_numbers[key]
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Unknown chemical symbol '{key}' in atomic_energies") from exc
        else:
            z = int(key)

        table[z] = float(value)

    return table


def atomic_energy_tensor(energy_table, device):
    if not energy_table:
        return None

    max_z = max(energy_table)
    tensor = torch.zeros(max_z + 1, dtype=torch.float32, device=device)
    for z, val in energy_table.items():
        tensor[z] = val
    return tensor

def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(description="Train Flash-ACE")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    args = parser.parse_args()

    config, config_path = _load_config(args.config)
    print(f"--- Loading {config_path} ---")
    print("--- Configuration ---")
    print(yaml.safe_dump(config, sort_keys=True, default_flow_style=False).strip())
    print("-" * 40)
    device = config['device']
    device_type = device.split(":")[0]

    use_amp = config.get('use_amp', False) and device_type == 'cuda'
    amp_dtype = torch.float16 if config.get('amp_dtype', 'float16') == 'float16' else torch.bfloat16
    grad_accum_steps = max(1, int(config.get('grad_accum_steps', 1)))

    if device_type == "cuda":
        sdp_backend = str(config.get('sdp_backend', 'auto')).lower()
        if sdp_backend not in {'auto', 'flash', 'mem_efficient', 'math'}:
            raise ValueError("sdp_backend must be one of {'auto', 'flash', 'mem_efficient', 'math'}")
        if sdp_backend == "auto":
            sdp_backend = "math"
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(sdp_backend == "flash")
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(sdp_backend == "mem_efficient")
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(sdp_backend == "math")
    

    print(f"Reading data from {config['train_file']}...")
    all_atoms = read(config['train_file'], index=":")

    if config['valid_file']:
        val_atoms = read(config['valid_file'], index=":")
        train_atoms = all_atoms
    else:
        val_len = max(1, int(len(all_atoms) * config.get('val_split', 0.1)))
        train_len = len(all_atoms) - val_len
        split_seed = int(config.get('split_seed', 42))
        strat_bins = int(config.get('stratified_val_bins', 0))
        if strat_bins > 0 and len(all_atoms) > strat_bins:
            energies = np.array([atoms.get_potential_energy() for atoms in all_atoms], dtype=float)
            edges = np.quantile(energies, np.linspace(0, 1, strat_bins + 1))
            indices = np.arange(len(all_atoms))
            val_indices = []
            rng = np.random.default_rng(split_seed)
            for i in range(strat_bins):
                in_bin = indices[(energies >= edges[i]) & (energies <= edges[i + 1])]
                if len(in_bin) == 0:
                    continue
                rng.shuffle(in_bin)
                take = max(1, int(len(in_bin) * config.get('val_split', 0.1)))
                val_indices.extend(in_bin[:take].tolist())
            val_indices = np.unique(val_indices)
            train_indices = np.setdiff1d(indices, val_indices)
            train_atoms = torch.utils.data.Subset(all_atoms, train_indices.tolist())
            val_atoms = torch.utils.data.Subset(all_atoms, val_indices.tolist())
            print(f"Stratified Split: {len(train_indices)} Training | {len(val_indices)} Validation")
        else:
            train_atoms, val_atoms = random_split(
                all_atoms, [train_len, val_len],
                generator=torch.Generator().manual_seed(split_seed)
            )
            print(f"Random Split: {train_len} Training | {val_len} Validation")

    atomic_energy_map = parse_atomic_energy_table(config.get('atomic_energies'))

    if not atomic_energy_map and config.get('solve_atomic_energies', False):
        atomic_energy_map = compute_atomic_energies_from_dataset(train_atoms)
        if atomic_energy_map:
            pretty = ", ".join(
                f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
            )
            print(f"Solved per-species reference energies from training set -> {pretty}")

    if atomic_energy_map:
        energy_shift_per_atom = None
        config['atomic_energies'] = atomic_energy_map
        pretty = ", ".join(
            f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
        )
        print(f"Using per-species reference energies for normalization: {pretty}")
    elif config.get('energy_shift_per_atom') is not None:
        energy_shift_per_atom = float(config['energy_shift_per_atom'])
        print(f"Using user-provided energy shift per atom: {energy_shift_per_atom:.6f} eV")
    else:
        energy_shift_per_atom = compute_mean_energy_per_atom(train_atoms)
        print(f"Computed mean energy per atom for normalization: {energy_shift_per_atom:.6f} eV")

    # DATALOADERS
    train_ds = AtomisticDataset(
        train_atoms,
        config['r_max'],
        random_rotation=config.get('random_rotation', False),
        precompute_neighbors=config.get('precompute_neighbors', False),
        precompute_log_interval=config.get('precompute_neighbors_log_interval', 0),
    )
    val_ds = AtomisticDataset(
        val_atoms,
        config['r_max'],
        random_rotation=False,
        precompute_neighbors=config.get('precompute_neighbors', False),
        precompute_log_interval=config.get('precompute_neighbors_log_interval', 0),
    )

    # Reduced workers to prevent CPU overhead issues
    num_workers = int(config.get('dataloader_num_workers', 2))
    pin_memory = bool(config.get('dataloader_pin_memory', device_type == "cuda"))
    if num_workers == 0:
        pin_memory = False
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        collate_fn=AtomisticDataset.collate_fn,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    valid_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        collate_fn=AtomisticDataset.collate_fn,
        num_workers=num_workers,
    )

    print("--- Initializing FlashACE ---")
    model = FlashACE(
        r_max=config['r_max'], l_max=config['l_max'], num_radial=config['num_radial'],
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
        radial_basis_type=config.get('radial_basis_type', 'bessel'),
        radial_trainable=config.get('radial_trainable', False),
        envelope_exponent=config.get('envelope_exponent', 5),
        gaussian_width=config.get('gaussian_width', 0.5),
        ipa_num_heads=config.get('ipa_num_heads', 4),
        ipa_num_points=config.get('ipa_num_points', 4),
        ipa_point_weight_init=config.get('ipa_point_weight_init', 1.0),
        ipa_point_scale=config.get('ipa_point_scale', 1.0),
        ipa_point_dropout=config.get('ipa_point_dropout', 0.0),
        ipa_bias_norm=config.get('ipa_bias_norm', True),
        ipa_logit_clip=config.get('ipa_logit_clip', None),
        ipa_use_flash=config.get('ipa_use_flash', False),
        ipa_ffn_hidden=config.get('ipa_ffn_hidden', None),
        ipa_dropout=config.get('ipa_dropout', 0.0),
        ipa_residual_dropout=config.get('ipa_residual_dropout', 0.0),
        ipa_ffn_gated=config.get('ipa_ffn_gated', False),
        ipa_layer_scale_init=config.get('ipa_layer_scale_init', None),
        descriptor_passes=config.get('descriptor_passes', 1),
        descriptor_residual=config.get('descriptor_residual', True),
        radial_mlp_hidden=config.get('radial_mlp_hidden', 64),
        radial_mlp_layers=config.get('radial_mlp_layers', 2),
        readout_hidden_dims=config.get('readout_hidden_dims', None),
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config['learning_rate'], amsgrad=True,
        weight_decay=config.get('weight_decay', 0.0)
    )
    warmup_epochs = max(0, int(config.get('lr_warmup_epochs', 0)))
    warmup_start = float(config.get('lr_warmup_start_factor', 0.1))
    scheduler_interval = str(config.get('lr_scheduler_interval', 'epoch')).lower()
    if warmup_start <= 0.0:
        raise ValueError("lr_warmup_start_factor must be > 0.0")
    if scheduler_interval not in {'epoch', 'step'}:
        raise ValueError("lr_scheduler_interval must be 'epoch' or 'step'")

    steps_per_epoch = max(1, len(train_loader))
    configured_t_max = int(config.get('lr_scheduler_t_max', config['epochs']))
    if scheduler_interval == 'step':
        warmup_iters = warmup_epochs * steps_per_epoch
        total_iters = max(warmup_iters + 1, configured_t_max * steps_per_epoch)
    else:
        warmup_iters = warmup_epochs
        total_iters = max(warmup_epochs + 1, configured_t_max)
    cosine_t_max = max(1, total_iters - warmup_iters)

    use_restarts = bool(config.get('lr_scheduler_use_restarts', False))
    restart_mult = float(config.get('lr_restart_mult', 1.0))
    if use_restarts:
        cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_t_max,
            T_mult=max(1.0, restart_mult),
            eta_min=config.get('lr_min', 0.0),
        )
    else:
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=config.get('lr_min', 0.0),
        )
    if warmup_iters > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start,
            total_iters=warmup_iters,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_iters],
        )
    else:
        scheduler = cosine

    # Optional force-weight annealing to ease optimization toward energies.
    force_w_start = float(config.get('forces_weight', 10.0))
    force_w_final = float(config.get('forces_weight_final', force_w_start))
    force_w_decay_epochs = int(config.get('forces_weight_decay_epochs', 0))

    def _force_weight(epoch_idx: int) -> float:
        if force_w_decay_epochs <= 0 or force_w_start == force_w_final:
            return force_w_start
        frac = min(1.0, epoch_idx / max(1, force_w_decay_epochs))
        return force_w_start + frac * (force_w_final - force_w_start)

    resume_path = config.get('resume_from')
    start_epoch = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if not resume_path and config.get('resume_latest', False):
        ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
        latest_path = os.path.join(ckpt_dir, 'latest.pt')
        if os.path.isfile(latest_path):
            resume_path = latest_path

    if resume_path:
        print(f"--- Loading checkpoint from {resume_path} ---")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if config.get('resume_load_optimizer', False) and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if config.get('resume_load_scheduler', False) and checkpoint.get('scheduler_state_dict'):
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if config.get('resume_load_scaler', False) and checkpoint.get('scaler_state_dict') and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if config.get('use_checkpoint_energy_shift', True):
            ckpt_atomic = checkpoint.get('config', {}).get('atomic_energies') or {}
            if ckpt_atomic:
                atomic_energy_map = {int(k): float(v) for k, v in ckpt_atomic.items()}
                energy_shift_per_atom = None
                config['atomic_energies'] = atomic_energy_map
                pretty = ", ".join(
                    f"{chemical_symbols[z]}: {e:.6f} eV" for z, e in sorted(atomic_energy_map.items())
                )
                print(f"Using checkpoint atomic energy references: {pretty}")
            else:
                ckpt_shift = checkpoint.get('config', {}).get('energy_shift_per_atom')
                if ckpt_shift is not None:
                    energy_shift_per_atom = ckpt_shift
                    print(f"Using checkpoint energy shift per atom: {energy_shift_per_atom:.6f} eV")

        start_epoch = int(checkpoint.get('epoch', 0))
        print(f"Resuming training from epoch {start_epoch}")

    energy_shift = None
    if energy_shift_per_atom is not None:
        energy_shift = torch.tensor(energy_shift_per_atom, dtype=torch.float32, device=device)

    atomic_energy_vec = atomic_energy_tensor(atomic_energy_map, device)

    def baseline_energy(z_tensor):
        if atomic_energy_vec is not None:
            if torch.max(z_tensor).item() >= atomic_energy_vec.shape[0]:
                raise ValueError("Encountered atomic number without reference energy")
            return torch.sum(atomic_energy_vec[z_tensor])
        elif energy_shift is not None:
            return energy_shift * len(z_tensor)
        else:
            return torch.tensor(0.0, device=device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_energy_mev': [],
        'val_energy_mev': [],
        'train_force_rmse': [],
        'val_force_rmse': [],
        'train_stress_rmse': [],
        'val_stress_rmse': [],
    }
    metrics_interval = max(1, int(config.get('metrics_interval', 1)))

    ckpt_interval = int(config.get('checkpoint_interval', 0) or 0)
    ckpt_latest = bool(config.get('checkpoint_latest', False))
    ckpt_dir = config.get('checkpoint_dir', 'checkpoints')

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Temperature curriculum for attention sharpness.
    temp_scale_start = float(config.get('temperature_scale_start', 1.0))
    temp_scale_end = float(config.get('temperature_scale_end', temp_scale_start))
    temp_scale_epochs = int(config.get('temperature_scale_epochs', 0) or 0)

    def _temperature_scale(epoch_idx: int, force_ema: float | None = None):
        base = temp_scale_start
        if temp_scale_epochs > 0:
            frac = min(1.0, epoch_idx / float(temp_scale_epochs))
            base = temp_scale_start + frac * (temp_scale_end - temp_scale_start)
        ref = float(config.get('temperature_force_ref', 0.0))
        gamma = float(config.get('temperature_force_exponent', 0.0))
        if force_ema is not None and ref > 0.0 and gamma != 0.0:
            scale = (force_ema / ref) ** gamma
            base = base * torch.tensor(scale, device=device, dtype=amp_dtype if use_amp else torch.float32).item()
        return base

    force_consistency_weight = float(config.get('force_consistency_weight', 0.0))
    displacement_prob = float(config.get('displacement_prob', 0.0))
    displacement_sigma = float(config.get('displacement_sigma', 0.0))
    aux_force_weight = float(config.get('aux_force_weight', 0.0))
    aux_stress_weight = float(config.get('aux_stress_weight', 0.0))
    sobolev_weight = float(config.get('sobolev_weight', 0.0))
    sobolev_sigma = float(config.get('sobolev_sigma', 0.0))
    force_only_epochs = int(config.get('force_only_epochs', 0))
    
    print(
        f"{'Epoch':>5} | {'Loss':>10} | {'E (meV)':>10} | {'force_RMSE':>12} | {'force_MSE':>12} | {'force_MAE':>12} | {'S_RMSE':>10} || "
        f"{'Val Loss':>10} | {'Val E':>10} | {'Val force_RMSE':>16} | {'Val force_MSE':>16}",
        flush=True,
    )
    print("-" * 170, flush=True)
    
    force_loss_ema = None
    for epoch in range(start_epoch, config['epochs']):
        compute_metrics = ((epoch + 1) % metrics_interval == 0) or (epoch + 1 == config['epochs'])
        force_weight = _force_weight(epoch)
        if force_only_epochs > 0 and epoch < force_only_epochs:
            energy_weight = 0.0
            stress_weight = 0.0
        else:
            energy_weight = config['energy_weight']
            stress_weight = config['stress_weight']
        model.train()
        train_metrics = MetricTracker() if compute_metrics else None
        total_loss = 0.0
        total_items_seen = 0

        batch_idx = -1
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            batch_loss = 0.0

            # --- GRADIENT ACCUMULATION (FP32/AMP) ---
            # Optional small-displacement augmentation
            items = list(batch)
            if displacement_prob > 0.0 and displacement_sigma > 0.0:
                augmented = []
                for item in items:
                    if torch.rand(1).item() < displacement_prob:
                        perturbed = {
                            k: (v.clone() if isinstance(v, torch.Tensor) else v)
                            for k, v in item.items()
                        }
                        noise = torch.randn_like(perturbed['pos']) * displacement_sigma
                        perturbed['pos'] = perturbed['pos'] + noise
                        augmented.append(perturbed)
                items.extend(augmented)

            for item in items:
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        item[k] = v.to(device, non_blocking=True)
                if force_consistency_weight > 0.0:
                    item['pos'] = item['pos'].clone().detach().requires_grad_(True)

                # Standard Forward with optional AMP
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    temp_scale = _temperature_scale(epoch, force_loss_ema)
                    p_E, p_F, p_S, aux = model(
                        item,
                        training=True,
                        temperature_scale=temp_scale,
                        detach_pos=force_consistency_weight <= 0.0,
                    )
                    n_ats = len(item['z'])

                    target_E = item['t_E'] - baseline_energy(item['z'])
                    loss_e = ((p_E - target_E) / n_ats)**2
                    loss_f = torch.mean((p_F - item['t_F'])**2)
                    loss_s = torch.tensor(0.0, device=device)
                    if torch.norm(item['t_S']) > 1e-6:
                        loss_s = torch.mean((p_S - item['t_S'])**2)

                    loss_item = (energy_weight*loss_e) + \
                                (force_weight*loss_f) + \
                                (stress_weight*loss_s)

                    if aux_force_weight > 0.0 and 'force' in aux:
                        loss_item = loss_item + aux_force_weight * torch.mean((aux['force'] - item['t_F'])**2)
                    if aux_stress_weight > 0.0 and 'stress' in aux:
                        target_stress = item['t_S']
                        loss_item = loss_item + aux_stress_weight * torch.mean(
                            (aux['stress'] - torch.stack([
                                target_stress[0,0], target_stress[1,1], target_stress[2,2],
                                target_stress[0,1], target_stress[0,2], target_stress[1,2]
                            ]))**2
                        )

                    if sobolev_weight > 0.0 and sobolev_sigma > 0.0:
                        delta = torch.randn_like(item['pos']) * sobolev_sigma
                        pos_pert = item['pos'] + delta
                        perturbed = {**item, 'pos': pos_pert}
                        p_E_pert, _, _, _ = model(
                            perturbed,
                            training=True,
                            temperature_scale=temp_scale,
                            detach_pos=True,
                        )
                        fd = (p_E_pert - p_E) + (p_F.detach() * delta).sum()
                        loss_item = loss_item + sobolev_weight * fd.pow(2)

                    if force_consistency_weight > 0.0:
                        energy_grad = torch.autograd.grad(
                            p_E,
                            item['pos'],
                            create_graph=True,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if energy_grad is not None:
                            consistency = (p_F + energy_grad).pow(2).mean()
                            loss_item = loss_item + force_consistency_weight * consistency

                # Normalize and Backward
                loss_batch = loss_item / (len(items) * grad_accum_steps)
                scaler.scale(loss_batch).backward()

                batch_loss += loss_item.item()

                with torch.no_grad():
                    pred_E_abs = p_E + baseline_energy(item['z'])
                    if compute_metrics and train_metrics is not None:
                        train_metrics.update(pred_E_abs, p_F, p_S, item['t_E'], item['t_F'], item['t_S'], n_ats)
                    if force_loss_ema is None:
                        force_loss_ema = loss_f.detach()
                    else:
                        force_loss_ema = 0.9 * force_loss_ema + 0.1 * loss_f.detach()
                total_items_seen += 1

            if (batch_idx + 1) % grad_accum_steps == 0:
                if config.get('clip_grad_norm', None):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['clip_grad_norm']
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler_interval == 'step':
                    scheduler.step()
            total_loss += batch_loss

        # Flush any residual gradients if the last batch didn't trigger a step
        if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
            if config.get('clip_grad_norm', None):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['clip_grad_norm']
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler_interval == 'step':
                scheduler.step()

        avg_train_loss = total_loss / max(1, total_items_seen)
        if compute_metrics and train_metrics is not None:
            tr_e, tr_f, tr_s, tr_f_mse, tr_f_mae = train_metrics.get_metrics()
            history['train_loss'].append(avg_train_loss)
            history['train_energy_mev'].append(tr_e)
            history['train_force_rmse'].append(tr_f)
            history['train_stress_rmse'].append(tr_s)
        else:
            tr_e = tr_f = tr_s = tr_f_mse = tr_f_mae = float("nan")
            history['train_loss'].append(float("nan"))
            history['train_energy_mev'].append(float("nan"))
            history['train_force_rmse'].append(float("nan"))
            history['train_stress_rmse'].append(float("nan"))

        # Validation
        if compute_metrics:
            model.eval()
            val_metrics = MetricTracker()
            val_loss_accum = 0.0

            for batch in valid_loader:
                for item in batch:
                    for k, v in item.items():
                        if isinstance(v, torch.Tensor):
                            item[k] = v.to(device, non_blocking=True)

                    # Keep grad tracking on so autograd can form forces/stresses; we
                    # still avoid higher-order graphs with ``create_graph=False``
                    # inside the model during validation.
                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        stress_target = torch.norm(item['t_S']) > 1e-6
                        p_E, p_F, p_S, _ = model(
                            item,
                            training=False,
                            compute_stress=stress_target,
                        )
                        n_ats = len(item['z'])
                        target_E = item['t_E'] - baseline_energy(item['z'])
                        loss_e = ((p_E - target_E) / n_ats)**2
                        loss_f = torch.mean((p_F - item['t_F'])**2)
                        loss_s = torch.tensor(0.0, device=device)
                        if stress_target:
                            loss_s = torch.mean((p_S - item['t_S'])**2)
                        val_loss_accum += (config['energy_weight']*loss_e) + (force_weight*loss_f) + (config['stress_weight']*loss_s)

                    pred_E_abs = p_E + baseline_energy(item['z'])
                    val_metrics.update(pred_E_abs, p_F, p_S, item['t_E'], item['t_F'], item['t_S'], n_ats)

            avg_val_loss = val_loss_accum / len(val_atoms)
            val_e, val_f, val_s, val_f_mse, val_f_mae = val_metrics.get_metrics()
            history['val_loss'].append(avg_val_loss)
            history['val_energy_mev'].append(val_e)
            history['val_force_rmse'].append(val_f)
            history['val_stress_rmse'].append(val_s)
        else:
            avg_val_loss = val_e = val_f = val_s = val_f_mse = val_f_mae = float("nan")
            history['val_loss'].append(float("nan"))
            history['val_energy_mev'].append(float("nan"))
            history['val_force_rmse'].append(float("nan"))
            history['val_stress_rmse'].append(float("nan"))
        if scheduler_interval == 'epoch':
            scheduler.step()

        if compute_metrics:
            print(
                f"{epoch+1:5d} | "
                f"{avg_train_loss:10.4f} | {tr_e:10.2f} | {tr_f:12.6f} | {tr_f_mse:12.6f} | {tr_f_mae:12.6f} | {tr_s:10.4f} || "
                f"{avg_val_loss:10.4f} | {val_e:10.2f} | {val_f:16.6f} | {val_f_mse:16.6f}",
                flush=True,
            )

        if ckpt_interval > 0 and (epoch + 1) % ckpt_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                ckpt_path,
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                energy_shift_per_atom,
                atomic_energy_map,
            )
        if ckpt_latest:
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            save_checkpoint(
                latest_path,
                epoch + 1,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                energy_shift_per_atom,
                atomic_energy_map,
            )

    save_checkpoint(
        config['model_save_path'],
        config['epochs'],
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        energy_shift_per_atom,
        atomic_energy_map,
    )
    print(f"Training Finished. Saved to {config['model_save_path']}")
    plot_dir = config.get('plot_dir', 'plots')
    plot_training_metrics(history, save_dir=plot_dir)

if __name__ == "__main__": main()
