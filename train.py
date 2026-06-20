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
from flashace.model import TransformersACE
from flashace.plotting import plot_metric_history
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
            'attention_num_heads': config.get('attention_num_heads', config.get('transformer_num_heads', 4)),
            'attention_key_dim': config.get('attention_key_dim', None),
            'attention_ffn_hidden': config.get('attention_ffn_hidden', config.get('transformer_ffn_hidden', None)),
            'attention_dropout': config.get('attention_dropout', config.get('transformer_dropout', 0.0)),
            'attention_layer_scale_init': config.get('attention_layer_scale_init', 1e-2),
            'attention_distance_penalty': config.get('attention_distance_penalty', True),
        }
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def build_neighbor_tensors(atoms, r_max):
    """Return neighbor-to-center edges plus periodic shifts from ASE."""
    i, j, shifts = neighbor_list('ijS', atoms, r_max)
    edge_index = torch.stack(
        [torch.tensor(j, dtype=torch.long), torch.tensor(i, dtype=torch.long)],
        dim=0,
    )
    edge_shift = torch.tensor(shifts, dtype=torch.float32)
    return edge_index, edge_shift


class AtomisticDataset(Dataset):
    def __init__(self, atoms_list, r_max, random_rotation=False, precompute_neighbors=False):
        self.atoms_list = atoms_list
        self.r_max = r_max
        self.random_rotation = random_rotation
        self.precompute_neighbors = precompute_neighbors

        self._edge_cache = None
        if precompute_neighbors:
            self._edge_cache = []
            for atoms in atoms_list:
                self._edge_cache.append(build_neighbor_tensors(atoms, self.r_max))
        
    def __len__(self): return len(self.atoms_list)
    
    def __getitem__(self, idx):
        atoms = self.atoms_list[idx]
        
        # Geometry
        z = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
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
            cell = cell @ rot.T
            t_F = t_F @ rot.T
            t_S = rot @ t_S @ rot.T

        if self._edge_cache is not None:
            edge_index, edge_shift = self._edge_cache[idx]
        else:
            edge_index, edge_shift = build_neighbor_tensors(atoms, self.r_max)

        return {
            'z': z,
            'pos': pos,
            'cell': cell,
            'edge_index': edge_index,
            'edge_shift': edge_shift,
            'volume': vol,
            't_E': t_E,
            't_F': t_F,
            't_S': t_S,
        }
    
    @staticmethod
    def collate_fn(batch): return batch

class MetricTracker:
    def __init__(self): self.reset()
    def reset(self):
        self.sse_e = 0.0; self.sse_s = 0.0
        self.sum_force_mse = 0.0
        self.sum_force_mae = 0.0
        self.n_atoms = 0; self.n_stress_comp = 0; self.n_struct = 0
    def update(self, p_E, p_F, p_S, t_E, t_F, t_S, n_ats):
        err_e = (p_E - t_E).item() / n_ats
        self.sse_e += err_e**2 * n_ats
        diff_f = p_F - t_F
        # Per-structure force MSE/MAE averaged over 3N components.
        force_mse = diff_f.pow(2).mean().item()
        force_mae = diff_f.abs().mean().item()
        self.sum_force_mse += force_mse
        self.sum_force_mae += force_mae
        self.n_struct += 1
        if torch.norm(t_S) > 1e-6:
             self.sse_s += (p_S - t_S).pow(2).sum().item()
             self.n_stress_comp += 9
        self.n_atoms += n_ats
    def get_metrics(self):
        rmse_e = np.sqrt(self.sse_e / self.n_atoms) if self.n_atoms > 0 else 0.0
        rmse_s = np.sqrt(self.sse_s / self.n_stress_comp) if self.n_stress_comp > 0 else 0.0
        force_mse = (self.sum_force_mse / self.n_struct) if self.n_struct > 0 else 0.0
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

    Builds the standard per-species linear system where each structure's
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
    parser = argparse.ArgumentParser(description="Train Transformers-ACE")
    parser.add_argument("--config", "-c", default=None, help="Path to YAML config file")
    args = parser.parse_args()

    config, config_path = _load_config(args.config)
    print(f"--- Loading {config_path} ---")

    torch_num_threads = int(config.get('torch_num_threads', 0) or 0)
    if torch_num_threads > 0:
        torch.set_num_threads(torch_num_threads)
    torch_num_interop_threads = int(config.get('torch_num_interop_threads', 0) or 0)
    if torch_num_interop_threads > 0:
        try:
            torch.set_num_interop_threads(torch_num_interop_threads)
        except RuntimeError:
            # PyTorch only permits changing this before inter-op work starts.
            pass
    print(
        f"PyTorch CPU threads: {torch.get_num_threads()} intra-op, "
        f"{torch.get_num_interop_threads()} inter-op"
    )

    device = config['device']
    device_type = device.split(":")[0]

    use_amp = config.get('use_amp', False) and device_type == 'cuda'
    amp_dtype = torch.float16 if config.get('amp_dtype', 'float16') == 'float16' else torch.bfloat16
    grad_accum_steps = max(1, int(config.get('grad_accum_steps', 1)))

    if device_type == "cuda":
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    

    print(f"Reading data from {config['train_file']}...")
    all_atoms = read(config['train_file'], index=":")

    if config['valid_file']:
        val_atoms = read(config['valid_file'], index=":")
        train_atoms = all_atoms
    else:
        val_len = max(1, int(len(all_atoms) * config.get('val_split', 0.1)))
        train_len = len(all_atoms) - val_len
        train_atoms, val_atoms = random_split(
            all_atoms, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
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
    )
    val_ds = AtomisticDataset(
        val_atoms,
        config['r_max'],
        random_rotation=False,
        precompute_neighbors=config.get('precompute_neighbors', False),
    )

    num_workers = int(config.get('num_workers', 2))
    pin_memory = device_type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    
    valid_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                              collate_fn=AtomisticDataset.collate_fn, num_workers=num_workers)

    print("--- Initializing Transformers-ACE ---")
    model = TransformersACE(
        r_max=config['r_max'], l_max=config['l_max'], num_radial=config['num_radial'],
        hidden_dim=config['hidden_dim'], num_layers=config['num_layers'],
        radial_basis_type=config.get('radial_basis_type', 'bessel'),
        radial_trainable=config.get('radial_trainable', False),
        envelope_exponent=config.get('envelope_exponent', 5),
        gaussian_width=config.get('gaussian_width', 0.5),
        attention_num_heads=config.get('attention_num_heads', config.get('transformer_num_heads', 4)),
        attention_key_dim=config.get('attention_key_dim', None),
        attention_ffn_hidden=config.get('attention_ffn_hidden', config.get('transformer_ffn_hidden', None)),
        attention_dropout=config.get('attention_dropout', config.get('transformer_dropout', 0.0)),
        attention_layer_scale_init=config.get('attention_layer_scale_init', 1e-2),
        attention_distance_penalty=config.get('attention_distance_penalty', True),
        descriptor_passes=config.get('descriptor_passes', 1),
        descriptor_residual=config.get('descriptor_residual', True),
        radial_mlp_hidden=config.get('radial_mlp_hidden', 64),
        radial_mlp_layers=config.get('radial_mlp_layers', 2),
        interleave_descriptor=config.get('interleave_descriptor', False),
        use_aux_force_head=False,
        use_aux_stress_head=False,
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
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_energy_rmse': [],
        'val_energy_rmse': [],
        'train_force_rmse': [],
        'val_force_rmse': [],
        'train_stress_rmse': [],
        'val_stress_rmse': [],
    }

    ckpt_interval = int(config.get('checkpoint_interval', 0) or 0)

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
    
    print(
        f"{'Epoch':>5} | {'Loss':>10} | {'E (meV)':>10} | {'force_RMSE':>12} | {'force_MSE':>12} | {'force_MAE':>12} | {'S_RMSE':>10} || "
        f"{'Val Loss':>10} | {'Val E':>10} | {'Val force_RMSE':>16} | {'Val force_MSE':>16}"
    )
    print("-" * 170)
    
    force_loss_ema = None
    for epoch in range(start_epoch, config['epochs']):
        force_weight = _force_weight(epoch)
        model.train()
        train_metrics = MetricTracker()
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
                    stress_target = (
                        (torch.norm(item['t_S']) > 1e-6).item()
                        and float(config.get('stress_weight', 0.0)) > 0.0
                    )
                    p_E, p_F, p_S, aux = model(
                        item,
                        training=True,
                        temperature_scale=temp_scale,
                        detach_pos=force_consistency_weight <= 0.0,
                        compute_stress=stress_target,
                    )
                    n_ats = len(item['z'])

                    target_E = item['t_E'] - baseline_energy(item['z'])
                    loss_e = ((p_E - target_E) / n_ats)**2
                    loss_f = torch.mean((p_F - item['t_F'])**2)
                    loss_s = torch.tensor(0.0, device=device)
                    if stress_target:
                        loss_s = torch.mean((p_S - item['t_S'])**2)

                    loss_item = (config['energy_weight'] * loss_e) + \
                                (force_weight * loss_f) + \
                                (config['stress_weight'] * loss_s)

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
                            compute_stress=False,
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
        tr_e, tr_f, tr_s, tr_f_mse, tr_f_mae = train_metrics.get_metrics()
        # Validation
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
                    stress_target = (
                        (torch.norm(item['t_S']) > 1e-6).item()
                        and float(config.get('stress_weight', 0.0)) > 0.0
                    )
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
                    val_loss_accum += (
                        (config['energy_weight'] * loss_e)
                        + (force_weight * loss_f)
                        + (config['stress_weight'] * loss_s)
                    )

                pred_E_abs = p_E + baseline_energy(item['z'])
                val_metrics.update(pred_E_abs, p_F, p_S, item['t_E'], item['t_F'], item['t_S'], n_ats)

        avg_val_loss = val_loss_accum / len(val_atoms)
        val_e, val_f, val_s, val_f_mse, val_f_mae = val_metrics.get_metrics()
        avg_val_loss = float(avg_val_loss.detach().cpu())
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(avg_val_loss)
        history['train_energy_rmse'].append(float(tr_e))
        history['val_energy_rmse'].append(float(val_e))
        history['train_force_rmse'].append(float(tr_f))
        history['val_force_rmse'].append(float(val_f))
        history['train_stress_rmse'].append(float(tr_s))
        history['val_stress_rmse'].append(float(val_s))
        if scheduler_interval == 'epoch':
            scheduler.step()

        print(
            f"{epoch+1:5d} | "
            f"{avg_train_loss:10.4f} | {tr_e:10.2f} | {tr_f:12.6f} | {tr_f_mse:12.6f} | {tr_f_mae:12.6f} | {tr_s:10.4f} || "
            f"{avg_val_loss:10.4f} | {val_e:10.2f} | {val_f:16.6f} | {val_f_mse:16.6f}"
        )

        if ckpt_interval > 0 and (epoch + 1) % ckpt_interval == 0:
            ckpt_dir = config.get('checkpoint_dir', 'checkpoints')
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

    if config.get('plot_training_curves', True):
        try:
            plot_metric_history(history, save_dir=config.get('plot_dir', 'plots'))
        except Exception as error:
            print(f"[PLOTTING] Could not create training curves: {error}")

if __name__ == "__main__": main()
