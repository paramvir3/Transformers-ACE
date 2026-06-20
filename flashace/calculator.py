import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from .model import TransformersACE

class TransformersACECalculator(Calculator):
    """
    ASE calculator for Transformers-ACE.
    Uses standard ASE neighbor lists (highly compatible).
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model_path="model.pt", device=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        # 1. Device Setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Transformers-ACE from {model_path} on {self.device}...")

        # 2. Load Model & Config
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if 'config' not in checkpoint:
            raise KeyError("Model file missing 'config'. Please retrain with updated train.py.")

        conf = checkpoint['config']

        self.atomic_energy_map = {int(k): float(v) for k, v in (conf.get('atomic_energies') or {}).items()}
        self.energy_shift_per_atom = float(conf.get('energy_shift_per_atom', 0.0)) if conf.get('energy_shift_per_atom') is not None else 0.0
        self.atomic_energy_tensor = None
        if self.atomic_energy_map:
            max_z = max(self.atomic_energy_map)
            tensor = torch.zeros(max_z + 1, dtype=torch.float32, device=self.device)
            for z, val in self.atomic_energy_map.items():
                tensor[z] = val
            self.atomic_energy_tensor = tensor

        # Ensure cutoff is float
        self.r_max = float(conf['r_max'])

        # 3. Initialize Architecture
        self.model = TransformersACE(
            r_max=self.r_max,
            l_max=conf['l_max'],
            num_radial=conf['num_radial'],
            hidden_dim=conf['hidden_dim'],
            num_layers=conf['num_layers'],
            radial_basis_type=conf.get('radial_basis_type', 'bessel'),
            radial_trainable=conf.get('radial_trainable', False),
            envelope_exponent=conf.get('envelope_exponent', 5),
            gaussian_width=conf.get('gaussian_width', 0.5),
            descriptor_passes=conf.get('descriptor_passes', 1),
            descriptor_residual=conf.get('descriptor_residual', True),
            radial_mlp_hidden=conf.get('radial_mlp_hidden', 64),
            radial_mlp_layers=conf.get('radial_mlp_layers', 2),
            attention_num_heads=conf.get('attention_num_heads', conf.get('transformer_num_heads', 4)),
            attention_key_dim=conf.get('attention_key_dim', None),
            attention_ffn_hidden=conf.get('attention_ffn_hidden', conf.get('transformer_ffn_hidden', None)),
            attention_dropout=conf.get('attention_dropout', conf.get('transformer_dropout', 0.0)),
            attention_layer_scale_init=conf.get('attention_layer_scale_init', 1e-2),
            attention_distance_penalty=conf.get('attention_distance_penalty', True),
            use_aux_force_head=False,
            use_aux_stress_head=False,
        )
        
        # 4. Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        # Standard ASE setup
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 1. Periodic-correct neighbor list. ASE shift S gives
        # r_j - r_i + S @ cell, so edges are stored as neighbor -> center.
        i, j, shifts = neighbor_list('ijS', atoms, self.r_max)
        edge_index = torch.stack(
            [torch.tensor(j, dtype=torch.long), torch.tensor(i, dtype=torch.long)], dim=0
        ).to(self.device)
        edge_shift = torch.tensor(shifts, dtype=torch.float32, device=self.device)
        
        # 2. Prepare Data
        z = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device)
        pos = torch.tensor(atoms.positions, dtype=torch.float32, device=self.device)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32, device=self.device)
        
        # Volume handling (Use 1.0 for non-periodic systems)
        if atoms.pbc.any():
            vol = atoms.get_volume()
        else:
            vol = 1.0

        data = {
            'z': z,
            'pos': pos,
            'cell': cell,
            'edge_index': edge_index,
            'edge_shift': edge_shift,
            'volume': torch.tensor(vol, dtype=torch.float32, device=self.device),
        }
        
        # 3. Run Model
        calc_stress = 'stress' in properties

        pred_E, pred_F, pred_S, _ = self.model(
            data,
            training=False,
            compute_stress=calc_stress,
        )

        if self.atomic_energy_tensor is not None:
            if torch.max(z).item() >= self.atomic_energy_tensor.shape[0]:
                raise ValueError("Encountered atomic number without reference energy in atomic_energies")
            baseline = torch.sum(self.atomic_energy_tensor[z])
        else:
            baseline = torch.tensor(self.energy_shift_per_atom * len(atoms), device=self.device)

        pred_E = pred_E + baseline

        # 4. Store Results
        self.results['energy'] = pred_E.item()
        self.results['forces'] = pred_F.detach().cpu().numpy()
        
        if calc_stress:
            S_mat = pred_S.detach().cpu().numpy()
            # Convert 3x3 to Voigt (xx, yy, zz, yz, xz, xy)
            self.results['stress'] = np.array([
                S_mat[0,0], S_mat[1,1], S_mat[2,2],
                S_mat[1,2], S_mat[0,2], S_mat[0,1]
            ])


# Preserve the original calculator import for existing user scripts.
FlashACECalculator = TransformersACECalculator
