import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from .model import FlashACE

class FlashACECalculator(Calculator):
    """
    ASE Calculator for FlashACE.
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
            
        print(f"Loading FlashACE from {model_path} on {self.device}...")

        # 2. Load Model & Config
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
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
        self.model = FlashACE(
            r_max=self.r_max,
            l_max=conf['l_max'],
            num_radial=conf['num_radial'],
            hidden_dim=conf['hidden_dim'],
            num_layers=conf['num_layers'],
            radial_basis_type=conf.get('radial_basis_type', 'bessel'),
            radial_trainable=conf.get('radial_trainable', False),
            envelope_exponent=conf.get('envelope_exponent', 5),
            gaussian_width=conf.get('gaussian_width', 0.5),
            transformer_num_heads=conf.get('transformer_num_heads', 4),
            transformer_ffn_hidden=conf.get('transformer_ffn_hidden', None),
            transformer_dropout=conf.get('transformer_dropout', 0.0),
            transformer_residual_dropout=conf.get('transformer_residual_dropout', 0.0),
            transformer_ffn_gated=conf.get('transformer_ffn_gated', False),
            transformer_layer_scale_init=conf.get('transformer_layer_scale_init', None),
            transformer_attention_chunk_size=conf.get('transformer_attention_chunk_size', None),
            use_transformer=conf.get('use_transformer', True),
            transformer_scalar_only=conf.get('transformer_scalar_only', False),
            attention_neighbor_mask=conf.get('attention_neighbor_mask', False),
            attention_short_range=conf.get('attention_short_range', False),
            attention_short_range_ratio=conf.get('attention_short_range_ratio', 0.5),
            attention_short_range_gate=conf.get('attention_short_range_gate', True),
            descriptor_passes=conf.get('descriptor_passes', 1),
            descriptor_residual=conf.get('descriptor_residual', True),
            radial_mlp_hidden=conf.get('radial_mlp_hidden', 64),
            radial_mlp_layers=conf.get('radial_mlp_layers', 2),
            message_passing_layers=conf.get('message_passing_layers', 0),
            interleave_descriptor=conf.get('interleave_descriptor', False),
            edge_update_per_layer=conf.get('edge_update_per_layer', False),
            node_update_mlp=conf.get('node_update_mlp', False),
            equivariant_mix_per_layer=conf.get('equivariant_mix_per_layer', False),
            edge_state_dim=conf.get('edge_state_dim', None),
            edge_attention=conf.get('edge_attention', False),
            equivariant_rms_norm=conf.get('equivariant_rms_norm', False),
            equivariant_rms_norm_eps=conf.get('equivariant_rms_norm_eps', 1e-8),
            readout_hidden_dims=conf.get('readout_hidden_dims', None),
            use_equiformer_v2=conf.get('use_equiformer_v2', False),
            use_aux_force_head=conf.get('use_aux_force_head', False),
            use_aux_stress_head=conf.get('use_aux_stress_head', False),
        )
        
        # 4. Load Weights
        state_dict = checkpoint['model_state_dict']
        model_state = self.model.state_dict()
        filtered_state = {}
        mismatched = []
        for key, value in state_dict.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered_state[key] = value
        if mismatched:
            print("[FlashACE] Skipping mismatched checkpoint tensors:")
            for key, old_shape, new_shape in mismatched:
                print(f"  - {key}: checkpoint {old_shape} vs model {new_shape}")
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        # Standard ASE setup
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # 1. Neighbor List (Standard ASE)
        i, j = neighbor_list('ij', atoms, self.r_max)
        edge_index = torch.stack([torch.tensor(i), torch.tensor(j)], dim=0).to(self.device)
        
        # 2. Prepare Data
        z = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device)
        pos = torch.tensor(atoms.positions, dtype=torch.float32, device=self.device)
        
        # Volume handling (Use 1.0 for non-periodic systems)
        if atoms.pbc.any():
            vol = atoms.get_volume()
        else:
            vol = 1.0

        data = {
            'z': z, 'pos': pos, 'edge_index': edge_index,
            'volume': torch.tensor(vol, dtype=torch.float32, device=self.device)
        }
        
        # 3. Run Model
        calc_stress = 'stress' in properties

        # If calculating stress, enable gradients w.r.t cell (training=True)
        if calc_stress:
            pred_E, pred_F, pred_S, _ = self.model(data, training=True)
        else:
            pred_E, pred_F, _, _ = self.model(data, training=False)

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
