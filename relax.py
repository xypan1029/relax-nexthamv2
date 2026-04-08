import collections
import itertools
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import UnitCellFilter
from ase.optimize import BFGS

WORKFLOW_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WORKFLOW_ROOT / "equiformer_ef"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tg_src.from_pymatgen.lattice import find_neighbors, _compute_cube_index, _three_to_one

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model_1.pth"


class MyModelCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model, directory="."):
        super().__init__(directory=directory)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.results = {}
        self.default_dtype = torch.float32

    @staticmethod
    def _stress_3x3_to_voigt6(stress_3x3: np.ndarray) -> np.ndarray:
        s = stress_3x3
        return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]], dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        pos = torch.tensor(atoms.get_positions(), dtype=self.default_dtype)
        cell = torch.tensor(atoms.get_cell().array, dtype=self.default_dtype)
        types = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        batch = torch.zeros_like(types)
        x, edge_index, edge_key, edge_fea = self.graph(pos, cell, types)
        non_zero_mask = edge_fea[:, 0] != 0
        edge_index = edge_index[:, non_zero_mask]
        edge_fea = edge_fea[non_zero_mask]

        energy, forces, virial = self.model(
            node_atom=x.to(self.device),
            edge_src=edge_index[0].to(self.device),
            edge_dst=edge_index[1].to(self.device),
            edge_vec=edge_fea[:, [1, 2, 3]].to(self.device),
            batch=batch.to(self.device),
        )

        energy = float(energy.detach().cpu())
        forces = np.asarray(forces.detach().cpu(), dtype=float)
        virial = np.asarray(virial[0].detach().cpu(), dtype=float)

        if forces.ndim != 2 or forces.shape[1] != 3:
            raise ValueError(f"forces must have shape (N,3), got {forces.shape}")
        if virial.shape != (3, 3):
            raise ValueError(f"virial must have shape (3,3), got {virial.shape}")

        volume = atoms.get_volume()
        stress_voigt = None if volume <= 0 else self._stress_3x3_to_voigt6(-virial / volume)

        self.results = {"energy": energy, "forces": forces}
        if stress_voigt is not None:
            self.results["stress"] = stress_voigt

    def graph(self, cart_coords, lattice, numbers):
        numerical_tol = 1e-8
        cutoff = 6
        max_num_nbr = 0
        default_dtype_torch = self.default_dtype
        frac_coords = cart_coords @ torch.linalg.inv(lattice)
        cart_coords_np = cart_coords.detach().numpy()
        frac_coords_np = frac_coords.detach().numpy()
        lattice_np = lattice.detach().numpy()
        num_atom = cart_coords.shape[0]

        center_coords_min = np.min(cart_coords_np, axis=0)
        center_coords_max = np.max(cart_coords_np, axis=0)
        global_min = center_coords_min - cutoff - numerical_tol
        global_max = center_coords_max + cutoff + numerical_tol
        global_min_torch = torch.tensor(global_min)
        global_max_torch = torch.tensor(global_max)

        reciprocal_lattice = np.linalg.inv(lattice_np).T * 2 * np.pi
        recp_len = np.sqrt(np.sum(reciprocal_lattice ** 2, axis=1))
        maxr = np.ceil((cutoff + 0.15) * recp_len / (2 * np.pi))
        nmin = np.floor(np.min(frac_coords_np, axis=0)) - maxr
        nmax = np.ceil(np.max(frac_coords_np, axis=0)) + maxr
        all_ranges = [np.arange(x, y, dtype="int64") for x, y in zip(nmin, nmax)]
        images = torch.tensor(list(itertools.product(*all_ranges))).type_as(lattice)

        coords = (images @ lattice)[:, None, :] + cart_coords[None, :, :]
        indices = torch.arange(num_atom).unsqueeze(0).expand(images.shape[0], num_atom)
        valid_index_bool = coords.gt(global_min_torch) * coords.lt(global_max_torch)
        valid_index_bool = valid_index_bool.all(dim=-1)
        valid_coords = coords[valid_index_bool]
        valid_indices = indices[valid_index_bool]

        valid_coords_np = valid_coords.detach().numpy()
        all_cube_index = _compute_cube_index(valid_coords_np, global_min, cutoff)
        nx, ny, nz = _compute_cube_index(global_max, global_min, cutoff) + 1
        all_cube_index = _three_to_one(all_cube_index, ny, nz)
        site_cube_index = _three_to_one(_compute_cube_index(cart_coords_np, global_min, cutoff), ny, nz)
        cube_to_coords_index = collections.defaultdict(list)
        for index, cart_coord in enumerate(all_cube_index.ravel()):
            cube_to_coords_index[cart_coord].append(index)

        site_neighbors = find_neighbors(site_cube_index, nx, ny, nz)

        inv_lattice = torch.inverse(lattice).type(default_dtype_torch)
        edge_idx, edge_fea, edge_idx_first, edge_key = [], [], [], []
        for index_first, (cart_coord, neighbors) in enumerate(zip(cart_coords, site_neighbors)):
            neighbor_cube_ids = np.array(_three_to_one(neighbors, ny, nz), dtype=int).ravel()
            valid_cube_ids = [cube_id for cube_id in neighbor_cube_ids if cube_id in cube_to_coords_index]
            nn_coords_index = np.concatenate([cube_to_coords_index[cube_id] for cube_id in valid_cube_ids], axis=0)
            nn_coords = valid_coords[nn_coords_index]
            nn_indices = valid_indices[nn_coords_index]
            dist = torch.norm(nn_coords - cart_coord[None, :], dim=1)

            nn_coords = nn_coords.squeeze()
            nn_indices = nn_indices.squeeze()
            dist = dist.squeeze()

            R = torch.round((nn_coords - cart_coords[nn_indices]) @ inv_lattice).int()
            edge_key_single = torch.cat(
                [R, torch.full([R.shape[0], 1], index_first) + 1, nn_indices.unsqueeze(1) + 1],
                dim=1,
            )
            edge_is_ij = torch.full([nn_indices.shape[0]], True)

            if max_num_nbr > 0:
                if len(dist) >= max_num_nbr:
                    dist_top, index_top = dist.topk(max_num_nbr, largest=False, sorted=True)
                    edge_idx.extend(nn_indices[index_top])
                    edge_idx_first.extend([index_first] * len(index_top))
                    edge_fea_single = torch.cat([dist_top.view(-1, 1), nn_coords[index_top] - cart_coord], dim=-1)
                    edge_fea.append(edge_fea_single)
                else:
                    edge_idx.extend(nn_indices)
                    edge_idx_first.extend([index_first] * len(nn_indices))
                    edge_fea_single = torch.cat([dist.view(-1, 1), nn_coords - cart_coord], dim=-1)
                    edge_fea.append(edge_fea_single)
            else:
                index_top = dist.lt(cutoff + numerical_tol) * edge_is_ij
                edge_idx.extend(nn_indices[index_top])
                edge_idx_first.extend([index_first] * len(nn_indices[index_top]))
                edge_fea_single = torch.cat([dist[index_top].view(-1, 1), nn_coords[index_top] - cart_coord], dim=-1)
                edge_fea.append(edge_fea_single)
                edge_key.append(edge_key_single[index_top])

        edge_fea = torch.cat(edge_fea).type(default_dtype_torch)
        edge_idx_first = torch.LongTensor(edge_idx_first)
        edge_idx = torch.stack([edge_idx_first, torch.LongTensor(edge_idx)])
        edge_key = torch.cat(edge_key, dim=0)
        return numbers, edge_idx, edge_key, edge_fea


def relax_atoms(
    atoms,
    model=None,
    fmax=0.003,
    steps=1000,
    relax_cell=True,
    optimizer_cls=BFGS,
    optimizer_kwargs=None,
):
    if model is None:
        model = torch.load(DEFAULT_MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    relaxed_atoms = atoms.copy()
    relaxed_atoms.calc = MyModelCalculator(model=model)

    target = UnitCellFilter(relaxed_atoms) if relax_cell else relaxed_atoms
    optimizer_kwargs = optimizer_kwargs or {}
    dyn = optimizer_cls(target, **optimizer_kwargs)
    dyn.run(fmax=fmax, steps=steps)

    return relaxed_atoms