from typing import Union, Dict, Tuple, List
import os
import time
import tqdm
import warnings
import collections
import itertools

from pymatgen.core.structure import Structure
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset 
from torch_geometric.datasets import MD17
from pathos.multiprocessing import ProcessingPool as Pool

from .graph import get_graph, load_orbital_types
from .utils import process_targets
from .e3modules import e3TensorDecomp
from torch_geometric.data import Data, Batch
from .from_pymatgen.lattice import find_neighbors, _one_to_three, _compute_cube_index, _three_to_one

class AijData(InMemoryDataset):
    def __init__(self, raw_data_dir: str, graph_dir: str, target: str,
                 dataset_name : str, multiprocessing: bool, radius: float, max_num_nbr: int, edge_Aij: bool,
                 default_dtype_torch, nums: int = None, inference:bool = False, only_ij: bool = False, load_graph=True):
        """
        :param raw_data_dir: 原始数据目录, 允许存在嵌套
when interface == 'h5',
raw_data_dir
├── 00
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 01
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 02
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── ...
        :param graph_dir: 存储图的目录
        :param multiprocessing: 多进程生成图
        :param radius: 生成图的截止半径
        :param max_num_nbr: 生成图限制最大近邻数, 为 0 时不限制
        :param edge_Aij: 图的边是否一一对应 Aij, 如果是为 False 则存在两套图的连接, 一套用于节点更新, 一套用于记录 Aij
        :param default_dtype_torch: 浮点数数据类型
        """
        self.raw_data_dir = raw_data_dir
        assert dataset_name.find('-') == -1, '"-" can not be included in the dataset name'
        create_from_DFT = radius < 0
        radius_info = 'rFromDFT' if create_from_DFT else f'{radius}r{max_num_nbr}mn'
        if target == 'hamiltonian':
            graph_file_name = f'HGraph-{dataset_name}-{radius_info}-edge{"" if edge_Aij else "!"}=Aij{"-undrct" if only_ij else ""}.pkl' # undrct = undirected
        elif target == 'density_matrix':
            graph_file_name = f'DMGraph-{dataset_name}-{radius_info}-{edge_Aij}edge{"-undrct" if only_ij else ""}.pkl'
        else:
            raise ValueError('Unknown prediction target: {}'.format(target))
        self.data_file = os.path.join(graph_dir, graph_file_name)
        os.makedirs(graph_dir, exist_ok=True)
        self.data, self.slices = None, None
        self.target = target
        self.target_file_name = 'overlaps.h5' if inference else f'{self.target}s.h5'
        self.dataset_name = dataset_name
        self.multiprocessing = multiprocessing
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.create_from_DFT = create_from_DFT
        self.edge_Aij = edge_Aij
        self.default_dtype_torch = default_dtype_torch

        self.nums = nums
        self.inference = inference
        self.only_ij = only_ij
        self.transform = None
        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        print(f'Graph data file: {graph_file_name}')
        if os.path.exists(self.data_file):
            print('Use existing graph data file')
        else:
            assert raw_data_dir, 'Required graph does not exist, or graph filename cannot be correctly identified'
            print('Process new data file......')
            self.process()
        if load_graph:
            begin = time.time()
            loaded_data = torch.load(self.data_file)
            self.data, self.slices, self.info = loaded_data
            print(f'Finish loading the processed {len(self)} structures '
                f'(the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.2f} seconds')
    
    @classmethod
    def from_existing_graph(cls, existing_graph_dir, default_dtype_torch):
        assert os.path.isfile(existing_graph_dir), f'Required graph {existing_graph_dir} does not exist'
        save_graph_dir = os.path.dirname(existing_graph_dir)
        existing_graph_dir = os.path.basename(existing_graph_dir)
        assert existing_graph_dir[-4:] == '.pkl', 'graph filename extension should be .pkl'
        options = existing_graph_dir.rstrip('.pkl').split('-')
        if options[0][0] == 'H':
            target = 'hamiltonian'
        elif options[0][0:2] == 'DM':
            target = 'density_matrix'
        else:
            raise ValueError(f'Cannot identify graph file {existing_graph_dir}')
        dataset_name = options[1]
        if options[2] == 'rFromDFT':
            cutoff_radius = -1
        else:
            cutoff_radius = float(options[2].split('r')[0])
        only_ij = options[-1] == 'undrct'
        return cls(
            raw_data_dir=None, 
            graph_dir=save_graph_dir, 
            target=target, 
            dataset_name=dataset_name, 
            multiprocessing=False, 
            radius=cutoff_radius, 
            max_num_nbr=0,               #todo
            edge_Aij=True,               #todo
            default_dtype_torch=default_dtype_torch,
            only_ij=only_ij,
            )

    def process_worker(self, folder, **kwargs):
        stru_id = os.path.split(folder)[-1]

        
        site_positions = np.loadtxt(os.path.join(folder, 'site_positions.dat')).T
        elements = np.loadtxt(os.path.join(folder, 'element.dat'))
        if len(elements.shape) == 0:
            elements = elements[None]
            site_positions = site_positions[None, :]
        structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')).T,
                              elements,
                              site_positions,
                              coords_are_cartesian=True,
                              to_unit_cell=False)

        cart_coords = torch.tensor(structure.cart_coords, dtype=self.default_dtype_torch)
        frac_coords = torch.tensor(structure.frac_coords, dtype=self.default_dtype_torch)
        numbers = torch.tensor(structure.atomic_numbers)
        structure.lattice.matrix.setflags(write=True)
        lattice = torch.tensor(structure.lattice.matrix, dtype=self.default_dtype_torch)
        return get_graph(cart_coords, frac_coords, numbers, stru_id, r=self.radius, max_num_nbr=self.max_num_nbr,
                         edge_Aij=self.edge_Aij, lattice=lattice, default_dtype_torch=self.default_dtype_torch,
                         data_folder=folder, target_file_name=self.target_file_name, inference=self.inference, 
                         only_ij=self.only_ij, create_from_DFT=self.create_from_DFT, **kwargs)
    
    def process(self):
        begin = time.time()
        folder_list = []
        print(f'Looking for preprocessed data under: {self.raw_data_dir}')
        if True:
            cart_coords = np.load(os.path.join(self.raw_data_dir, 'set.000', 'coord.npy'), allow_pickle=True)
            datasize = len(cart_coords)
            # cart_coords = torch.tensor(cart_coords.reshape(datasize, -1, 3), dtype=self.default_dtype_torch)
            cart_coords = [torch.tensor(cart_coord.reshape(-1, 3), dtype=self.default_dtype_torch) for cart_coord in cart_coords]
            # lattice = torch.tensor(np.load(os.path.join(self.raw_data_dir, 'set.000', 'box.npy')), dtype=self.default_dtype_torch).reshape(datasize,3,3)
            lattice = [torch.tensor(l.reshape(3, 3), dtype=self.default_dtype_torch) for l in 
                       np.load(os.path.join(self.raw_data_dir, 'set.000', 'box.npy'), allow_pickle=True)]
            if False:
                numbers_file = os.path.join(self.raw_data_dir, 'type.raw')
                with open(numbers_file, 'r') as file:
                    numbers = file.readlines()
                numbers = torch.tensor([int(line.strip()) for line in numbers])
            else:
                numbers = [torch.tensor(atom) for atom in np.load(os.path.join(self.raw_data_dir, 'set.000', 'atom.npy'), allow_pickle=True)]
            # energy = torch.tensor(np.load(os.path.join(self.raw_data_dir, 'set.000', 'energy.npy')), dtype=self.default_dtype_torch)
            # force = torch.tensor(np.load(os.path.join(self.raw_data_dir, 'set.000', 'force.npy')), dtype=self.default_dtype_torch).reshape(datasize,-1,3)
            energy = [torch.tensor(energy, dtype=self.default_dtype_torch) for energy in 
                      np.load(os.path.join(self.raw_data_dir, 'set.000', 'energy.npy'), allow_pickle=True)]
            force = [torch.tensor(force.reshape(-1,3), dtype=self.default_dtype_torch) for force in 
                     np.load(os.path.join(self.raw_data_dir, 'set.000', 'force.npy'), allow_pickle=True)]
        
        else:
            dataset = MD17(root='/home/yinshi/workspace/pxy/MD17', name='aspirin').shuffle()[:1000]
            datasize = len(dataset)
            lattice = [torch.tensor([[100.0,0,0],[0,100,0],[0,0,100]]) for data in dataset]
            cart_coords = [data.pos for data in dataset]
            numbers = dataset[0].z
            energy = [data.energy for data in dataset]
            force = [data.force for data in dataset]
        
        data_list = [self.graph(cart_coords[i], lattice[i], numbers[i], energy[i], force[i]) for i in range(datasize)]
        print('Finish processing %d structures, have cost %d seconds' % (len(data_list), time.time() - begin))
        index_to_Z, Z_to_index = self.element_statistics(data_list)
        begin = time.time()
        data, slices = self.collate(data_list)
        torch.save((data, slices, dict(spinful=None, index_to_Z=index_to_Z, Z_to_index=Z_to_index, orbital_types=None)), self.data_file)
        print('Finished saving %d structures to save_graph_dir, have cost %d seconds' % (len(data_list), time.time() - begin))

    def element_statistics(self, data_list):
        # TODO 没有处理数据集包括不同元素组成的情况
        index_to_Z, inverse_indices = torch.unique(data_list[0].x, sorted=True, return_inverse=True)
        Z_to_index = torch.full((100,), -1, dtype=torch.int64)
        Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))

        # for data in data_list:
        #     data.x = Z_to_index[data.x]

        return index_to_Z, Z_to_index

    def set_mask(self, targets, del_Aij=True, convert_to_net=False):
        begin = time.time()
        print("\nSetting mask for dataset...")
        
        spinful = self.info['spinful']
        
        dtype = torch.get_default_dtype()
        if spinful:
            if dtype == torch.float32:
                dtype = torch.complex64
            elif dtype == torch.float64:
                dtype = torch.complex128
            else:
                raise ValueError(f'Unsupported dtype: {dtype}')
        
        equivariant_blocks, out_js_list, out_slices = process_targets(self.info['orbital_types'], self.info["index_to_Z"], targets)
        if convert_to_net:
            construct_kernel = e3TensorDecomp(None, out_js_list, torch.get_default_dtype(), spinful=spinful, if_sort=True) # todo: dtype
        
        atom_num_orbital = [sum(map(lambda x: 2 * x + 1,atom_orbital_types)) for atom_orbital_types in self.info['orbital_types']]

        data_list_mask = []
        for data in self:
            assert data.spinful == spinful
            if data.Aij is not None:
                if not torch.all(data.Aij_mask):
                    raise NotImplementedError("Not yet have support for graph radius including Aij without calculation")

            # label of each edge is a vector which is each target H block flattened and concatenated
            if spinful:
                label = torch.zeros(data.num_edges, 4, out_slices[-1], dtype=dtype)
                mask = torch.zeros(data.num_edges, 4, out_slices[-1], dtype=torch.int8)
            else:
                label = torch.zeros(data.num_edges, out_slices[-1], dtype=dtype)
                mask = torch.zeros(data.num_edges, out_slices[-1], dtype=torch.int8)

            atomic_number_edge_i = data.x[data.edge_index[0]]
            atomic_number_edge_j = data.x[data.edge_index[1]]

            for index_out, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    condition_atomic_number_i, condition_atomic_number_j = map(lambda x: self.info["Z_to_index"][int(x)], N_M_str.split())
                    condition_slice_i = slice(block_slice[0], block_slice[1])
                    condition_slice_j = slice(block_slice[2], block_slice[3])
                    if spinful:
                        condition_slice_i_ds = slice(atom_num_orbital[condition_atomic_number_i] + block_slice[0],
                                                      atom_num_orbital[condition_atomic_number_i] + block_slice[1]) # ds = down spin
                        condition_slice_j_ds = slice(atom_num_orbital[condition_atomic_number_j] + block_slice[2],
                                                     atom_num_orbital[condition_atomic_number_j] + block_slice[3])
                    if data.Aij is not None:
                        out_slice = slice(out_slices[index_out], out_slices[index_out + 1])
                        condition_index = torch.where(
                            (atomic_number_edge_i == condition_atomic_number_i)
                            & (atomic_number_edge_j == condition_atomic_number_j)
                        )
                        if spinful:
                            # noncollinear spin block order:
                            # 0(uu) 1(ud)
                            # 2(du) 3(dd)
                            label[condition_index[0], 0, out_slice] += data.Aij[:, condition_slice_i, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 1, out_slice] += data.Aij[:, condition_slice_i, condition_slice_j_ds].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 2, out_slice] += data.Aij[:, condition_slice_i_ds, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 3, out_slice] += data.Aij[:, condition_slice_i_ds, condition_slice_j_ds].reshape(data.num_edges, -1)[condition_index]
                            mask[condition_index[0], :, out_slice] += 1
                        else:
                            label[condition_index[0], out_slice] += data.Aij[:, condition_slice_i, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            mask[condition_index[0], out_slice] += 1
            if del_Aij:
                del data.Aij_mask
            if data.Aij is not None:
                if convert_to_net:
                    label = construct_kernel.get_net_out(label)
                data.label = label
                assert torch.all((mask == 1) | (mask == 0)), 'Some blocks are required to predict multiple times'
                mask = mask.bool()
                if spinful and convert_to_net:
                    mask = construct_kernel.convert_mask(mask)
                data.mask = mask
                if del_Aij:
                    del data.Aij
            data_list_mask.append(data)

        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None
        data, slices = self.collate(data_list_mask)
        self.data, self.slices = data, slices
        print(f"Finished setting mask for dataset, cost {time.time() - begin:.2f} seconds")

        return out_js_list, out_slices
    
    def graph(self, cart_coords, lattice, numbers, energy, force):
        print(cart_coords.shape, numbers.shape)
        numerical_tol=1e-8
        r = 10
        max_num_nbr = 0
        default_dtype_torch = self.default_dtype_torch
        frac_coords = cart_coords @ torch.linalg.inv(lattice.T) #需要验证
        cart_coords_np = cart_coords.detach().numpy()
        frac_coords_np = frac_coords.detach().numpy()
        lattice_np = lattice.detach().numpy()
        num_atom = cart_coords.shape[0]

        center_coords_min = np.min(cart_coords_np, axis=0)
        center_coords_max = np.max(cart_coords_np, axis=0)
        # The lower bound of all considered atom coords
        global_min = center_coords_min - r - numerical_tol
        global_max = center_coords_max + r + numerical_tol
        global_min_torch = torch.tensor(global_min)
        global_max_torch = torch.tensor(global_max)

        reciprocal_lattice = np.linalg.inv(lattice_np).T * 2 * np.pi
        recp_len = np.sqrt(np.sum(reciprocal_lattice ** 2, axis=1))
        maxr = np.ceil((r + 0.15) * recp_len / (2 * np.pi))
        nmin = np.floor(np.min(frac_coords_np, axis=0)) - maxr
        nmax = np.ceil(np.max(frac_coords_np, axis=0)) + maxr
        all_ranges = [np.arange(x, y, dtype='int64') for x, y in zip(nmin, nmax)]
        images = torch.tensor(list(itertools.product(*all_ranges))).type_as(lattice)

        # Filter out those beyond max range
        # coords = (images @ lattice).unsqueeze(1).expand(-1, num_atom, 3) + cart_coords.unsqueeze(0).expand(images.shape[0], num_atom, 3)
        coords = (images @ lattice)[:, None, :] + cart_coords[None, :, :]
        indices = torch.arange(num_atom).unsqueeze(0).expand(images.shape[0], num_atom)
        valid_index_bool = coords.gt(global_min_torch) * coords.lt(global_max_torch)
        valid_index_bool = valid_index_bool.all(dim=-1)
        valid_coords = coords[valid_index_bool]
        valid_indices = indices[valid_index_bool]


        # Divide the valid 3D space into cubes and compute the cube ids
        valid_coords_np = valid_coords.detach().numpy()
        all_cube_index = _compute_cube_index(valid_coords_np, global_min, r)
        nx, ny, nz = _compute_cube_index(global_max, global_min, r) + 1
        all_cube_index = _three_to_one(all_cube_index, ny, nz)
        site_cube_index = _three_to_one(_compute_cube_index(cart_coords_np, global_min, r), ny, nz)
        # create cube index to coordinates, images, and indices map
        cube_to_coords_index = collections.defaultdict(list)  # type: Dict[int, List]

        # for cart_coord, j, k, l in zip(all_cube_index.ravel(), valid_coords_np, valid_images, valid_indices):
        for index, cart_coord in enumerate(all_cube_index.ravel()):
            cube_to_coords_index[cart_coord].append(index)

        # find all neighboring cubes for each atom in the lattice cell
        site_neighbors = find_neighbors(site_cube_index, nx, ny, nz)

        #####
        inv_lattice = torch.inverse(lattice).type(default_dtype_torch) # used for constructing key
        edge_idx, edge_fea, edge_idx_first, edge_key = [], [], [], []
        for index_first, (cart_coord, j) in enumerate(zip(cart_coords, site_neighbors)):
            l1 = np.array(_three_to_one(j, ny, nz), dtype=int).ravel()
            # use the cube index map to find the all the neighboring
            # coords, images, and indices
            ks = [k for k in l1 if k in cube_to_coords_index]
            nn_coords_index = np.concatenate([cube_to_coords_index[k] for k in ks], axis=0)
            nn_coords = valid_coords[nn_coords_index]
            nn_indices = valid_indices[nn_coords_index]
            dist = torch.norm(nn_coords - cart_coord[None, :], dim=1)

            # allow edge with distance = 0
            if True:
                nn_coords = nn_coords.squeeze()
                nn_indices = nn_indices.squeeze()
                dist = dist.squeeze()
            else:
                nonzero_index = dist.nonzero(as_tuple=False)
                nn_coords = nn_coords[nonzero_index]
                nn_coords = nn_coords.squeeze(1)
                nn_indices = nn_indices[nonzero_index].view(-1)
                dist = dist[nonzero_index].view(-1)
                
            R = torch.round((nn_coords - cart_coords[nn_indices]) @ inv_lattice).int()
            edge_key_single = torch.cat([R, 
                                        torch.full([R.shape[0], 1], index_first) + 1, 
                                        nn_indices.unsqueeze(1) + 1], dim=1)
            edge_is_ij = torch.full([nn_indices.shape[0]], True)

            if max_num_nbr > 0:
                if len(dist) >= max_num_nbr:
                    dist_top, index_top = dist.topk(max_num_nbr, largest=False, sorted=True)
                    edge_idx.extend(nn_indices[index_top])
                    edge_idx_first.extend([index_first] * len(index_top))
                    edge_fea_single = torch.cat([dist_top.view(-1, 1), nn_coords[index_top] - cart_coord], dim=-1)
                    edge_fea.append(edge_fea_single)
                else:
                    warnings.warn("Can not find a number of max_num_nbr atoms within the radius")
                    edge_idx.extend(nn_indices)
                    edge_idx_first.extend([index_first] * len(nn_indices))
                    edge_fea_single = torch.cat([dist.view(-1, 1), nn_coords - cart_coord], dim=-1)
                    edge_fea.append(edge_fea_single)
            else:
                index_top = dist.lt(r + numerical_tol) * edge_is_ij
                edge_idx.extend(nn_indices[index_top])
                edge_idx_first.extend([index_first] * len(nn_indices[index_top]))
                edge_fea_single = torch.cat([dist[index_top].view(-1, 1), nn_coords[index_top] - cart_coord], dim=-1)
                edge_fea.append(edge_fea_single)
                edge_key.append(edge_key_single[index_top])

        edge_fea = torch.cat(edge_fea).type(default_dtype_torch)
        edge_idx_first = torch.LongTensor(edge_idx_first)
        edge_idx = torch.stack([edge_idx_first, torch.LongTensor(edge_idx)])
        # edge_key: shape [num_edge,5], each row [Rx, Ry, Rz, i, j] where i and j both start from 1
        # to get the hopping key for ith edge, use str(edge_key[i].tolist())
        edge_key = torch.cat(edge_key, dim=0)

        data = Data(x=numbers, edge_index=edge_idx, edge_attr=edge_fea, stru_id=None,
                        pos=cart_coords.type(default_dtype_torch),
                        lattice=lattice.unsqueeze(0),
                        edge_key=edge_key,
                        Aij=None,
                        Aij_mask=None,
                        atom_num_orbital=None,
                        spinful=None,
                        energy=energy,
                        force=force,)
        return data