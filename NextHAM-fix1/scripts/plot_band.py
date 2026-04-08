import pyatb
import numpy as np
import re
import os
import json
from ase.io import read, write
from pyatb.kpt import kpoint_generator
from pyatb import RANK, COMM, SIZE, OUTPUT_PATH, RUNNING_LOG, timer
from pyatb.parallel import op_gather_numpy
from matplotlib import pyplot as plt


K_B_EV = 8.617333262145e-5




class plot_band: 
    def __init__(self, nspin, stru_file, hr1, sr1, emin, emax, mu, save_path):
        self.nspin = nspin
        self.stru_file = stru_file
        self.hr1 = hr1
        self.sr1 = sr1
        self.emin = emin
        self.emax = emax
        self.mu = mu
        self.save_path = save_path
        self.band_summary = None
        self.gap_mp_grid = (20, 20, 20)

    def _read_nelec_from_running_log(self):
        candidates = [
            os.path.join(self.save_path, 'running_get_hs.log'),
            os.path.join(self.save_path, 'running_scf.log'),
            os.path.join(self.save_path, 'running.log'),
            os.path.join(os.path.dirname(self.hr1), 'running_get_hs.log'),
            os.path.join(os.path.dirname(self.hr1), 'running_scf.log'),
            os.path.join(os.path.dirname(self.hr1), 'running.log'),
        ]
        patterns = [
            re.compile(r'AUTOSET\s+number\s+of\s+electrons\s*:\s*=\s*([0-9.+-Ee]+)', re.I),
            re.compile(r'number\s+of\s+electrons\s*[:=]\s*([0-9.+-Ee]+)', re.I),
            re.compile(r'nelec\s*[:=]\s*([0-9.+-Ee]+)', re.I),
        ]
        for path in candidates:
            if not os.path.isfile(path):
                continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    for pat in patterns:
                        m = pat.search(line)
                        if m:
                            return float(m.group(1))
        return None

    def _collect_eigenvalues_on_mp_grid(self, m_tb1, mp_grid):
        k_start = np.array([0.0, 0.0, 0.0], dtype=float)
        k_vect1 = np.array([1.0, 0.0, 0.0], dtype=float)
        k_vect2 = np.array([0.0, 1.0, 0.0], dtype=float)
        k_vect3 = np.array([0.0, 0.0, 1.0], dtype=float)
        k_generator = kpoint_generator.mp_generator(
            8000,
            k_start,
            k_vect1,
            k_vect2,
            k_vect3,
            np.array(mp_grid, dtype=int),
        )

        eig_local_all = []
        k_local_all = []
        for ik in k_generator:
            ik_process = kpoint_generator.kpoints_in_different_process(SIZE, RANK, ik)
            if ik_process.k_direct_coor_local.shape[0]:
                k_local = np.asarray(ik_process.k_direct_coor_local, dtype=np.float64)
                eig_local = m_tb1.tb_solver.diago_H_eigenvaluesOnly(k_local)
            else:
                k_local = np.zeros((0, 3), dtype=np.float64)
                eig_local = np.zeros((0, m_tb1.tb_solver.basis_num), dtype=np.float64)
            k_local_all.append(k_local)
            eig_local_all.append(eig_local)

        if k_local_all:
            k_local_all = np.concatenate(k_local_all, axis=0)
        else:
            k_local_all = np.zeros((0, 3), dtype=np.float64)

        if eig_local_all:
            eig_local_all = np.concatenate(eig_local_all, axis=0)
        else:
            eig_local_all = np.zeros((0, m_tb1.tb_solver.basis_num), dtype=np.float64)

        k_all = COMM.reduce(k_local_all, root=0, op=op_gather_numpy)
        eig_all = COMM.reduce(eig_local_all, root=0, op=op_gather_numpy)
        return k_all, eig_all

    def _estimate_mu_from_mp_grid(self, m_tb1, mp_grid=(8, 8, 8), temperature=0.0):
        nelec = self._read_nelec_from_running_log()
        if nelec is None:
            raise RuntimeError('无法从 running_get_hs.log / running_scf.log 中读取电子数，不能现场求 mu')

        _, eig_all = self._collect_eigenvalues_on_mp_grid(m_tb1, mp_grid)
        mu = None
        if RANK == 0:
            eig = np.asarray(eig_all, dtype=float).reshape(-1)
            eig.sort()
            total_k = int(np.prod(mp_grid))
            if self.nspin == 1:
                occ_per_state = 2.0 / total_k
            else:
                occ_per_state = 1.0 / total_k
            target = float(nelec)
            cumulative = np.arange(1, eig.size + 1, dtype=float) * occ_per_state
            idx = int(np.searchsorted(cumulative, target, side='left'))
            idx = min(max(idx, 0), eig.size - 1)
            vbm = eig[idx]
            cbm = eig[idx + 1] if idx + 1 < eig.size else eig[idx]
            if cbm > vbm:
                mu = 0.5 * (vbm + cbm)
            else:
                mu = vbm
        mu = COMM.bcast(mu, root=0)
        return float(mu), float(nelec)

    def _analyze_gap_on_full_k_grid(self, kpoints, band, mp_grid):
        band = np.asarray(band, dtype=float)
        kpoints = np.asarray(kpoints, dtype=float)
        shifted = band - float(self.mu)
        valence_mask = shifted <= 0.0
        conduction_mask = shifted > 0.0

        if not np.any(valence_mask):
            raise RuntimeError('无法从全 k 网格结果中找到价带（所有能级均高于费米能）')
        if not np.any(conduction_mask):
            raise RuntimeError('无法从全 k 网格结果中找到导带（所有能级均不高于费米能）')

        vbm_relative = float(np.max(shifted[valence_mask]))
        cbm_relative = float(np.min(shifted[conduction_mask]))
        band_gap = float(cbm_relative - vbm_relative)

        vbm_locs = np.argwhere(np.isclose(shifted, vbm_relative, atol=1e-8, rtol=1e-6))
        cbm_locs = np.argwhere(np.isclose(shifted, cbm_relative, atol=1e-8, rtol=1e-6))
        vbm_k_indices = sorted({int(idx[0]) for idx in vbm_locs})
        cbm_k_indices = sorted({int(idx[0]) for idx in cbm_locs})
        shared_k = sorted(set(vbm_k_indices) & set(cbm_k_indices))
        band_type = 'Direct' if shared_k else 'Indirect'

        direct_gaps = []
        for ik in range(shifted.shape[0]):
            vals = shifted[ik]
            vb_vals = vals[vals <= 0.0]
            cb_vals = vals[vals > 0.0]
            if vb_vals.size and cb_vals.size:
                direct_gaps.append((float(np.min(cb_vals) - np.max(vb_vals)), ik))
        if direct_gaps:
            direct_gap, direct_gap_kindex = min(direct_gaps, key=lambda x: x[0])
        else:
            direct_gap, direct_gap_kindex = band_gap, -1

        def _first_frac(indices):
            if not indices:
                return None
            return [float(x) for x in kpoints[int(indices[0])].tolist()]

        summary = {
            'analysis_k_grid': [int(x) for x in mp_grid],
            'fermi_energy_eV': float(self.mu),
            'electron_number': float(getattr(self, 'nelec', np.nan)),
            'band_gap_eV': band_gap,
            'direct_band_gap_eV': float(direct_gap),
            'direct_gap_kindex': int(direct_gap_kindex),
            'direct_gap_k_frac': _first_frac([direct_gap_kindex]) if direct_gap_kindex >= 0 else None,
            'band_type': band_type,
            'vbm_relative_to_fermi_eV': vbm_relative,
            'cbm_relative_to_fermi_eV': cbm_relative,
            'vbm_absolute_eV': float(self.mu + vbm_relative),
            'cbm_absolute_eV': float(self.mu + cbm_relative),
            'vbm_band_indices': sorted({int(idx[1]) for idx in vbm_locs}),
            'cbm_band_indices': sorted({int(idx[1]) for idx in cbm_locs}),
            'vbm_k_indices': vbm_k_indices,
            'cbm_k_indices': cbm_k_indices,
            'vbm_k_frac': _first_frac(vbm_k_indices),
            'cbm_k_frac': _first_frac(cbm_k_indices),
            'line_band_kpoint_count': int(getattr(self, 'line_band_kpoint_count', -1)),
        }
        self.band_summary = summary
        return summary

    def _write_band_summary(self, summary):
        summary_path = os.path.join(self.save_path, 'band_summary.txt')
        lines = [
            f"fermi_energy_eV = {summary['fermi_energy_eV']:.10f}",
            f"electron_number = {summary['electron_number']:.10f}",
            f"band_gap_eV = {summary['band_gap_eV']:.10f}",
            f"direct_band_gap_eV = {summary['direct_band_gap_eV']:.10f}",
            f"band_type = {summary['band_type']}",
            f"vbm_relative_to_fermi_eV = {summary['vbm_relative_to_fermi_eV']:.10f}",
            f"cbm_relative_to_fermi_eV = {summary['cbm_relative_to_fermi_eV']:.10f}",
            f"vbm_absolute_eV = {summary['vbm_absolute_eV']:.10f}",
            f"cbm_absolute_eV = {summary['cbm_absolute_eV']:.10f}",
            f"vbm_band_indices = {summary['vbm_band_indices']}",
            f"cbm_band_indices = {summary['cbm_band_indices']}",
            f"vbm_k_indices = {summary['vbm_k_indices']}",
            f"cbm_k_indices = {summary['cbm_k_indices']}",
            f"direct_gap_kindex = {summary['direct_gap_kindex']}",
        ]
        if 'analysis_k_grid' in summary:
            lines.append(f"analysis_k_grid = {summary['analysis_k_grid']}")
        if 'vbm_k_frac' in summary:
            lines.append(f"vbm_k_frac = {summary['vbm_k_frac']}")
        if 'cbm_k_frac' in summary:
            lines.append(f"cbm_k_frac = {summary['cbm_k_frac']}")
        if 'direct_gap_k_frac' in summary:
            lines.append(f"direct_gap_k_frac = {summary['direct_gap_k_frac']}")
        if 'line_band_kpoint_count' in summary:
            lines.append(f"line_band_kpoint_count = {summary['line_band_kpoint_count']}")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        with open(os.path.join(self.save_path, 'band_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.save_path, 'fermi_energy.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{summary['fermi_energy_eV']:.10f}\n")
        with open(os.path.join(self.save_path, 'band_gap.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{summary['band_gap_eV']:.10f}\n")
    
    def generate_kline(self, kline_density = 0.03, tolerance=5e-4, kpath = None, knum=0 ):
        stru_file = self.stru_file
        # 读取 STRU 文件
        ase_stru = read(stru_file, format='abacus')
        ase_stru.wrap()
        lattice_vector = np.array(ase_stru.get_cell())
        bandpath = ase_stru.cell.bandpath(path = kpath, density=kline_density , eps = tolerance)
        path_label = bandpath.path
        pattern = re.findall(r'[A-Z][0-9]*', path_label)
        path_label_array = list(pattern)
        cleaned_labels = [label for label in pattern if label != ',']
        special_points = bandpath.special_points
        shifted_points = {
            key: value if np.all((value >= -1) & (value <= 1)) else np.mod(value, 1)
            for key, value in special_points.items()
        }
        special_points = shifted_points

        kpt_output = []
        kpoint_label = []
        kpoint_num_in_line = []

        rec_lat_cell = ase_stru.cell.reciprocal()
        rec_lat_matrix  = rec_lat_cell[:]
        for i in range(len(path_label_array)):
            label = path_label_array[i]
            label_next = path_label_array[i+1] if i+1 < len(path_label_array) else None
            coordinates = special_points.get(label)
            coordinates_next = special_points.get(label_next) if label_next else None
            if coordinates is not None and coordinates_next is not None:
                if knum == 0:
                    k_real = coordinates @ rec_lat_matrix
                    k_real_next = coordinates_next @ rec_lat_matrix
                    distance_in_reciprocal = np.linalg.norm(k_real - k_real_next)
                    distance = distance_in_reciprocal
                    density = max(int(distance * (2* np.pi) / kline_density ), 3)
                    kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format(density, '<4')}   # {label}")
                    kpoint_label.append(f"{label}   ")
                    kpoint_num_in_line.append(f"{density}  ")
                else:
                    kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format(knum, '<4')}   # {label}")
                    kpoint_label.append(f"{label}   ")
                    kpoint_num_in_line.append(f"{knum}  ")
            elif coordinates is None:
                pass
            else:
                kpt_output.append(f"{'  '.join([f'{coord: .10f}' for coord in coordinates])}  {format('1', '<4')}   # {label}")
                kpoint_label.append(f"{label}   ")
                kpoint_num_in_line.append(f"{ format('1', '<4') }")
        
        kpt_output.append(f' kpoint_label{" " * 20}{",".join(cleaned_labels)}')
        
        kpoint_num_in_line_list = []
        high_symmetry_kpoint_list = []
        for ii in range(len(path_label_array)):
            label = path_label_array[ii]
            coordinates = special_points.get(label)
            high_symmetry_kpoint_list.append(coordinates)
            kpoint_num_in_line_list.append(int(kpoint_num_in_line[ii]))
        high_symmetry_kpoint = np.stack(high_symmetry_kpoint_list, axis=0)
        kpoint_num_in_line = np.array(kpoint_num_in_line_list)
        kline = kpoint_generator.line_generator(8000, high_symmetry_kpoint, kpoint_num_in_line)
        return kpt_output, kpoint_label, kpoint_num_in_line, kline, lattice_vector
    
    def cal_band(self):
        kpt_output, kpoint_label, kpoint_num_in_line, kline, lattice_vector = self.generate_kline()

        m_tb1 = pyatb.init_tb(
                package = 'ABACUS',
                nspin = self.nspin,
                lattice_constant = 1,
                lattice_vector = lattice_vector,
                max_kpoint_num = 8000,
                isSparse = False,
                HR_route = self.hr1,
                HR_unit = 'Ry',
                SR_route = self.sr1,
                need_rR = False,
                rR_route = None,
                rR_unit = 'Bohr',
        )

        self.mu, self.nelec = self._estimate_mu_from_mp_grid(m_tb1)
        if RANK == 0:
            print(f'Start to calculate band structure (estimated mu = {self.mu:.10f} eV, nelec = {self.nelec:.1f})', flush=True)
            print(f'Band gap analysis will scan full k-grid: {self.gap_mp_grid}', flush=True)
        COMM.Barrier()

        for ik in kline:
            ik_process = kpoint_generator.kpoints_in_different_process(SIZE, RANK, ik)
            kpoint_num = ik_process.k_direct_coor_local.shape[0]

            if kpoint_num:
                eigenvalues1= m_tb1.tb_solver.diago_H_eigenvaluesOnly(ik_process.k_direct_coor_local)
            else:
                eigenvalues1 = np.zeros((0, m_tb1.tb_solver.basis_num), dtype=np.float64)

        band1 = COMM.reduce(eigenvalues1, root=0, op=op_gather_numpy)
        grid_k_all, grid_band_all = self._collect_eigenvalues_on_mp_grid(m_tb1, self.gap_mp_grid)

        if RANK == 0:
            print('Band structure calculated finished', flush=True)
            band1_name = os.path.join(self.save_path, 'band1.txt')
            np.savetxt(band1_name, band1)
            self.line_band_kpoint_count = int(band1.shape[0])
            summary = self._analyze_gap_on_full_k_grid(grid_k_all, grid_band_all, self.gap_mp_grid)
            self._write_band_summary(summary)
        COMM.Barrier()    
                
        return band1
    
    def set_fig(self, fig, ax, bwidth=1.0, width=1, mysize=10):
        ax.spines['top'].set_linewidth(bwidth)
        ax.spines['right'].set_linewidth(bwidth)
        ax.spines['left'].set_linewidth(bwidth)
        ax.spines['bottom'].set_linewidth(bwidth)
        ax.tick_params(length=5, width=width, labelsize=mysize)

    def plot_pic(self, kline_density = 0.02):
        band1 = np.loadtxt(os.path.join(self.save_path, 'band1.txt'))
        kpt_output, kpoint_label, kpoint_num_in_line, kline, lattice_vector = self.generate_kline()
        band_data1 = band1 - self.mu
        y_min = self.emin
        y_max = self.emax
        fig_name = os.path.join(self.save_path, 'band1.pdf')
        k_num = band1.shape[0]
        k_length = k_num * kline_density
        x_coor_array = np.linspace(0, k_length, k_num)
        high_symmetry_kpoint_labels = kpoint_label
        high_symmetry_kpoint_x_coor = []
        for ii in range(len(high_symmetry_kpoint_labels)):
            high_symmetry_kpoint_x_coor.append(sum(kpoint_num_in_line[:ii]*kline_density))

        mysize=10
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        self.set_fig(fig, ax,  mysize=mysize)
        linewidth = [1.0, 1.0]
        color = ['red', 'blue']
        linestyle = ['-', '--']  

        ax.plot(x_coor_array, band_data1, color=color[0], linewidth=linewidth[0], linestyle=linestyle[0])
        label = ['pred_tot']
        ax.plot([], [], color=color[0], linewidth=linewidth[0], linestyle=linestyle[0], label=label[0])
        ax.legend(loc="upper right")
        ax.set_title('Band Structure', fontsize=mysize)
        ax.set_ylabel('E - E$_F$ (eV)', fontsize=mysize)
        ax.set_xlim(0, x_coor_array[-1])
        ax.set_ylim(y_min, y_max)
        plt.xticks(high_symmetry_kpoint_x_coor, high_symmetry_kpoint_labels)     
        for i in high_symmetry_kpoint_x_coor:
            plt.axvline(i, color ="grey", alpha = 0.5, lw = 1, linestyle='--')
            ax.axhline(0.0, color ="black", alpha = 1, lw = 1, linestyle='--')

        plt.savefig(fig_name)
        plt.close('all')
