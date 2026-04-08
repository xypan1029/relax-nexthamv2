import os, re
from pathlib import Path
from datetime import datetime
import numpy as np
import shutil
import sys
import subprocess
import torch
from ase.io import read, write
from scripts.read_hs_data import get_hamiltion_data
from scripts.outwf import output_wfc
from scripts.output_hs_data import transform_abacus_hamiltion_data
from scripts.add_element import add_hs_matrix


def _normalize_path_list(paths):
    return [str(Path(p).resolve()) for p in paths]


def prepare_inference_inputs(read_path, save_path, nspin=4):
    read_path = _normalize_path_list(read_path)
    save_path = str(Path(save_path).resolve())

    path_stru = []
    path_weak = []
    path_overlap = []
    for ii in read_path:
        stru_cif = os.path.join(ii, 'OUT.ABACUS/STRU_READIN_ADJUST.cif')
        stru_abacus = os.path.join(ii, 'STRU')
        weak_old = os.path.join(ii, 'OUT.ABACUS/data-HR-sparse_SPIN0.csr')
        weak_new = os.path.join(ii, 'OUT.ABACUS/hrs1_nao.csr')
        overlap_old = os.path.join(ii, 'OUT.ABACUS/data-SR-sparse_SPIN0.csr')
        overlap_new = os.path.join(ii, 'OUT.ABACUS/srs1_nao.csr')
        path_stru.append(stru_cif if os.path.isfile(stru_cif) else stru_abacus)
        path_weak.append(weak_old if os.path.isfile(weak_old) else weak_new)
        path_overlap.append(overlap_old if os.path.isfile(overlap_old) else overlap_new)

    print('-------------------------------------------------', flush=True)
    print('STEP1:开始读入文件', flush=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not (len(path_stru) == len(path_weak) == len(path_overlap)):
        raise ValueError(
            f"Lengths -> stru: {len(path_stru)}, weak: {len(path_weak)}, overlap: {len(path_overlap)}"
        )

    save_dir = []
    for ii in range(len(path_stru)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir_name = f"data_{ii}_{timestamp}"
        subdir = os.path.join(save_path, subdir_name)
        os.mkdir(subdir)
        src_stru = path_stru[ii]
        stru_format = 'cif' if src_stru.endswith('.cif') else 'abacus'
        atoms = read(src_stru, format=stru_format)
        src_weak = path_weak[ii]
        src_overlap = path_overlap[ii]
        src_running_get_hs = os.path.join(read_path[ii], 'OUT.ABACUS/running_get_hs.log')
        dst_stru = os.path.join(subdir, "STRU")
        dst_weak = os.path.join(subdir, "data-HR-sparse_SPIN0_weak.csr")
        dst_overlap = os.path.join(subdir, "data-SR-sparse_SPIN0.csr")
        dst_running_get_hs = os.path.join(subdir, 'running_get_hs.log')
        if not os.path.isfile(src_stru) or not os.path.isfile(src_weak) or not os.path.isfile(src_overlap):
            print(f"Skip copying : source file not complect for {subdir_name}", flush=True)
            save_dir.append(None)
        else:
            write(dst_stru, atoms, format='abacus', scaled=False)
            shutil.copy2(src_weak, dst_weak)
            shutil.copy2(src_overlap, dst_overlap)
            if os.path.isfile(src_running_get_hs):
                shutil.copy2(src_running_get_hs, dst_running_get_hs)
            save_dir.append(subdir)

    for path in save_dir:
        if path:
            stru_file = os.path.join(path, "STRU")
            weak_file = os.path.join(path, "data-HR-sparse_SPIN0_weak.csr")
            overlap_file = os.path.join(path, "data-SR-sparse_SPIN0.csr")
            read_hs = get_hamiltion_data('inference', nspin, path, stru_file, weak_file, overlap_file)
            read_hs.read_stru()
            read_hs.read_data()

    print('STEP1:文件读入完成', flush=True)
    print('-------------------------------------------------', flush=True)
    return [path for path in save_dir if path]


def run_inference_from_prepared_dirs(save_dir, project_root=None):
    project_root = Path(project_root) if project_root is not None else Path(__file__).parent
    project_root = project_root.resolve()
    data_set_path = project_root / 'datasets'
    test_file = data_set_path / 'test.txt'
    out_file = data_set_path / 'outpath.txt'

    input_path = []
    output_path = []
    for path in save_dir:
        if not path:
            continue
        base = Path(path)
        input_path.append(str(base / 'input_inference.pth'))
        output_path.append(str(base / 'output_inference.pth'))

    for f in (test_file, out_file):
        if f.exists():
            f.unlink()

    test_file.write_text("\n".join(input_path) + ("\n" if input_path else ""), encoding='utf-8')
    out_file.write_text("\n".join(output_path) + ("\n" if output_path else ""), encoding='utf-8')

    script = project_root / 'scripts/test/test.sh'
    env = dict(os.environ)
    env['MKL_THREADING_LAYER'] = 'GNU'
    env.pop('LD_PRELOAD', None)
    subprocess.run(['/bin/bash', str(script)], cwd=project_root, env=env, check=True)
    print('STEP2:网络推理完成', flush=True)
    print('-------------------------------------------------', flush=True)


def _postprocess_single_output(path, nspin=4, *, keep_original_band_plot=True):
    stru_file = os.path.join(path, 'STRU')
    file_read = os.path.join(path, 'output_inference.pth')
    transform_data = transform_abacus_hamiltion_data('inference', nspin, stru_file, file_read, path)
    transform_data.read_stru()
    transform_data.get_hs_data()
    predict_hr_cut = os.path.join(path, 'predict_hr_cut')
    weak_file = os.path.join(path, 'data-HR-sparse_SPIN0_weak.csr')
    add_element = add_hs_matrix(nspin, stru_file, predict_hr_cut, weak_file, path)
    add_element.read_stru()
    add_element.add_matrxi_element()
    hr = os.path.join(path, 'predict_hr_tot')
    sr = os.path.join(path, 'data-SR-sparse_SPIN0.csr')

    from scripts.plot_band import plot_band
    emin = -10
    emax = 10
    mu = 2.2853490295
    band_plot = plot_band(nspin, stru_file, hr, sr, emin, emax, mu, path)
    band_plot.cal_band()
    if keep_original_band_plot:
        band_plot.plot_pic()

    if getattr(band_plot, 'band_summary', None):
        summary = band_plot.band_summary
        print(
            f"  [band] {path}: fermi={summary['fermi_energy_eV']:.6f} eV, gap={summary['band_gap_eV']:.6f} eV, type={summary['band_type']}",
            flush=True,
        )


def postprocess_inference_outputs(save_dir, nspin=4):
    print('STEP3:开始输出', flush=True)
    for path in save_dir:
        if path:
            try:
                _postprocess_single_output(path, nspin=nspin, keep_original_band_plot=False)
            except Exception as e:
                print(f'  [pyatb][error] {path}: {e}', flush=True)
                raise

    print('STEP3:文件输出完成', flush=True)
    print('-------------------------------------------------', flush=True)


def run_inference_from_abacus_dirs(read_path, save_path, nspin=4, run_postprocess=True):
    save_dir = prepare_inference_inputs(read_path=read_path, save_path=save_path, nspin=nspin)
    run_inference_from_prepared_dirs(save_dir=save_dir, project_root=Path(__file__).parent)
    if run_postprocess:
        postprocess_inference_outputs(save_dir=save_dir, nspin=nspin)
    return save_dir


def main():
    try:
        from config import usage, nspin, save_path
    except Exception as e:
        raise RuntimeError(
            "基础配置读取失败：请在同目录下创建 config.py, 并至少定义 usage, nspin, save_path 变量。"
        ) from e

    try:
        if usage == "inference":
            from config import read_path
            train_weak_path = []
            train_fine_path = []
            val_weak_path = []
            val_fine_path = []
        elif usage == "train":
            from config import train_weak_path, train_fine_path, val_weak_path, val_fine_path
            read_path = []
        else:
            raise ValueError(f"未知的 usage 值：{usage}，应为 'train' 或 'inference'")
    except Exception as e:
        raise RuntimeError(
            "路径配置读取失败：请在同目录下创建 paths_config.py，并根据 usage 定义相应的路径列表。"
        ) from e

    if usage == "inference":
        save_dir = prepare_inference_inputs(read_path=read_path, save_path=save_path, nspin=nspin)
        print('STEP2:开始网络推理', flush=True)
        run_inference_from_prepared_dirs(save_dir=save_dir, project_root=Path(__file__).parent)
        print('STEP3:开始输出', flush=True)
        for path in save_dir:
            if path:
                try:
                    _postprocess_single_output(path, nspin=nspin, keep_original_band_plot=True)
                except Exception as e:
                    print(f'  [pyatb][error] {path}: {e}', flush=True)
                    raise
        print('STEP3:文件输出完成', flush=True)
        print('-------------------------------------------------', flush=True)
        return

    print('--' * 80, flush=True)
    print('STEP1:开始创建训练集文件', flush=True)
    train_path_stru = []
    train_path_weak = []
    train_path_fine = []
    train_path_overlap = []
    train_path_runinglog = []
    if len(train_weak_path) != len(train_fine_path):
        print("训练集合使用数据路径 weak_path 和 fine_path 数目不一致！", flush='True')
    else:
        for ii, jj in zip(train_weak_path, train_fine_path):
            train_path_stru.append(os.path.join(ii, 'OUT.ABACUS/STRU_READIN_ADJUST.cif'))
            train_path_weak.append(os.path.join(ii, 'OUT.ABACUS/data-HR-sparse_SPIN0.csr'))
            train_path_overlap.append(os.path.join(ii, 'OUT.ABACUS/data-SR-sparse_SPIN0.csr'))
            train_path_fine.append(os.path.join(jj, 'OUT.ABACUS/data-HR-sparse_SPIN0.csr'))
            train_path_runinglog.append(os.path.join(jj, 'OUT.ABACUS/running_scf.log'))

    val_path_stru = []
    val_path_weak = []
    val_path_fine = []
    val_path_overlap = []
    val_path_runinglog = []
    if len(val_weak_path) != len(val_fine_path):
        print("验证集合使用数据路径 weak_path 和 fine_path 数目不一致！", flush='True')
    else:
        for ii, jj in zip(val_weak_path, val_fine_path):
            val_path_stru.append(os.path.join(ii, 'OUT.ABACUS/STRU_READIN_ADJUST.cif'))
            val_path_weak.append(os.path.join(ii, 'OUT.ABACUS/data-HR-sparse_SPIN0.csr'))
            val_path_overlap.append(os.path.join(ii, 'OUT.ABACUS/data-SR-sparse_SPIN0.csr'))
            val_path_fine.append(os.path.join(jj, 'OUT.ABACUS/data-HR-sparse_SPIN0.csr'))
            val_path_runinglog.append(os.path.join(jj, 'OUT.ABACUS/running_scf.log'))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_save_path = os.path.join(save_path, 'train')
    val_save_path = os.path.join(save_path, 'val')
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    if not os.path.exists(val_save_path):
        os.makedirs(val_save_path)
    print(f'训练集地址创建完成  -------->  {train_save_path}')
    print(f'验证集地址创建完成  -------->  {val_save_path}')
    print('--' * 40, flush=True)

    if not (len(train_path_stru) == len(train_path_weak) == len(train_path_overlap) == len(train_path_fine)):
        print("Error: training data path_stru, path_weak, path_fine and path_overlap have inconsistent lengths!", flush='True')
        sys.exit(1)
    else:
        train_save_dir = []
        for ii in range(len(train_path_stru)):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subdir_name = f"data_{ii}_{timestamp}"
            subdir = os.path.join(train_save_path, subdir_name)
            os.mkdir(subdir)
            src_stru = train_path_stru[ii]
            atoms = read(src_stru, format='cif')
            src_weak = train_path_weak[ii]
            src_overlap = train_path_overlap[ii]
            src_fine = train_path_fine[ii]
            running_log = train_path_runinglog[ii]
            dst_stru = os.path.join(subdir, "STRU")
            dst_weak = os.path.join(subdir, "data-HR-sparse_SPIN0_weak.csr")
            dst_overlap = os.path.join(subdir, "data-SR-sparse_SPIN0.csr")
            dst_fine = os.path.join(subdir, "data-HR-sparse_SPIN0_fine.csr")
            dst_fermi = os.path.join(subdir, "fermi_energy.txt")
            if not os.path.isfile(src_stru) or not os.path.isfile(src_weak) or not os.path.isfile(src_overlap) or not os.path.isfile(src_fine):
                print(f"Skip copying : source file not complect for {subdir_name}", flush=True)
                train_save_dir.append(None)
            else:
                write(dst_stru, atoms, format='abacus', scaled=False)
                pattern = re.compile(r'EFERMI\s*=\s*([+-]?[0-9\.Ee-]+)')
                with open(running_log, 'r') as f:
                    for line in f:
                        if 'EFERMI' in line:
                            match = pattern.search(line)
                            if match:
                                fermi_energy = match.group(1)
                                with open(dst_fermi, 'w') as f_out:
                                    f_out.write(fermi_energy)
                shutil.copy2(src_weak, dst_weak)
                shutil.copy2(src_overlap, dst_overlap)
                shutil.copy2(src_fine, dst_fine)
                train_save_dir.append(subdir)

    if not len(val_path_stru) == len(val_path_weak) == len(val_path_overlap) == len(val_path_fine):
        print("Error: valation data path_stru, path_weak, path_fine and path_overlap have inconsistent lengths!", flush='True')
        sys.exit(1)
    else:
        val_save_dir = []
        for ii in range(len(val_path_stru)):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subdir_name = f"data_{ii}_{timestamp}"
            subdir = os.path.join(val_save_path, subdir_name)
            os.mkdir(subdir)
            src_stru = val_path_stru[ii]
            atoms = read(src_stru, format='cif')
            src_weak = val_path_weak[ii]
            src_overlap = val_path_overlap[ii]
            src_fine = val_path_fine[ii]
            running_log = val_path_runinglog[ii]
            dst_stru = os.path.join(subdir, "STRU")
            dst_weak = os.path.join(subdir, "data-HR-sparse_SPIN0_weak.csr")
            dst_overlap = os.path.join(subdir, "data-SR-sparse_SPIN0.csr")
            dst_fine = os.path.join(subdir, "data-HR-sparse_SPIN0_fine.csr")
            dst_fermi = os.path.join(subdir, "fermi_energy.txt")
            if not os.path.isfile(src_stru) or not os.path.isfile(src_weak) or not os.path.isfile(src_overlap) or not os.path.isfile(src_fine):
                print(f"Skip copying : source file not complect for {subdir_name}", flush=True)
                val_save_dir.append(None)
            else:
                write(dst_stru, atoms, format='abacus', scaled=False)
                pattern = re.compile(r'EFERMI\s*=\s*([+-]?[0-9\.Ee-]+)')
                with open(running_log, 'r') as f:
                    for line in f:
                        if 'EFERMI' in line:
                            match = pattern.search(line)
                            if match:
                                fermi_energy = match.group(1)
                                with open(dst_fermi, 'w') as f_out:
                                    f_out.write(fermi_energy)
                shutil.copy2(src_weak, dst_weak)
                shutil.copy2(src_overlap, dst_overlap)
                shutil.copy2(src_fine, dst_fine)
                val_save_dir.append(subdir)

    for path in train_save_dir:
        if path:
            stru_file = os.path.join(path, "STRU")
            weak_file = os.path.join(path, "data-HR-sparse_SPIN0_weak.csr")
            overlap_file = os.path.join(path, "data-SR-sparse_SPIN0.csr")
            label_file = os.path.join(path, "data-HR-sparse_SPIN0_fine.csr")
            fermi_file = os.path.join(path, "fermi_energy.txt")
            read_hs = get_hamiltion_data(usage, nspin, path, stru_file, weak_file, overlap_file, label_file)
            read_hs.read_stru()
            read_hs.read_data()
            print(f'{path} 波函数计算中', flush=True)
            fermi_energy = np.loadtxt(fermi_file)
            ecut = 10
            mp_grid = np.array([4, 4, 4])
            wf_data = output_wfc(nspin, stru_file, label_file, overlap_file, ecut, fermi_energy, mp_grid, path)
            wf_data.output_wfc()

    print('训练集数据创建完成', flush=True)
    print('--' * 40, flush=True)

    for path in val_save_dir:
        if path:
            stru_file = os.path.join(path, "STRU")
            weak_file = os.path.join(path, "data-HR-sparse_SPIN0_weak.csr")
            overlap_file = os.path.join(path, "data-SR-sparse_SPIN0.csr")
            label_file = os.path.join(path, "data-HR-sparse_SPIN0_fine.csr")
            fermi_file = os.path.join(path, "fermi_energy.txt")
            read_hs = get_hamiltion_data(usage, nspin, path, stru_file, weak_file, overlap_file, label_file)
            read_hs.read_stru()
            read_hs.read_data()
            print(f'{path} 波函数计算中', flush=True)
            fermi_energy = np.loadtxt(fermi_file)
            ecut = 10
            mp_grid = np.array([4, 4, 4])
            wf_data = output_wfc(nspin, stru_file, label_file, overlap_file, ecut, fermi_energy, mp_grid, path)
            wf_data.output_wfc()

    print('验证集数据创建完成', flush=True)
    print('STEP1:训练文件全部读入完成', flush=True)
    print('--' * 80, flush=True)

    print('STEP2:开始网络训练', flush=True)
    data_set_path = './datasets'
    train_file = os.path.join(data_set_path, 'train.txt')
    train_wf_file = os.path.join(data_set_path, 'train_wf.txt')
    val_file = os.path.join(data_set_path, 'val.txt')
    val_wf_file = os.path.join(data_set_path, 'val_wf.txt')
    train_input_path = []
    train_wf_path = []
    val_input_path = []
    val_wf_path = []
    for path in train_save_dir:
        if not path:
            continue
        base = Path(path)
        train_input_path.append(str(base / 'input_inference.pth'))
        train_wf_path.append(str(base / 'wfc.pth'))
    for path in val_save_dir:
        if not path:
            continue
        base = Path(path)
        val_input_path.append(str(base / 'input_inference.pth'))
        val_wf_path.append(str(base / 'wfc.pth'))

    for f in (train_file, train_wf_file, val_file, val_wf_file):
        if os.path.exists(f):
            os.remove(f)

    with open(train_file, 'w', encoding='utf-8') as f_in:
        f_in.write("\n".join(train_input_path) + ("\n" if train_input_path else ""))
    with open(train_wf_file, 'w', encoding='utf-8') as f_wf:
        f_wf.write("\n".join(train_wf_path) + ("\n" if train_wf_path else ""))
    with open(val_file, 'w', encoding='utf-8') as f_in:
        f_in.write("\n".join(val_input_path) + ("\n" if val_input_path else ""))
    with open(val_wf_file, 'w', encoding='utf-8') as f_wf:
        f_wf.write("\n".join(val_wf_path) + ("\n" if val_wf_path else ""))

    script = Path(__file__).parent / 'scripts/train_val/train_val.sh'
    env = dict(os.environ)
    env['MKL_THREADING_LAYER'] = 'GNU'
    env.pop('LD_PRELOAD', None)
    subprocess.run(['/bin/bash', str(script)], cwd=Path(__file__).parent, env=env, check=True)
    print('STEP2:网络训练完成', flush=True)
    print('--' * 80, flush=True)


if __name__ == '__main__':
    main()
