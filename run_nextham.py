import os
import subprocess
import shutil
from pathlib import Path
from ase.io import write

WORKFLOW_ROOT = Path(__file__).resolve().parent
NEXTHAM_ROOT = WORKFLOW_ROOT / "NextHAM-fix1"

def run_abacus(relaxed_atoms, run_dir, nspin=4):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制模板文件到 run_dir
    template_dir = Path("/data/xypan/dataset/run_data/example")
    for item in template_dir.iterdir():
        if item.is_file():
            shutil.copy(str(item), str(run_dir / item.name))
        else:
            shutil.copytree(str(item), str(run_dir / item.name), dirs_exist_ok=True)
    
    pp = {'C': 'C.upf', 'N': 'N.upf', 'H': 'H.upf', 'Cs': 'Cs.upf', 'Sn': 'Sn.upf', 'Pb': 'Pb.upf', 'Br': 'Br.upf', 'I': 'I.upf', 'Cl': 'Cl.upf'}
    basis = {'C': 'C_gga_7au_100Ry_2s2p1d.orb', 'N': 'N_gga_7au_100Ry_2s2p1d.orb', 'H': 'H_gga_7au_100Ry_2s1p.orb', 'Cs': 'Cs_gga_8au_100Ry_4s2p1d.orb', 'Pb': 'Pb_gga_7au_100Ry_2s2p2d1f.orb', 'Sn': 'Sn_gga_7au_100Ry_2s2p2d1f.orb', 'Br': 'Br_gga_8au_100Ry_2s2p1d.orb', 'I': 'I_gga_7au_100Ry_2s2p1d.orb', 'Cl': 'Cl_gga_7au_100Ry_2s2p1d.orb'}
    write(str(run_dir / 'STRU'), relaxed_atoms, format='abacus', pp=pp, basis=basis)
    
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '1'
    abacus_path = '/data/xypan/Software/abacus-develop-largescale/build/abacus'
    
    subprocess.run([abacus_path], cwd=str(run_dir), env=env, check=True)
    
    return run_dir

def run_nextham_from_relaxed_atoms(relaxed_atoms, run_name = "test", nspin=4):
    run_dir = WORKFLOW_ROOT / run_name
    
    if run_dir.exists():
        shutil.rmtree(str(run_dir))
    
    run_abacus(relaxed_atoms, str(run_dir), nspin)
    
    import sys
    if str(NEXTHAM_ROOT) not in sys.path:
        sys.path.insert(0, str(NEXTHAM_ROOT))
    from test_main import run_inference_from_abacus_dirs
    save_dir = run_inference_from_abacus_dirs([str(run_dir)], str(run_dir / 'out'), nspin=nspin)
    
    print(f"Output dir: {str(run_dir / 'out')}")
    return str(run_dir / 'out')

if __name__ == "__main__":
    import argparse
    from ase.io import read
    
    parser = argparse.ArgumentParser(description='Run ABACUS and NextHAM')
    parser.add_argument('stru_path', type=str, help='Path to input STRU file')
    parser.add_argument('run_name', type=str, help='Run name')
    args = parser.parse_args()
    
    atoms = read(args.stru_path, format='abacus')
    run_nextham_from_relaxed_atoms(atoms, args.run_name)
