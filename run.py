from ase.io import read
from run_nextham import run_nextham_from_relaxed_atoms
from relax import relax_atoms

atoms = read("/data/xypan/workflow/Cs0.17FA0.33MA0.50Sn0.50Pb0.50I1.0Cl0.0Br2.0_1.cif")

#输入原子结构，输出优化后的原子结构
relaxed_atoms = relax_atoms(atoms)

# relaxed_atoms = read("STRU_nn", format='abacus')

#输入原子结构，在run_name中填写临时工作目录，输出NextHAM的结果文件路径
out = run_nextham_from_relaxed_atoms(relaxed_atoms, run_name="test", nspin=4)

print(f"NextHAM output saved in: {out}")
