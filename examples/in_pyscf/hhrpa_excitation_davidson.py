from pyscf import gto, scf

from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O",  (0.00000000,  -0.00000000,  -0.00614048)],
    ["H",  (0.76443318,  -0.00000000,  0.58917024)],
    ["H",  (-0.76443318,  0.00000000,  0.58917024)],
]
mol.basis = "def2svp"
mol.build()

mf = scf.RHF(mol)
mf.kernel()

nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True)

# 1. hhRPA
pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq, channel="hh")
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# # 2. hhTDA
# pprpa = ppRPA_Davidson(nocc, mf.mo_energy, Lpq, channel="hh", TDA="hh")
# pprpa.kernel("s")
# pprpa.kernel("t")
# pprpa.analyze()
