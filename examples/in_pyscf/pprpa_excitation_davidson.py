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

# 1. simple ppRPA calculation
pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# 2. simple ppTDA calculation
# pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq, TDA="pp")
# pprpa.kernel("s")
# pprpa.kernel("t")
# pprpa.analyze()

# 3. only run singlet/triplet calculation
pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq)
pprpa.kernel("s")
#pprpa.kernel("t")
pprpa.analyze()

# 4. full control parameters
pprpa = ppRPA_Davidson(nocc, mo_energy, Lpq, nroot=15, max_vec=300,
                       max_iter=100, residue_thresh=1.0e-8, print_thresh=0.2)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()
