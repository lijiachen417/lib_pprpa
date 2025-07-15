from pyscf import gto, scf

from lib_pprpa.pprpa_direct import ppRPA_direct
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
pprpa = ppRPA_direct(nocc, mo_energy, Lpq)
# get_correlation() will run singlet and triplet calculations internally
ec = pprpa.get_correlation()
print("ppRPA correlation energy = %.8f" % ec)
