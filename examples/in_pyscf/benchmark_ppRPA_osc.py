import numpy

from pyscf import gto, scf
from pyscf.tdscf.rhf import _charge_center

from lib_pprpa.pprpa_direct import ppRPA_direct
from lib_pprpa.pprpa_davidson import ppRPA_Davidson
from lib_pprpa.pyscf_util import get_pyscf_input_mol

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["H", (0,0,0)],
    ["H", (.741,0,0)],
]
mol.basis = "631++g**"
mol.charge = +2  # start from the N-2 electron system
mol.build()

mf = scf.RHF(mol)
mf.kernel()

mo = numpy.asarray(mf.mo_coeff, order="F")
# Same formulation as in the TDDFT module
with mol.with_common_orig(_charge_center(mol)):
    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
# Convert AO dipole moment to MO dipole moment
mo_dip = mo.T @ ao_dip @ mo

nocc, mo_energy, Lpq = get_pyscf_input_mol(mf)
pprpa = ppRPA_direct(
    nocc, mo_energy, Lpq, mo_dip=mo_dip)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

mol2 = gto.Mole()
mol2.verbose = 0
mol2.atom = [
    ["H", (0,0,0)],
    ["H", (.741,0,0)],
]
mol2.basis = "631++g**"
mol2.build()

mf2 = scf.RHF(mol2)
mf2.kernel()

mytd = mf2.TDA()
mytd.singlet = True
mytd.nstates = 8
mytd.kernel()
mytd.analyze(verbose=3)