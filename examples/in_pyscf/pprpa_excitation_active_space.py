import numpy

from pyscf import df, gto, scf
from pyscf.ao2mo import _ao2mo

from lib_pprpa.pprpa_direct import ppRPA_direct

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O",  (0.00000000,  -0.00000000,  -0.00614048)],
    ["H",  (0.76443318,  -0.00000000,  0.58917024)],
    ["H",  (-0.76443318,  0.00000000,  0.58917024)],
]
mol.basis = "def2svp"
mol.charge = 2  # start from the N-2 electron system
mol.build()

mf = scf.RHF(mol)
mf.kernel()

nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, with_dip=True, nocc_act=nocc_act, nvir_act=nvir_act)
pprpa = ppRPA_direct(nocc, mf.mo_energy, Lpq, nocc_act=nocc_act, nvir_act=nvir_act)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze()

# 2. get density-fitting matrix in active MO space
mo = numpy.asarray(mf.mo_coeff, order='F')
ijslice = (nocc_fro, nmo-nvir_fro, nocc_fro, nmo-nvir_fro)
Lpq = None
Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym='s2', out=Lpq)
Lpq = Lpq.reshape(naux, nmo_act, nmo_act)

pprpa = ppRPA_direct(nocc_act, mf.mo_energy[nocc_fro:(nmo-nvir_fro)], Lpq)
pprpa.kernel("s")
pprpa.kernel("t")
pprpa.analyze(nocc_fro=nocc_fro)  # manually assign the index of the first active occupied orbital
