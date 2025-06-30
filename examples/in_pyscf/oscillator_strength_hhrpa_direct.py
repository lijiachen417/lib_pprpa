import numpy

from pyscf import df, gto, scf
from pyscf.ao2mo import _ao2mo
from pyscf.tdscf.rhf import _charge_center

from lib_pprpa.pprpa_direct import ppRPA_direct
from lib_pprpa.pprpa_davidson import ppRPA_Davidson

mol = gto.Mole()
mol.verbose = 0
mol.atom = [
    ["O", (0.00000000, -0.00000000, -0.00614048)],
    ["H", (0.76443318, -0.00000000, 0.58917024)],
    ["H", (-0.76443318, 0.00000000, 0.58917024)],
]
mol.basis = "def2svp"
mol.charge = -2  # start from the N+2 electron system
mol.build()

mf = scf.RHF(mol)
mf.kernel()

# get density-fitting matrix in AO
if getattr(mf, "with_df", None):
    pass
else:
    mf.with_df = df.DF(mf.mol)
    try:
        mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=True)
    except:
        mf.with_df.auxbasis = df.make_auxbasis(mf.mol, mp2fit=False)
    mf._keys.update(["with_df"])

# get density-fitting matrix in MO space
nmo = len(mf.mo_energy)
nocc = mf.mol.nelectron // 2
nvir = nmo - nocc
naux = mf.with_df.get_naoaux()
mo = numpy.asarray(mf.mo_coeff, order="F")
ijslice = (0, nmo, 0, nmo)
Lpq = None
Lpq = _ao2mo.nr_e2(mf.with_df._cderi, mo, ijslice, aosym="s2", out=Lpq)
Lpq = Lpq.reshape(naux, nmo, nmo)

# Same formulation as in the TDDFT module
with mol.with_common_orig(_charge_center(mol)):
    ao_dip = mol.intor_symmetric("int1e_r", comp=3)
# Convert AO dipole moment to MO dipole moment
mo_dip = mo.T @ ao_dip @ mo

pp_RPA_functions = [
    ppRPA_Davidson,
    ppRPA_direct,
]

for ppRPA in pp_RPA_functions:
    print(f"Testing {ppRPA.__name__}...")
    try:
        pprpa = ppRPA(
            nocc, mf.mo_energy, Lpq, mo_dip=mo_dip, osc_channel="hh", pp_state=0, nelec="n+2"
        )
        # pprpa.kernel("ab")
        # pprpa.analyze_ab()
    except:
        pprpa = ppRPA(nocc, mf.mo_energy, Lpq, mo_dip=mo_dip, channel="hh")
    pprpa.kernel("s")
    pprpa.kernel("t")
    pprpa.analyze()
