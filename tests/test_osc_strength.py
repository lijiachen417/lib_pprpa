import numpy, sys, os
from io import StringIO
from pyscf import gto, scf
from lib_pprpa.pyscf_util import get_pyscf_input_mol
from lib_pprpa.pprpa_direct import ppRPA_direct
from lib_pprpa.pprpa_davidson import ppRPA_Davidson


def test_direct_Davidson_osc_agreement():
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ["O", (0.00000000, -0.00000000, -0.00614048)],
        ["H", (0.76443318, -0.00000000, 0.58917024)],
        ["H", (-0.76443318, 0.00000000, 0.58917024)],
    ]
    mol.basis = "def2svp"
    mol.charge = 2  # start from the N-2 electron system
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    pp_RPA_functions = [ppRPA_Davidson, ppRPA_direct]
    osc_lists = [[], []]
    nocc, mo_energy, Lpq, mo_dip = get_pyscf_input_mol(mf, calc_dip=True)
    for ppRPA, lst in zip(pp_RPA_functions, osc_lists):
        try:
            pprpa = ppRPA( # direct function sig
                nocc, mo_energy, Lpq, mo_dip=mo_dip, osc_channel="pp", hh_state=0
            )
        except: # davidson function sig
            pprpa = ppRPA(nocc, mo_energy, Lpq, mo_dip=mo_dip)
        pprpa.kernel("s")
        pprpa.kernel("t")
        tmp = sys.stdout
        sys.stdout = StringIO()
        pprpa.analyze()
        output = sys.stdout.getvalue()
        sys.stdout = tmp
        lines = output.splitlines()
        oscs = [float(x) for line in lines if "oscillator strength" in line \
                for x in line.split() if x.replace('.', '', 1).isnumeric()]
        lst.extend(oscs)


    oscs_array = numpy.array(osc_lists)
    assert numpy.allclose(oscs_array[0], oscs_array[1]), \
        "Oscillator strengths do not match between ppRPA_Davidson and ppRPA_direct."
