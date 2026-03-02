"""
Benchmark script comparing optimized vs reference implementations of make_rdm1_relaxed_rhf_pprpa.
"""

import numpy as np
from unittest.mock import patch

from benchmark_utils import BenchmarkRunner, setup_pprpa, get_benzene_mol
from lib_pprpa.grad import pprpa as pprpa_grad_mod
import references as ref_implementations


def _run_with_stubbed_external_calls(func, pp, mf, xy, mult='s'):
    """Run make_rdm1 while patching heavy PySCF-dependent kernels.

    This isolates I/D intermediate construction by replacing response-kernel
    calls and CPHF solve with deterministic lightweight stand-ins.
    """
    nmo = len(mf.mo_energy)
    nao = mf.mo_coeff.shape[0]

    def fake_gen_response(*_args, **_kwargs):
        def fake_vresp(dm_ao):
            # Return a deterministic AO matrix with the same shape.
            # Scale preserves rough magnitude while staying lightweight.
            return 0.1 * dm_ao

        return fake_vresp

    def fake_cphf_solve(_fvind, _mo_ene, _mo_occ, x_int, max_cycle=20, tol=1.0e-8):
        # Keep a deterministic output shape expected by caller.
        return x_int.ravel() * 0.0, None

    def fake_get_k(dm=None, hermi=1, **_kwargs):
        if dm is None:
            return np.zeros((nao, nao), dtype=mf.mo_coeff.dtype)
        return 0.05 * dm

    with patch.object(mf, 'gen_response', fake_gen_response), \
         patch.object(mf, 'get_k', fake_get_k), \
         patch('pyscf.scf.cphf.solve', fake_cphf_solve):
        return func(pp, mf, xy=xy, mult=mult)


def benchmark_make_rdm1_intermediates_only():
    """Benchmark only I/D intermediate construction (external calls stubbed)."""
    mol = get_benzene_mol(basis='cc-pvtz')
    mf, pp, xy = setup_pprpa(mol, use_df=True, mult='s')

    runner = BenchmarkRunner()
    results = runner.run(
        opt_func=lambda p, m: _run_with_stubbed_external_calls(
            pprpa_grad_mod.make_rdm1_relaxed_rhf_pprpa, p, m, xy, mult='s'
        ),
        ref_func=lambda p, m: _run_with_stubbed_external_calls(
            ref_implementations.make_rdm1_relaxed_rhf_pprpa_ref, p, m, xy, mult='s'
        ),
        args=(pp, mf),
        kwargs={},
        n_iterations=20,
        func_name="make_rdm1_relaxed_rhf_pprpa (I/D-only, stubbed external calls)",
    )

    print(
        f"| Benzene/cc-pVTZ I/D-only | {results.reference_avg:.6f}s | "
        f"{results.optimized_avg:.6f}s | {results.speedup:.2f}x |"
    )


def benchmark_make_rdm1():
    # Setup test case (Benzene/cc-pVTZ)
    mol = get_benzene_mol(basis='cc-pvtz')
    mf, pp, xy = setup_pprpa(mol, use_df=True, mult='s')
    
    runner = BenchmarkRunner()
    results = runner.run(
        opt_func=pprpa_grad_mod.make_rdm1_relaxed_rhf_pprpa,
        ref_func=ref_implementations.make_rdm1_relaxed_rhf_pprpa_ref,
        args=(pp, mf),
        kwargs={'xy': xy, 'mult': 's'},
        n_iterations=5,
        func_name="make_rdm1_relaxed_rhf_pprpa"
    )

    print(f"| Benzene/cc-pVTZ | {results.reference_avg:.3f}s | {results.optimized_avg:.3f}s | {results.speedup:.2f}x |")


if __name__ == "__main__":
    benchmark_make_rdm1_intermediates_only()
    benchmark_make_rdm1()