"""
Benchmark script for _pprpa_contraction comparing optimized vs reference implementation.
"""

from benchmark_utils import BenchmarkRunner, setup_pprpa, get_benzene_mol
from lib_pprpa import pprpa_davidson
import references as ref_implementations


def benchmark_pprpa_contraction():
    # Setup test case (Benzene/cc-pVDZ)
    mol = get_benzene_mol(basis='cc-pvdz')
    _, pp, _ = setup_pprpa(mol, use_df=True, mult='s', nroot=5)
    
    # Generate random trial vectors for testing
    ntri = min(pp.nroot * 4, pp.vv_dim)
    tri_vec, _ = pprpa_davidson.get_identity_trial_vector(pprpa=pp, ntri=ntri)
    
    runner = BenchmarkRunner()
    results = runner.run(
        opt_func=pprpa_davidson._pprpa_contraction,
        ref_func=ref_implementations._pprpa_contraction_ref,
        args=(pp, tri_vec),
        n_iterations=10,
        func_name="_pprpa_contraction"
    )

    print(f"| Benzene/cc-pVDZ ({ntri} vecs) | {results.reference_avg:.3f}s | {results.optimized_avg:.3f}s | {results.speedup:.2f}x |")


if __name__ == "__main__":
    benchmark_pprpa_contraction()