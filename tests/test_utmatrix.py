from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import dft, gto, scf

import lib_pprpa.tmatrix as restricted_tmatrix
import lib_pprpa.utmatrix as unrestricted_tmatrix
from lib_pprpa.tmatrix import (
    TMatrix,
    get_sigma_derivative,
)
from lib_pprpa.utmatrix import (
    UTMatrix,
    _get_sigma_derivative_for_spin,
    _get_sigma_for_spin,
    get_transition_density_diff_spin,
    get_transition_density_same_spin,
)


def _symmetric_df_tensor(rng, naux, nmo):
    tensor = rng.normal(size=(naux, nmo, nmo))
    return 0.5 * (tensor + tensor.transpose(0, 2, 1))


def test_transition_densities_match_explicit_eri_contractions():
    rng = np.random.default_rng(12)

    nocc = 2
    nvir = 2
    nmo = nocc + nvir
    Lpq = _symmetric_df_tensor(rng, naux=3, nmo=nmo)
    oo_dim = nocc * (nocc - 1) // 2
    vv_dim = nvir * (nvir - 1) // 2
    xy = rng.normal(size=(3, oo_dim + vv_dim))

    rho = get_transition_density_same_spin(nocc, nvir, xy, Lpq)
    eri = np.einsum('Ppr,Pqs->pqrs', Lpq, Lpq, optimize=True)
    expected = np.zeros_like(rho)
    tri_o = np.tril_indices(nocc, -1)
    tri_v = np.tril_indices(nvir, -1)
    for m in range(len(xy)):
        for pair, (i, j) in enumerate(zip(*tri_o)):
            expected[m] += (
                eri[:, :, i, j] - eri[:, :, j, i]) * xy[m, pair]
        for pair, (a, b) in enumerate(zip(*tri_v)):
            a += nocc
            b += nocc
            expected[m] += (
                eri[:, :, a, b] - eri[:, :, b, a]) \
                * xy[m, oo_dim + pair]
    assert np.allclose(rho, expected, atol=1.0e-12)

    nocc_u = (2, 1)
    nvir_u = (2, 3)
    nmo_u = tuple(nocc_u[s] + nvir_u[s] for s in range(2))
    Lpq_u = [
        _symmetric_df_tensor(rng, naux=4, nmo=nmo_u[0]),
        _symmetric_df_tensor(rng, naux=4, nmo=nmo_u[1]),
    ]
    oo_dim_u = nocc_u[0] * nocc_u[1]
    xy_u = rng.normal(
        size=(4, oo_dim_u + nvir_u[0] * nvir_u[1]))

    rho_u = get_transition_density_diff_spin(
        nocc_u, nvir_u, xy_u, Lpq_u)
    eri_ab = np.einsum(
        'Ppr,Pqs->pqrs', Lpq_u[0], Lpq_u[1], optimize=True)
    Y = xy_u[:, :oo_dim_u].reshape(-1, nocc_u[0], nocc_u[1])
    X = xy_u[:, oo_dim_u:].reshape(-1, nvir_u[0], nvir_u[1])
    expected_u = np.einsum(
        'pqij,mij->mpq',
        eri_ab[:, :, :nocc_u[0], :nocc_u[1]],
        Y,
        optimize=True,
    )
    expected_u += np.einsum(
        'pqab,mab->mpq',
        eri_ab[:, :, nocc_u[0]:, nocc_u[1]:],
        X,
        optimize=True,
    )
    assert np.allclose(rho_u, expected_u, atol=1.0e-12)


def test_diff_spin_transition_density_reuses_blas_buffers(monkeypatch):
    rng = np.random.default_rng(19)
    nocc = (2, 1)
    nvir = (2, 2)
    nmo = tuple(nocc[s] + nvir[s] for s in range(2))
    nroot = 3
    Lpq = [
        _symmetric_df_tensor(rng, naux=3, nmo=nmo[0]),
        _symmetric_df_tensor(rng, naux=3, nmo=nmo[1]),
    ]
    xy = rng.normal(
        size=(nroot, nocc[0] * nocc[1] + nvir[0] * nvir[1]))

    original_dgemm = unrestricted_tmatrix.scipy.linalg.blas.dgemm
    output_pointers = []

    def record_dgemm(*args, **kwargs):
        output_pointers.append(
            kwargs['c'].__array_interface__['data'][0])
        return original_dgemm(*args, **kwargs)

    monkeypatch.setattr(
        unrestricted_tmatrix.scipy.linalg.blas, 'dgemm', record_dgemm)
    get_transition_density_diff_spin(nocc, nvir, xy, Lpq)

    assert len(output_pointers) == 4 * nroot
    assert len(set(output_pointers[0::4])) == 1  # occupied work buffer
    assert len(set(output_pointers[2::4])) == 1  # virtual work buffer
    assert output_pointers[1::4] == output_pointers[3::4]  # rho in-place
    assert unrestricted_tmatrix.get_sigma is restricted_tmatrix.get_sigma


def _direct_channel(nocc, mo_energy, mo_energy_ref, exci, rho, oo_dim,
                    mu, eta):
    eta2 = (3.0 * eta) ** 2
    sigma = np.zeros(len(mo_energy))
    derivative = np.zeros(len(mo_energy))
    for p, omega in enumerate(mo_energy):
        for m in range(oo_dim):
            for q in range(nocc, len(mo_energy_ref)):
                ediff = omega + mo_energy_ref[q] - 2.0 * mu - exci[m]
                numerator = rho[m, p, q] ** 2
                sigma[p] += numerator * ediff / (ediff ** 2 + eta2)
                derivative[p] += (
                    numerator * (eta2 - ediff ** 2)
                    / (ediff ** 2 + eta2) ** 2)
        for m in range(oo_dim, len(exci)):
            for q in range(nocc):
                ediff = omega + mo_energy_ref[q] - 2.0 * mu - exci[m]
                numerator = rho[m, p, q] ** 2
                sigma[p] += numerator * ediff / (ediff ** 2 + eta2)
                derivative[p] += (
                    numerator * (eta2 - ediff ** 2)
                    / (ediff ** 2 + eta2) ** 2)
    return sigma, derivative


def test_spin_self_energy_uses_correct_internal_channel_and_transpose():
    rng = np.random.default_rng(41)
    nocc = (2, 1)
    nmo = 4
    mo_energy_ref = np.asarray([
        [-1.3, -0.4, 0.2, 0.8],
        [-1.1, 0.1, 0.5, 1.0],
    ])
    oo_aa = nocc[0] * (nocc[0] - 1) // 2
    oo_bb = nocc[1] * (nocc[1] - 1) // 2
    oo_ab = nocc[0] * nocc[1]
    nroot_aa = oo_aa + 1
    nroot_bb = oo_bb + 3
    nroot_ab = oo_ab + 6

    tm = SimpleNamespace(
        nocc=nocc,
        _scf=SimpleNamespace(mo_energy=mo_energy_ref),
        exci_aa=np.linspace(-0.7, 0.9, nroot_aa),
        exci_bb=np.linspace(0.2, 1.4, nroot_bb),
        exci_ab=np.linspace(-0.9, 1.6, nroot_ab),
        rho_aa=rng.normal(size=(nroot_aa, nmo, nmo)),
        rho_bb=rng.normal(size=(nroot_bb, nmo, nmo)),
        rho_ab=rng.normal(size=(nroot_ab, nmo, nmo)),
        mu=-0.05,
        eta=0.03,
    )
    mo_energy = np.asarray([
        [-1.0, -0.2, 0.3, 0.9],
        [-0.8, 0.0, 0.6, 1.2],
    ])

    for spin in range(2):
        if spin == 0:
            channels = [
                (nocc[0], mo_energy_ref[0], tm.exci_aa,
                 tm.rho_aa, oo_aa),
                (nocc[1], mo_energy_ref[1], tm.exci_ab,
                 tm.rho_ab, oo_ab),
            ]
        else:
            channels = [
                (nocc[1], mo_energy_ref[1], tm.exci_bb,
                 tm.rho_bb, oo_bb),
                (nocc[0], mo_energy_ref[0], tm.exci_ab,
                 tm.rho_ab.transpose(0, 2, 1), oo_ab),
            ]

        expected_sigma = np.zeros(nmo)
        expected_derivative = np.zeros(nmo)
        for channel in channels:
            sigma, derivative = _direct_channel(
                channel[0], mo_energy[spin], channel[1], channel[2],
                channel[3], channel[4], tm.mu, tm.eta)
            expected_sigma += sigma
            expected_derivative += derivative

        sigma = _get_sigma_for_spin(tm, spin, mo_energy[spin]).diagonal()
        derivative = _get_sigma_derivative_for_spin(
            tm, spin, mo_energy[spin])
        assert np.allclose(sigma, expected_sigma, atol=1.0e-12)
        assert np.allclose(
            derivative, expected_derivative, atol=1.0e-12)

        step = 1.0e-6
        finite_difference = (
            _get_sigma_for_spin(tm, spin, mo_energy[spin] + step).diagonal()
            - _get_sigma_for_spin(
                tm, spin, mo_energy[spin] - step).diagonal()
        ) / (2.0 * step)
        assert np.allclose(
            derivative, finite_difference, atol=2.0e-7)


def _build_be_reference():
    mol = gto.M(
        atom='Be 0 0 0',
        basis='sto-3g',
        verbose=0,
    )
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    return mf


def test_closed_shell_reduction_matches_restricted_tmatrix():
    rhf = _build_be_reference()
    uhf = scf.addons.convert_to_uhf(rhf)

    restricted = TMatrix(rhf)
    restricted.eta = 1.0e-5
    restricted.qpe_linearized = True
    restricted.kernel()

    unrestricted = UTMatrix(uhf)
    unrestricted.Lpq = [restricted.Lpq.copy(), restricted.Lpq.copy()]
    unrestricted.eta = restricted.eta
    unrestricted.qpe_linearized = True
    unrestricted.kernel()

    assert np.allclose(
        unrestricted.mo_energy[0], restricted.mo_energy, atol=1.0e-10)
    assert np.allclose(
        unrestricted.mo_energy[1], restricted.mo_energy, atol=1.0e-10)

    nocc = restricted.nocc
    der_s = get_sigma_derivative(
        nocc, np.asarray(rhf.mo_energy), np.asarray(rhf.mo_energy),
        restricted.exci_s, restricted.rho_s,
        nocc * (nocc + 1) // 2, restricted.mu, restricted.eta)
    der_t = get_sigma_derivative(
        nocc, np.asarray(rhf.mo_energy), np.asarray(rhf.mo_energy),
        restricted.exci_t, restricted.rho_t,
        nocc * (nocc - 1) // 2, restricted.mu, restricted.eta)
    expected_derivative = der_s + 3.0 * der_t
    for spin in range(2):
        derivative = _get_sigma_derivative_for_spin(
            unrestricted, spin, np.asarray(uhf.mo_energy[spin]))
        assert np.allclose(
            derivative, expected_derivative, atol=1.0e-10)

    omega = np.asarray([-0.5, 0.0, 0.5])
    gf_r, gf0_r, sigma_r = restricted.make_gf(
        omega, eta=0.01, fullsigma=True, mode='linear')
    gf_u, gf0_u, sigma_u = unrestricted.make_gf(
        omega, eta=0.01, fullsigma=True, mode='linear')
    for spin in range(2):
        assert np.allclose(sigma_u[spin], sigma_r, atol=1.0e-10)
        assert np.allclose(gf0_u[spin], gf0_r, atol=1.0e-10)
        assert np.allclose(gf_u[spin], gf_r, atol=1.0e-10)

    dos_r = restricted.make_diag_dos(omega, eta=0.01)
    dos_u = unrestricted.make_diag_dos(omega, eta=0.01)
    assert np.allclose(dos_u[0], dos_r, atol=1.0e-10)
    assert np.allclose(dos_u[1], dos_r, atol=1.0e-10)

    nw = 8
    e_tot_u, e_hf_u, e_c_u = unrestricted.energy_tot(nw=nw)
    pts, wts = np.polynomial.legendre.leggauss(nw)
    freqs = (1.0 + pts) / (1.0 - pts)
    weights = wts * 2.0 / (1.0 - pts) ** 2
    omega_imag = 1j * freqs + restricted.mu
    _, gf0_r, sigma_r = restricted.make_gf(
        omega_imag, eta=0, fullsigma=True, mode='linear')
    e_c_r = (
        2.0 * np.einsum('wpq,wqp,w->', gf0_r, sigma_r, weights)
        / (2.0 * np.pi)
    ).real
    assert np.isclose(e_c_u, e_c_r, atol=1.0e-10)
    assert np.isclose(e_hf_u, rhf.e_tot, atol=1.0e-10)
    assert np.isclose(e_tot_u, e_hf_u + e_c_u, atol=1.0e-12)


def _build_beh_reference(method):
    mol = gto.M(
        atom='Be 0 0 0; H 0 0 1.342',
        basis='sto-3g',
        spin=1,
        verbose=0,
    )
    if method == 'UHF':
        mf = scf.UHF(mol).density_fit()
    else:
        mf = dft.UKS(mol).density_fit()
        mf.xc = 'pbe'
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.parametrize('method', ['UHF', 'UKS'])
def test_open_shell_feature_paths(method):
    mf = _build_beh_reference(method)
    tm = UTMatrix(mf)
    tm.eta = 1.0e-4
    tm.qpe_linearized = (method == 'UKS')
    tm.vhf_df = (method == 'UKS')
    tm.kernel()

    nmo = len(mf.mo_energy[0])
    assert tm.mo_energy.shape == (2, nmo)
    assert tm.vk.shape == (2, nmo, nmo)
    assert tm.vxc.shape == (2, nmo, nmo)
    assert tm.rho_ab.shape[1:] == (nmo, nmo)
    assert np.all(np.isfinite(tm.mo_energy))
    assert not np.allclose(tm.mo_energy[0], tm.mo_energy[1])

    omega = np.asarray([-0.4, 0.1])
    gf, gf0, sigma = tm.make_gf(
        omega, eta=0.02, fullsigma=True, mode='dyson')
    assert gf.shape == (2, len(omega), nmo, nmo)
    assert gf0.shape == gf.shape
    assert sigma.shape == gf.shape
    assert np.all(np.isfinite(gf))

    dos = tm.make_diag_dos(omega, eta=0.02)
    assert dos.shape == (2, nmo, len(omega))
    assert np.all(np.isfinite(dos))
    assert np.all(dos >= 0.0)

    e_tot, e_hf, e_c = tm.energy_tot(nw=8)
    assert np.all(np.isfinite([e_tot, e_hf, e_c]))
    assert np.isclose(e_tot, e_hf + e_c)

    with pytest.raises(ValueError):
        tm.make_gf(omega, eta=0.02, mode='invalid')
    with pytest.raises(NotImplementedError):
        tm.make_gf(omega, eta=0.02, fullsigma=False)


def test_restricted_reference_is_rejected():
    with pytest.raises(TypeError):
        UTMatrix(_build_be_reference())
