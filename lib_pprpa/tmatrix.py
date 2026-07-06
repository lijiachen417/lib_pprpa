"""
Spin-restricted T-matrix method based on the particle-particle RPA (ppRPA).
This implementation has N^6 scaling, and is accurate for all states.

The T-matrix method is the particle-particle counterpart to the GW approximation:
  - GW uses particle-hole RPA (ring diagrams)
  - T-matrix uses particle-particle RPA (ladder diagrams)

Reference:
    [1] Du Zhang, Neil Qiang Su, and Weitao Yang,
        "Accurate Quasiparticle Spectra from the T-Matrix Self-Energy
         and the Particle-Particle Random Phase Approximation",
        J. Phys. Chem. Lett. 2017, 8, 3223-3227.
    [2] Jiachen Li, Zehua Chen, and Weitao Yang,
        "Renormalized Singles Green's Function in the T-Matrix Approximation
         for Accurate Quasiparticle Energy Calculation",
        J. Phys. Chem. Lett. 2021, 12, 6203-6210.
"""

import numpy as np
import scipy
import time

try:
    import numexpr as ne
except ImportError:
    ne = None

from pyscf import dft, scf, df, lib
from pyscf.ao2mo import _ao2mo

from lib_pprpa.pprpa_direct import diagonalize_pprpa_singlet, diagonalize_pprpa_triplet
from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock


# =============================================================================
# Module-level functions
# =============================================================================

def get_transition_density(multi, nocc, nvir, xy, Lpq):
    r"""Calculate the T-matrix transition density from ppRPA eigenvectors.

    The transition density rho_{m}(p, q) is defined such that the T-matrix
    self-energy can be written in a compact form analogous to GW.
    For the singlet channel, integrals are symmetrized: <pq|rs> + <pq|sr>.
    For the triplet channel, integrals are anti-symmetrized: <pq|rs> - <pq|sr>.

    Parameters
    ----------
    multi : str
        Multiplicity channel, 's' for singlet, 't' for triplet.
    nocc : int
        Number of occupied orbitals.
    nvir : int
        Number of virtual orbitals.
    xy : double 2d array
        ppRPA eigenvectors, shape (nroot, oo_dim + vv_dim).
        Ordered as [hh states, pp states] with the oo block first.
    Lpq : double 3d array
        Three-center density-fitting matrix in MO space, shape (naux, nmo, nmo).

    Returns
    -------
    rho : double 3d array
        Transition density, shape (nroot, nmo, nmo).
        rho[m, p, q] is the transition density for eigenstate m.
    """
    nmo = nocc + nvir
    naux = Lpq.shape[0]
    nroot = xy.shape[0]

    is_singlet = (multi == 's')
    pm = 1.0 if is_singlet else -1.0  # +1 for singlet, -1 for triplet

    if is_singlet:
        oo_dim = (nocc + 1) * nocc // 2
        tri_row_o, tri_col_o = np.tril_indices(nocc)
        tri_row_v, tri_col_v = np.tril_indices(nvir)
    else:
        oo_dim = (nocc - 1) * nocc // 2
        tri_row_o, tri_col_o = np.tril_indices(nocc, -1)
        tri_row_v, tri_col_v = np.tril_indices(nvir, -1)

    rho = np.zeros((nroot, nmo, nmo))

    L_occ = Lpq[:, :, :nocc]
    L_vir = Lpq[:, :, nocc:]

    # Pre-transpose and reshape L_occ and L_vir for contractions
    L_occ_T = np.ascontiguousarray(L_occ.transpose(1, 0, 2)).reshape(nmo, -1)
    L_vir_T = np.ascontiguousarray(L_vir.transpose(1, 0, 2)).reshape(nmo, -1)

    T_oo = np.zeros((nmo, naux * nocc))
    T_vv = np.zeros((nmo, naux * nvir))
    term1 = np.zeros((nmo, nmo))

    for m in range(nroot):
        xy_m = xy[m]

        # Expand the compressed eigenvector into full occ-occ and vir-vir matrices
        # hh block (Y): xy_m[:oo_dim] -> Y_full[i, j]
        # pp block (X): xy_m[oo_dim:] -> X_full[a, b]
        Y_full = np.zeros((nocc, nocc))
        Y_full[tri_row_o, tri_col_o] = xy_m[:oo_dim]

        X_full = np.zeros((nvir, nvir))
        X_full[tri_row_v, tri_col_v] = xy_m[oo_dim:]

        # Scale diagonal elements for singlet
        Y_scaled = Y_full.copy()
        X_scaled = X_full.copy()
        if is_singlet:
            np.fill_diagonal(Y_scaled, Y_scaled.diagonal() / np.sqrt(2.0))
            np.fill_diagonal(X_scaled, X_scaled.diagonal() / np.sqrt(2.0))

        # Divide by sqrt(2) as in the C code (from Du Zhang's original code)
        scipy.linalg.blas.dscal(a=1.0 / np.sqrt(2.0), x=Y_scaled.ravel())
        scipy.linalg.blas.dscal(a=1.0 / np.sqrt(2.0), x=X_scaled.ravel())

        # Contract using matrix multiplication (BLAS GEMM) for efficiency
        # T_oo[P, q, i] = sum_j L_occ[P, q, j] * Y_scaled[i, j]
        # T_oo = (L_occ_T.reshape(-1, nocc) @ Y_scaled.T).reshape(nmo, -1)
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=Y_scaled.T,
            b=L_occ_T.reshape(-1, nocc).T,
            beta=0.0,
            c=T_oo.reshape(-1, nocc).T,
            trans_a=True,
            trans_b=False,
            overwrite_c=True,
        )

        # rho_m[p, q] += sum_{P, i} L_occ[P, p, i] * T_oo[P, q, i]
        #              + pm * sum_{P, i} L_occ[P, q, i] * T_oo[P, p, i]
        # term1 = L_occ_T @ T_oo.reshape(nmo, -1).T
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=T_oo.reshape(nmo, -1).T,
            b=L_occ_T.T,
            beta=0.0,
            c=term1.T,
            trans_a=True,
            trans_b=False,
            overwrite_c=True,
        )
        rho[m] += term1 + pm * term1.T

        # T_vv[P, q, a] = sum_b L_vir[P, q, b] * X_scaled[a, b]
        # T_vv = (L_vir_T.reshape(-1, nvir) @ X_scaled.T).reshape(nmo, -1)
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=X_scaled.T,
            b=L_vir_T.reshape(-1, nvir).T,
            beta=0.0,
            c=T_vv.reshape(-1, nvir).T,
            trans_a=True,
            trans_b=False,
            overwrite_c=True,
        )

        # rho_m[p, q] += sum_{P, a} L_vir[P, p, a] * T_vv[P, q, a]
        #              + pm * sum_{P, a} L_vir[P, q, a] * T_vv[P, p, a]
        # term1 = L_vir_T @ T_vv.reshape(nmo, -1).T
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=T_vv.reshape(nmo, -1).T,
            b=L_vir_T.T,
            beta=0.0,
            c=term1.T,
            trans_a=True,
            trans_b=False,
            overwrite_c=True,
        )
        rho[m] += term1 + pm * term1.T

    return rho


def get_sigma(nocc, mo_energy, mo_energy_ref, exci, rho, oo_dim, mu, eta=1.0e-5, fullsigma=False, mode="b"):
    r"""Get the real part of the T-matrix correlation self-energy for singlet or triplet channel.
    mode 'a' and 'b' correspond to equation 10 and 11 in doi.org/10.1103/PhysRevB.76.165106

    Parameters
    ----------
    nocc : int
        Number of occupied orbitals.
    mo_energy : double 1d array
        Orbital energies at which to evaluate the self-energy (QP energies).
    mo_energy_ref : double 1d array
        Reference orbital energies used in the Green's function.
    exci : double 1d array
        ppRPA eigenvalues, ordered as [hh states, pp states].
    rho : double 3d array
        Transition density, shape (nroot, nmo, nmo).
    oo_dim : int
        Number of hh eigenvalues (boundary between hh and pp sectors).
    mu : double
        Chemical potential.
    eta : double, optional
        Broadening parameter. Default 1.0e-5.
    fullsigma : bool, optional
        If True, compute full self-energy matrix. Default False (diagonal only).
    mode : str, optional
        Mode for off-diagonal elements, by default "b"

    Returns
    -------
    sigma : double 2d array
        Correlation self-energy. Diagonal matrix if fullsigma is False.
    """
    nmo = len(mo_energy)
    eta2 = (3.0 * eta) ** 2

    if fullsigma is False:
        sigma_diag = np.zeros(nmo)
        # hole-hole
        exci_hh = exci[:oo_dim]
        rho_hh = rho[:oo_dim, :, nocc:]
        ediff = mo_energy[None, :, None] + mo_energy_ref[None, None, nocc:] - 2.0 * mu - exci_hh[:, None, None]
        if ne is not None:
            contrib = ne.evaluate('(rho_hh ** 2) * ediff / (ediff ** 2 + eta2)')
        else:
            contrib = np.square(rho_hh) * ediff / (np.square(ediff) + eta2)
        sigma_diag += np.sum(contrib, axis=(0, 2))

        # particle-particle
        exci_pp = exci[oo_dim:]
        rho_pp = rho[oo_dim:, :, :nocc]
        ediff = mo_energy[None, :, None] + mo_energy_ref[None, None, :nocc] - 2.0 * mu - exci_pp[:, None, None]
        if ne is not None:
            contrib = ne.evaluate('(rho_pp ** 2) * ediff / (ediff ** 2 + eta2)')
        else:
            contrib = np.square(rho_pp) * ediff / (np.square(ediff) + eta2)
        sigma_diag += np.sum(contrib, axis=(0, 2))

        sigma = np.diag(sigma_diag)
    else:
        raise NotImplementedError("Full self-energy matrix computation is not implemented yet.")

    return sigma


def get_sigma_derivative(nocc, mo_energy, mo_energy_ref, exci, rho, oo_dim, mu, eta=1.0e-5):
    r"""Get the first-order derivative of the T-matrix self-energy
    with respect to frequency for one spin channel (singlet or triplet).

    d Sigma_c / d omega |_{omega=e_p}
    = sum_{m,q} rho_m(p,q)^2 * (eta^2 - ediff^2) / (ediff^2 + eta^2)^2

    Parameters
    ----------
    nocc : int
        Number of occupied orbitals.
    mo_energy : double 1d array
        Orbital energies.
    mo_energy_ref : double 1d array
        Reference orbital energies.
    exci : double 1d array
        ppRPA eigenvalues, ordered as [hh states, pp states].
    rho : double 3d array
        Transition density, shape (nroot, nmo, nmo).
    oo_dim : int
        Number of hh eigenvalues.
    mu : double
        Chemical potential.
    eta : double, optional
        Broadening parameter. Default 1.0e-5.

    Returns
    -------
    derivative : double 1d array
        First-order derivative of the correlation self-energy.
    """
    nmo = len(mo_energy)
    eta2 = (3.0 * eta) ** 2
    derivative = np.zeros(nmo)

    # m < oo_dim: q >= nocc
    exci_hh = exci[:oo_dim]
    rho_hh = rho[:oo_dim, :, nocc:]
    ediff = mo_energy[None, :, None] + mo_energy_ref[None, None, nocc:] - 2.0 * mu - exci_hh[:, None, None]
    if ne is not None:
        ediffsq = ne.evaluate('ediff ** 2')
        contrib = ne.evaluate('(rho_hh ** 2) * (eta2 - ediffsq) / (ediffsq + eta2) ** 2')
    else:
        ediffsq = np.square(ediff)
        contrib = np.square(rho_hh) * (eta2 - ediffsq) / np.square(ediffsq + eta2)
    derivative += np.sum(contrib, axis=(0, 2))

    # m >= oo_dim: q < nocc
    exci_pp = exci[oo_dim:]
    rho_pp = rho[oo_dim:, :, :nocc]
    ediff = mo_energy[None, :, None] + mo_energy_ref[None, None, :nocc] - 2.0 * mu - exci_pp[:, None, None]
    if ne is not None:
        ediffsq = ne.evaluate('ediff ** 2')
        contrib = ne.evaluate('(rho_pp ** 2) * (eta2 - ediffsq) / (ediffsq + eta2) ** 2')
    else:
        ediffsq = np.square(ediff)
        contrib = np.square(rho_pp) * (eta2 - ediffsq) / np.square(ediffsq + eta2)
    derivative += np.sum(contrib, axis=(0, 2))

    return derivative


def get_sigma_dynamic(nocc, mo_energy_ref, exci, rho, oo_dim, mu, omega, eta=1.0e-5, fullsigma=True):
    r"""Get the dynamical T-matrix self-energy and Green's function on a frequency grid.

    Parameters
    ----------
    nocc : int
        Number of occupied orbitals.
    mo_energy_ref : double 1d array
        Reference orbital energies.
    exci : double 1d array
        ppRPA eigenvalues, ordered as [hh, pp].
    rho : double 3d array
        Transition density, shape (nroot, nmo, nmo).
    oo_dim : int
        Number of hh eigenvalues.
    mu : double
        Chemical potential.
    omega : double or complex 1d array
        Frequency grid.
    eta : double, optional
        Broadening parameter. Default 1.0e-5.
    fullsigma : bool, optional
        If True, compute full self-energy matrix. Default True.

    Returns
    -------
    sigma : complex 3d array
        Self-energy, shape (nw, nmo, nmo).
    """
    nmo = len(mo_energy_ref)
    nw = len(omega)
    sigma = np.zeros(shape=[nw, nmo, nmo], dtype=np.complex128)

    if fullsigma is False:
        raise NotImplementedError
    else:
        exci_hh = np.ascontiguousarray(exci[:oo_dim])
        exci_pp = np.ascontiguousarray(exci[oo_dim:])
        rho_hh = np.ascontiguousarray(rho[:oo_dim, :, nocc:].transpose(1, 0, 2)).astype(np.complex128)
        rho_pp = np.ascontiguousarray(rho[oo_dim:, :, :nocc].transpose(1, 0, 2)).astype(np.complex128)
        rho_hh_tmp = np.zeros_like(rho_hh, dtype=np.complex128)
        rho_pp_tmp = np.zeros_like(rho_pp, dtype=np.complex128)

        if ne is not None:
            base_hh = ne.evaluate(
                "mo_energy_ref - 2.0 * mu - exci_hh + 3.0j * eta",
                local_dict={
                    "mo_energy_ref": mo_energy_ref[None, nocc:],
                    "mu": mu,
                    "exci_hh": exci_hh[:, None],
                    "eta": eta,
                },
            )
            base_pp = ne.evaluate(
                "mo_energy_ref - 2.0 * mu - exci_pp + 3.0j * eta",
                local_dict={
                    "mo_energy_ref": mo_energy_ref[None, :nocc],
                    "mu": mu,
                    "exci_pp": exci_pp[:, None],
                    "eta": eta,
                },
            )
        else:
            base_hh = mo_energy_ref[None, nocc:] - 2.0 * mu - exci_hh[:, None] + 3.0j * eta
            base_pp = mo_energy_ref[None, :nocc] - 2.0 * mu - exci_pp[:, None] + 3.0j * eta

        ediff_hh = np.zeros_like(base_hh, dtype=np.complex128)
        ediff_pp = np.zeros_like(base_pp, dtype=np.complex128)

        for w in range(nw):
            if ne is not None:
                ne.evaluate("omega + base_hh", local_dict={"omega": omega[w], "base_hh": base_hh}, out=ediff_hh)
                ne.evaluate("omega + base_pp", local_dict={"omega": omega[w], "base_pp": base_pp}, out=ediff_pp)
                ne.evaluate(
                    "rho_hh / ediff_hh", local_dict={"rho_hh": rho_hh, "ediff_hh": ediff_hh[None]}, out=rho_hh_tmp
                )
                ne.evaluate(
                    "rho_pp / ediff_pp", local_dict={"rho_pp": rho_pp, "ediff_pp": ediff_pp[None]}, out=rho_pp_tmp
                )
            else:
                ediff_hh[:] = omega[w] + mo_energy_ref[None, nocc:] - 2.0 * mu - exci_hh[:, None] + 3.0j * eta
                ediff_pp[:] = omega[w] + mo_energy_ref[None, :nocc] - 2.0 * mu - exci_pp[:, None] + 3.0j * eta
                rho_hh_tmp[:] = rho_hh / ediff_hh[None]
                rho_pp_tmp[:] = rho_pp / ediff_pp[None]
            sigma[w] += rho_hh.reshape(nmo, -1) @ rho_hh_tmp.reshape(nmo, -1).T
            sigma[w] += rho_pp.reshape(nmo, -1) @ rho_pp_tmp.reshape(nmo, -1).T

    return sigma


def kernel(tm):
    """Main kernel for the T-matrix calculation.

    Parameters
    ----------
    tm : TMatrix
        T-matrix object.
    """
    nmo = tm.nmo
    nocc = tm.nocc
    nvir = nmo - nocc
    mf = tm._scf
    mo_energy = np.asarray(tm.mo_energy)
    mo_coeff = np.asarray(tm.mo_coeff)
    mf_mo_energy = np.asarray(mf.mo_energy)

    # Get density-fitting integrals
    if tm.Lpq is None:
        tm.Lpq = tm.ao2mo(mo_coeff)

    # Chemical potential
    mu = get_chemical_potential(nocc=nocc, mo_energy=mf_mo_energy)
    tm.mu = mu

    # Mean-field exchange-correlation matrix
    tm.vxc = mo_coeff.T @ (mf.get_veff() - mf.get_j()) @ mo_coeff

    # Exchange self-energy from density fitting
    if tm.vhf_df:
        vk = -np.einsum('Lpi,Liq->pq', tm.Lpq[:, :, :nocc], tm.Lpq[:, :nocc, :], optimize=True)
    else:
        dm = mf.make_rdm1()
        if (not isinstance(mf, dft.rks.RKS)) and isinstance(mf, scf.hf.RHF):
            rhf = mf
        else:
            rhf = scf.RHF(tm.mol)
        vk = rhf.get_veff(dm=dm) - rhf.get_j(dm=dm)
        vk = mo_coeff.T @ vk @ mo_coeff
    tm.vk = vk

    # Diagonalize ppRPA for singlet and triplet channels
    start_clock("ppRPA diagonalization: singlet")
    exci_s, xy_s, _ = diagonalize_pprpa_singlet(nocc=nocc, mo_energy=mf_mo_energy, Lpq=tm.Lpq, mu=mu)
    stop_clock("ppRPA diagonalization: singlet")

    start_clock("ppRPA diagonalization: triplet")
    exci_t, xy_t, _ = diagonalize_pprpa_triplet(nocc=nocc, mo_energy=mf_mo_energy, Lpq=tm.Lpq, mu=mu)
    stop_clock("ppRPA diagonalization: triplet")

    tm.exci_s = exci_s
    tm.exci_t = exci_t
    tm.xy_s = xy_s
    tm.xy_t = xy_t

    oo_dim_s = (nocc + 1) * nocc // 2
    oo_dim_t = (nocc - 1) * nocc // 2

    # Compute transition densities
    start_clock("T-matrix transition density: singlet")
    rho_s = get_transition_density(multi='s', nocc=nocc, nvir=nvir, xy=xy_s, Lpq=tm.Lpq)
    stop_clock("T-matrix transition density: singlet")

    start_clock("T-matrix transition density: triplet")
    rho_t = get_transition_density(multi='t', nocc=nocc, nvir=nvir, xy=xy_t, Lpq=tm.Lpq)
    stop_clock("T-matrix transition density: triplet")

    tm.rho_s = rho_s
    tm.rho_t = rho_t

    # Compute self-energy
    # For restricted T-matrix: Sigma_c = Sigma_s + 3 * Sigma_t
    start_clock("T-matrix self-energy: singlet")
    sigma_s = get_sigma(
        nocc=nocc, mo_energy=mo_energy, mo_energy_ref=mf_mo_energy,
        exci=exci_s, rho=rho_s, oo_dim=oo_dim_s, mu=mu, eta=tm.eta,
        fullsigma=False)
    stop_clock("T-matrix self-energy: singlet")

    start_clock("T-matrix self-energy: triplet")
    sigma_t = get_sigma(
        nocc=nocc, mo_energy=mo_energy, mo_energy_ref=mf_mo_energy,
        exci=exci_t, rho=rho_t, oo_dim=oo_dim_t, mu=mu, eta=tm.eta,
        fullsigma=False)
    stop_clock("T-matrix self-energy: triplet")

    sigma = sigma_s + 3.0 * sigma_t

    # Quasiparticle equation
    if tm.qpe_linearized:
        # Linearized one-shot G0T0
        der_s = get_sigma_derivative(
            nocc=nocc, mo_energy=mo_energy, mo_energy_ref=mf_mo_energy,
            exci=exci_s, rho=rho_s, oo_dim=oo_dim_s, mu=mu, eta=tm.eta)
        der_t = get_sigma_derivative(
            nocc=nocc, mo_energy=mo_energy, mo_energy_ref=mf_mo_energy,
            exci=exci_t, rho=rho_t, oo_dim=oo_dim_t, mu=mu, eta=tm.eta)
        derivative = der_s + 3.0 * der_t
        z = 1.0 / (1.0 - derivative)
        if tm.qpe_linearized_range is not None:
            z = np.where(
                (z < tm.qpe_linearized_range[0]) |
                (z > tm.qpe_linearized_range[1]),
                1.0, z)
        mo_energy = mf_mo_energy + z * (vk + sigma).diagonal()
        mo_energy += - z * np.diag(tm.vxc)
    else:
        # Iterative solution of the quasiparticle equation
        def quasiparticle(qp_energy):
            s_s = get_sigma(
                nocc=nocc, mo_energy=qp_energy, mo_energy_ref=mf_mo_energy,
                exci=exci_s, rho=rho_s, oo_dim=oo_dim_s, mu=mu, eta=tm.eta,
                fullsigma=False)
            s_t = get_sigma(
                nocc=nocc, mo_energy=qp_energy, mo_energy_ref=mf_mo_energy,
                exci=exci_t, rho=rho_t, oo_dim=oo_dim_t, mu=mu, eta=tm.eta,
                fullsigma=False)
            s_total = s_s + 3.0 * s_t
            return qp_energy - (mf_mo_energy +
                                (s_total + vk - tm.vxc).diagonal())

        try:
            mo_energy = scipy.optimize.newton(
                quasiparticle, mf_mo_energy,
                tol=tm.qpe_tol * nmo, maxiter=tm.qpe_max_iter)
        except RuntimeError:
            print('WARNING: quasiparticle equation fails to converge!')

    tm.mo_energy = mo_energy
    print('\n  T-matrix QP energies (Hartree):')
    for i in range(nmo):
        marker = ' (HOMO)' if i == nocc - 1 else \
                 ' (LUMO)' if i == nocc else ''
        print('    MO %4d:  KS = %12.6f  QP = %12.6f%s' %
              (i + 1, mf_mo_energy[i], mo_energy[i], marker))
    print('')

    return


def get_g0(omega, mo_energy, eta):
    r"""Build the non-interacting Green's function on a frequency grid.

    G0(p, p; w) = 1 / (w - e_p +/- i*eta)

    Parameters
    ----------
    omega : complex 1d array
        Frequency grid.
    mo_energy : double 1d array
        Orbital energies.
    eta : double
        Broadening parameter.

    Returns
    -------
    gf0 : complex 3d array
        Non-interacting Green's function, shape (nw, nmo, nmo).
    """
    nmo = len(mo_energy)
    nw = len(omega)
    gf0 = np.zeros(shape=[nw, nmo, nmo], dtype=np.complex128)
    gf0[:, np.arange(nmo), np.arange(nmo)] = 1.0 / (omega[:, None] - mo_energy[None, :] + 1j * eta)
    return gf0


# =============================================================================
# TMatrix class
# =============================================================================

class TMatrix(lib.StreamObject):
    """Restricted T-matrix method based on ppRPA.

    Attributes
    ----------
    mol : Mole
        PySCF Mole object.
    _scf : SCF
        PySCF mean-field object (RHF or RKS).
    nocc : int
        Number of occupied orbitals.
    nmo : int
        Total number of molecular orbitals.
    mo_energy : double 1d array
        Quasiparticle energies.
    mo_coeff : double 2d array
        MO coefficient matrix.
    Lpq : double 3d array
        Three-center density-fitting matrix in MO space.
    exci_s : double 1d array
        Singlet ppRPA eigenvalues.
    exci_t : double 1d array
        Triplet ppRPA eigenvalues.
    rho_s : double 3d array
        Singlet transition density.
    rho_t : double 3d array
        Triplet transition density.
    vk : double 2d array
        Exchange self-energy.
    vxc : double 2d array
        Mean-field exchange-correlation matrix.
    mu : double
        Chemical potential.
    """

    def __init__(self, mf, auxbasis=None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.auxbasis = auxbasis

        # options
        self.eta = 5.0e-3
        self.vhf_df = True  # use density-fitting for exchange self-energy
        self.qpe_linearized = False
        self.qpe_linearized_range = [0.5, 1.5]
        self.qpe_max_iter = 100
        self.qpe_tol = 1.0e-6

        # internal quantities
        self._nocc = None
        self._nmo = None
        self.mo_energy = np.array(mf.mo_energy, copy=True)
        self.mo_coeff = np.array(mf.mo_coeff, copy=True)
        self.Lpq = None

        # results
        self.mu = None
        self.vk = None
        self.vxc = None
        self.exci_s = None
        self.exci_t = None
        self.xy_s = None
        self.xy_t = None
        self.rho_s = None
        self.rho_t = None

        return

    @property
    def nocc(self):
        if self._nocc is not None:
            return self._nocc
        return self._scf.mol.nelectron // 2

    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        if self._nmo is not None:
            return self._nmo
        return len(self._scf.mo_energy)

    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    def dump_flags(self):
        print('')
        print('******** %s ********' % self.__class__)
        print('method = %s' % self.__class__.__name__)
        print('T-matrix nocc = %d, nvir = %d' % (self.nocc, self.nmo - self.nocc))
        print('density-fitting for exchange = %s' % self.vhf_df)
        print('broadening parameter = %.3e' % self.eta)
        print('use perturbative linearized QP eqn = %s' % self.qpe_linearized)
        if self.qpe_linearized:
            print('linearized factor range = %s' % self.qpe_linearized_range)
        else:
            print('QPE max iter = %d' % self.qpe_max_iter)
            print('QPE tolerance = %.1e' % self.qpe_tol)
        print('')
        return

    def initialize_df(self, auxbasis=None):
        """Initialize density fitting."""
        if getattr(self._scf, 'with_df', None):
            self.with_df = self._scf.with_df
        else:
            self.with_df = df.DF(self._scf.mol)
            if auxbasis is not None:
                self.with_df.auxbasis = auxbasis
            else:
                try:
                    self.with_df.auxbasis = df.make_auxbasis(
                        self._scf.mol, mp2fit=True)
                except RuntimeError:
                    self.with_df.auxbasis = df.make_auxbasis(
                        self._scf.mol, mp2fit=False)
        return

    def ao2mo(self, mo_coeff=None):
        """Transform density-fitting integrals from AO to MO.

        Parameters
        ----------
        mo_coeff : double 2d array, optional
            MO coefficient matrix.

        Returns
        -------
        Lpq : double 3d array
            Three-center density-fitting matrix in MO, shape (naux, nmo, nmo).
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        nmo = self.nmo

        if not hasattr(self, 'with_df'):
            self.initialize_df(auxbasis=self.auxbasis)

        naux = self.with_df.get_naoaux()
        mo = np.asarray(mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        Lpq = None
        Lpq = _ao2mo.nr_e2(self.with_df._cderi, mo, ijslice,
                           aosym='s2', out=Lpq)
        return Lpq.reshape(naux, nmo, nmo)

    def kernel(self):
        """Run the T-matrix calculation."""
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        self.dump_flags()
        cput0 = (time.process_time(), time.perf_counter())
        kernel(self)
        cpu_time = time.process_time() - cput0[0]
        wall_time = time.perf_counter() - cput0[1]
        print('T-matrix CPU time: %.2f s, wall time: %.2f s' %
              (cpu_time, wall_time))
        return

    def make_gf(self, omega, eta, fullsigma=True, mode='linear'):
        r"""Get exact dynamical Green's function and self-energy.

        Two modes for solving Dyson equation:
        - 'linear': G = G0 + G0 Sigma G0 (perturbative)
        - 'dyson':  G = (G0^{-1} - Sigma)^{-1} (full Dyson)

        Parameters
        ----------
        omega : double or complex 1d array
            Frequency grid.
        eta : double
            Broadening parameter.
        fullsigma : bool, optional
            Compute off-diagonal self-energy elements. Default True.
        mode : str, optional
            'linear' or 'dyson'. Default 'linear'.

        Returns
        -------
        gf : complex 3d array
            T-matrix Green's function, shape (nmo, nmo, nw).
        gf0 : complex 3d array
            Non-interacting Green's function, shape (nmo, nmo, nw).
        sigma : complex 3d array
            Correlation self-energy, shape (nmo, nmo, nw).
        """
        nmo = self.nmo
        nocc = self.nocc
        mo_energy = np.asarray(self._scf.mo_energy)
        oo_dim_s = (nocc + 1) * nocc // 2
        oo_dim_t = (nocc - 1) * nocc // 2

        # Dynamical self-energy
        sigma_s = get_sigma_dynamic(
            nocc=nocc, mo_energy_ref=mo_energy, exci=self.exci_s,
            rho=self.rho_s, oo_dim=oo_dim_s, mu=self.mu,
            omega=omega, eta=eta, fullsigma=fullsigma)
        sigma_t = get_sigma_dynamic(
            nocc=nocc, mo_energy_ref=mo_energy, exci=self.exci_t,
            rho=self.rho_t, oo_dim=oo_dim_t, mu=self.mu,
            omega=omega, eta=eta, fullsigma=fullsigma)
        sigma = sigma_s + 3.0 * sigma_t

        # Non-interacting Green's function
        gf0 = get_g0(omega, mo_energy, eta)

        # Dyson equation
        gf = np.zeros_like(gf0)
        sigma_diff = np.array(sigma, copy=True)
        if fullsigma:
            sigma_diff += self.vk - self.vxc
        else:
            for w in range(len(omega)):
                for i in range(nmo):
                    sigma_diff[w, i, i] += self.vk[i, i] - self.vxc[i, i]

        if mode == 'linear':
            gf = gf0 + gf0 @ sigma_diff @ gf0
        elif mode == 'dyson':
            gf = np.linalg.inv(np.linalg.inv(gf0) - sigma_diff)

        return gf, gf0, sigma

    def make_diag_dos(self, omega, eta):
        """Get orbital-resolved density of states using diagonal self-energy.

        Parameters
        ----------
        omega : double 1d array
            Real frequency grid.
        eta : double
            Broadening parameter.

        Returns
        -------
        dos : double 2d array
            Orbital-resolved density of states, shape (nmo, nw).
        """
        nocc = self.nocc
        nmo = self.nmo
        mo_energy = np.asarray(self._scf.mo_energy)
        oo_dim_s = (nocc + 1) * nocc // 2
        oo_dim_t = (nocc - 1) * nocc // 2
        eta2 = (3.0 * eta) ** 2
        nw = len(omega)

        sigma_real = np.zeros((nmo, nw))
        sigma_imag = np.zeros((nmo, nw))

        for multi, exci, rho, oo_dim, factor in [
            ('s', self.exci_s, self.rho_s, oo_dim_s, 1.0),
            ('t', self.exci_t, self.rho_t, oo_dim_t, 3.0),
        ]:
            nroot = len(exci)
            chunk_size = 500

            # m < oo_dim: q >= nocc
            if oo_dim > 0:
                exci_hh = exci[:oo_dim]
                rho_hh = rho[:oo_dim, :, nocc:]  # (oo_dim, nmo, nvir)
                nvir = nmo - nocc
                for q_idx in range(nvir):
                    q = nocc + q_idx
                    rho_q = rho_hh[:, :, q_idx]  # (oo_dim, nmo)
                    for start in range(0, oo_dim, chunk_size):
                        end = min(start + chunk_size, oo_dim)
                        exci_chunk = exci_hh[start:end]
                        rho_chunk = rho_q[start:end]
                        ediff = omega[None, :] + mo_energy[q] - 2.0 * self.mu - exci_chunk[:, None]
                        denom = ediff ** 2 + eta2
                        contrib_real = ediff / denom
                        contrib_imag = -eta / denom
                        sigma_real += factor * ((rho_chunk ** 2).T @ contrib_real)
                        sigma_imag += factor * ((rho_chunk ** 2).T @ contrib_imag)

            # m >= oo_dim: q < nocc
            if nroot > oo_dim:
                exci_pp = exci[oo_dim:]
                rho_pp = rho[oo_dim:, :, :nocc]  # (vv_dim, nmo, nocc)
                vv_dim = nroot - oo_dim
                for q in range(nocc):
                    rho_q = rho_pp[:, :, q]  # (vv_dim, nmo)
                    for start in range(0, vv_dim, chunk_size):
                        end = min(start + chunk_size, vv_dim)
                        exci_chunk = exci_pp[start:end]
                        rho_chunk = rho_q[start:end]
                        ediff = omega[None, :] + mo_energy[q] - 2.0 * self.mu - exci_chunk[:, None]
                        denom = ediff ** 2 + eta2
                        contrib_real = ediff / denom
                        contrib_imag = eta / denom
                        sigma_real += factor * ((rho_chunk ** 2).T @ contrib_real)
                        sigma_imag += factor * ((rho_chunk ** 2).T @ contrib_imag)

        vk_minus_vxc = (self.vk - self.vxc).diagonal()
        ereal = (omega[None, :] - mo_energy[:, None]
                 - (sigma_real + vk_minus_vxc[:, None]))
        dos = np.abs(sigma_imag) / (ereal ** 2 + sigma_imag ** 2)
        dos /= np.pi

        return dos

    def energy_tot(self, nw=60):
        r"""Calculate T-matrix total energy using Galitskii-Migdal formula.

        E_tot = E_HF + E_c
        E_c = 2 * Re[ Tr(G0 * Sigma_c) ]

        Parameters
        ----------
        nw : int, optional
            Number of imaginary frequency grids. Default 60.

        Returns
        -------
        e_tot : double
            T-matrix total energy.
        e_hf : double
            HF total energy.
        e_c : double
            T-matrix correlation energy.
        """
        from pyscf.lib import temporary_env

        # Gauss-Legendre quadrature on imaginary axis
        pts, wts = np.polynomial.legendre.leggauss(nw)
        # Map from [-1, 1] to [0, inf) using the transformation
        # x = (1+t)/(1-t), dx = 2/(1-t)^2 dt
        freqs = (1.0 + pts) / (1.0 - pts)
        weights = wts * 2.0 / (1.0 - pts) ** 2

        mo_energy = np.asarray(self._scf.mo_energy)
        ef = (mo_energy[self.nocc - 1] + mo_energy[self.nocc]) * 0.5
        omega = 1j * freqs + ef

        _, gf0, sigma = self.make_gf(
            omega=omega, eta=0, fullsigma=True, mode='linear')

        # GW-type correlation energy
        g0_sigma = np.einsum('ijw,jiw,w->', gf0, sigma, weights)
        e_c = 2.0 * (1.0 / (2.0 * np.pi) * g0_sigma).real

        # HF energy with DFT density matrix
        dm = self._scf.make_rdm1()
        if (not isinstance(self._scf, dft.rks.RKS)) and isinstance(
                self._scf, scf.hf.RHF):
            rhf = self._scf
        else:
            rhf = scf.RHF(self.mol)
        with temporary_env(rhf, verbose=0):
            e_hf = rhf.energy_elec(dm=dm)[0] + self._scf.energy_nuc()

        e_tot = e_hf + e_c

        print('HF energy@T-matrix density  = %.8f' % e_hf)
        print('T-matrix correlation energy = %.8f' % e_c)
        print('T-matrix total energy       = %.8f' % e_tot)

        return e_tot, e_hf, e_c


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, -0.7571, 0.5861)],
        [1, (0.0, 0.7571, 0.5861)],
    ]
    mol.basis = 'def2-svp'
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    # T-matrix with linearized QP equation
    tm = TMatrix(mf)
    tm.eta = 1.0e-5
    tm.qpe_linearized = True
    tm.kernel()

    print('\n--- T-matrix with iterative QP equation ---')
    tm2 = TMatrix(mf)
    tm2.eta = 1.0e-5
    tm2.qpe_linearized = False
    tm2.kernel()

    # Density of states
    omega = np.linspace(-0.8, 0.5, 201)
    gf, gf0, _ = tm2.make_gf(omega=omega, eta=0.01, fullsigma=True,
                              mode='dyson')
    print('\nDOS: KS, T-matrix')
    for iw in range(len(omega)):
        print('%.4f  %.6f  %.6f' % (
            omega[iw],
            -np.trace(gf0[:, :, iw].imag) / np.pi,
            -np.trace(gf[:, :, iw].imag) / np.pi))
