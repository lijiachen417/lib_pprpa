"""
Spin-unrestricted T-matrix method based on the particle-particle RPA.
This implementation has N^6 scaling, and is accurate for all states.

The unrestricted pair space separates into alpha-alpha, beta-beta, and
alpha-beta channels.  The alpha self-energy contains alpha-alpha and
alpha-beta contributions, while the beta self-energy contains beta-beta and
the transposed alpha-beta contribution.

References:
    [1] Du Zhang, Neil Qiang Su, and Weitao Yang,
        "Accurate Quasiparticle Spectra from the T-Matrix Self-Energy
         and the Particle-Particle Random Phase Approximation",
        J. Phys. Chem. Lett. 2017, 8, 3223-3227.
    [2] Jiachen Li, Zehua Chen, and Weitao Yang,
        "Renormalized Singles Green's Function in the T-Matrix Approximation
         for Accurate Quasiparticle Energy Calculation",
        J. Phys. Chem. Lett. 2021, 12, 6203-6210.
"""

import time

import numpy as np
import scipy

try:
    import numexpr as ne
except ImportError:
    ne = None

from pyscf import df, dft, lib, scf
from pyscf.ao2mo import _ao2mo

from lib_pprpa.pprpa_util import get_chemical_potential, start_clock, stop_clock
from lib_pprpa.tmatrix import (
    get_g0,
    get_sigma,
    get_sigma_derivative,
    get_transition_density as get_restricted_transition_density,
)
from lib_pprpa.upprpa_direct import (
    diagonalize_pprpa_subspace_diff_spin,
    diagonalize_pprpa_subspace_same_spin,
)


# =============================================================================
# Module-level functions
# =============================================================================

def get_transition_density_same_spin(nocc, nvir, xy, Lpq):
    r"""Calculate an unrestricted same-spin T-matrix transition density.

    The unrestricted alpha-alpha and beta-beta pair eigenvectors use bare
    antisymmetric pair states.  Their transition densities therefore differ
    by ``sqrt(2)`` from the spin-adapted restricted triplet density.

    Parameters
    ----------
    nocc : int
        Number of occupied orbitals in the spin channel.
    nvir : int
        Number of virtual orbitals in the spin channel.
    xy : double 2d array
        ppRPA eigenvectors ordered as ``[hh, pp]``.
    Lpq : double 3d array
        Three-center density-fitting matrix for the spin channel.

    Returns
    -------
    rho : double 3d array
        Same-spin transition density, shape ``(nroot, nmo, nmo)``.
    """
    rho = get_restricted_transition_density(
        multi='t', nocc=nocc, nvir=nvir, xy=xy, Lpq=Lpq)
    rho *= np.sqrt(2.0)
    return rho


def get_transition_density_diff_spin(nocc, nvir, xy, Lpq):
    r"""Calculate the alpha-beta T-matrix transition density.

    ``rho[m, p, q]`` is oriented as ``p = alpha`` and ``q = beta``:

    .. math::

        \rho_m(p_\alpha,q_\beta) =
        \sum_{ijP} L^\alpha_{Ppi}Y^m_{ij}L^\beta_{Pqj}
        + \sum_{abP} L^\alpha_{Ppa}X^m_{ab}L^\beta_{Pqb}.

    Parameters
    ----------
    nocc : tuple of int
        Numbers of occupied alpha and beta orbitals.
    nvir : tuple of int
        Numbers of virtual alpha and beta orbitals.
    xy : double 2d array
        Alpha-beta ppRPA eigenvectors ordered as ``[hh, pp]``.
    Lpq : sequence of double 3d arrays
        Alpha and beta three-center density-fitting matrices.

    Returns
    -------
    rho : double 3d array
        Opposite-spin transition density, shape
        ``(nroot, nmo_alpha, nmo_beta)``.
    """
    xy = np.asarray(np.real_if_close(xy))
    if np.iscomplexobj(xy):
        raise ValueError(
            "Complex alpha-beta ppRPA modes are not supported by UTMatrix.")

    nocc_a, nocc_b = (int(nocc[0]), int(nocc[1]))
    nvir_a, nvir_b = (int(nvir[0]), int(nvir[1]))
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b
    nroot = xy.shape[0]
    oo_dim = nocc_a * nocc_b

    Lpq_a = np.asarray(Lpq[0], dtype=np.double)
    Lpq_b = np.asarray(Lpq[1], dtype=np.double)
    if Lpq_a.shape[0] != Lpq_b.shape[0]:
        raise ValueError("Alpha and beta density-fitting tensors must share naux.")
    if Lpq_a.shape[1:] != (nmo_a, nmo_a):
        raise ValueError("Alpha density-fitting tensor has inconsistent dimensions.")
    if Lpq_b.shape[1:] != (nmo_b, nmo_b):
        raise ValueError("Beta density-fitting tensor has inconsistent dimensions.")

    xy = np.asarray(xy, dtype=np.double)
    naux = Lpq_a.shape[0]
    rho = np.zeros((nroot, nmo_a, nmo_b))

    # Arrange the DF tensors as [external orbital, auxiliary, pair orbital]
    # once.  Their flattened transposes are Fortran-contiguous views, so the
    # BLAS calls below do not repack the large, root-independent operands.
    L_a_occ_T = np.ascontiguousarray(
        Lpq_a[:, :, :nocc_a].transpose(1, 0, 2)).reshape(nmo_a, -1)
    L_b_occ_T = np.ascontiguousarray(
        Lpq_b[:, :, :nocc_b].transpose(1, 0, 2)).reshape(nmo_b, -1)
    L_a_vir_T = np.ascontiguousarray(
        Lpq_a[:, :, nocc_a:].transpose(1, 0, 2)).reshape(nmo_a, -1)
    L_b_vir_T = np.ascontiguousarray(
        Lpq_b[:, :, nocc_b:].transpose(1, 0, 2)).reshape(nmo_b, -1)

    # T_oo[p, P, j_beta] and T_vv[p, P, b_beta].  The buffers are reused for
    # every ppRPA root, matching the allocation strategy in tmatrix.py.
    T_oo = np.zeros((nmo_a, naux * nocc_b))
    T_vv = np.zeros((nmo_a, naux * nvir_b))

    for m in range(nroot):
        Y = xy[m, :oo_dim].reshape(nocc_a, nocc_b)
        X = xy[m, oo_dim:].reshape(nvir_a, nvir_b)

        # T_oo[p, P, j] = sum_i L_alpha[P, p, i] * Y[i, j]
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=Y.T,
            b=L_a_occ_T.reshape(-1, nocc_a).T,
            beta=0.0,
            c=T_oo.reshape(-1, nocc_b).T,
            overwrite_c=True,
        )

        # rho[p, q] = sum_{P,j} T_oo[p, P, j] * L_beta[P, q, j]
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=L_b_occ_T.T,
            b=T_oo.T,
            beta=0.0,
            c=rho[m].T,
            trans_a=True,
            overwrite_c=True,
        )

        # T_vv[p, P, b] = sum_a L_alpha[P, p, a] * X[a, b]
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=X.T,
            b=L_a_vir_T.reshape(-1, nvir_a).T,
            beta=0.0,
            c=T_vv.reshape(-1, nvir_b).T,
            overwrite_c=True,
        )

        # Accumulate the virtual-pair contribution into the same rho buffer.
        scipy.linalg.blas.dgemm(
            alpha=1.0,
            a=L_b_vir_T.T,
            b=T_vv.T,
            beta=1.0,
            c=rho[m].T,
            trans_a=True,
            overwrite_c=True,
        )

    return rho


def get_sigma_dynamic(nocc, mo_energy_ref, exci, rho, oo_dim, mu,
                      omega, eta=1.0e-5, fullsigma=True):
    r"""Get one pair channel of the dynamical T-matrix self-energy.

    Unlike the restricted helper, the external and internal orbital axes are
    treated independently.  This permits an alpha-beta transition density to
    be used directly for alpha and transposed for beta.

    Parameters
    ----------
    nocc : int
        Number of occupied orbitals on the internal Green's-function line.
    mo_energy_ref : double 1d array
        Reference energies for the internal spin channel.
    exci : double 1d array
        ppRPA eigenvalues ordered as ``[hh, pp]``.
    rho : double 3d array
        Oriented transition density, shape
        ``(nroot, nmo_external, nmo_internal)``.
    oo_dim : int
        Number of hole-hole eigenvalues.
    mu : double
        Chemical potential.
    omega : double or complex 1d array
        Frequency grid.
    eta : double, optional
        Broadening parameter.
    fullsigma : bool, optional
        Compute the full external-orbital self-energy matrix.

    Returns
    -------
    sigma : complex 3d array
        Self-energy, shape ``(nw, nmo_external, nmo_external)``.
    """
    if fullsigma is False:
        raise NotImplementedError(
            "Diagonal-only dynamical self-energy is not implemented; "
            "use make_diag_dos instead.")

    mo_energy_ref = np.asarray(mo_energy_ref)
    omega = np.asarray(omega)
    nmo_external = rho.shape[1]
    nw = len(omega)
    sigma = np.zeros(
        (nw, nmo_external, nmo_external), dtype=np.complex128)

    exci_hh = np.ascontiguousarray(exci[:oo_dim])
    exci_pp = np.ascontiguousarray(exci[oo_dim:])
    rho_hh = np.ascontiguousarray(
        rho[:oo_dim, :, nocc:].transpose(1, 0, 2)).astype(np.complex128)
    rho_pp = np.ascontiguousarray(
        rho[oo_dim:, :, :nocc].transpose(1, 0, 2)).astype(np.complex128)
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
        base_hh = (
            mo_energy_ref[None, nocc:] - 2.0 * mu
            - exci_hh[:, None] + 3.0j * eta)
        base_pp = (
            mo_energy_ref[None, :nocc] - 2.0 * mu
            - exci_pp[:, None] + 3.0j * eta)

    ediff_hh = np.zeros_like(base_hh, dtype=np.complex128)
    ediff_pp = np.zeros_like(base_pp, dtype=np.complex128)
    rho_hh_flat = rho_hh.reshape(nmo_external, -1)
    rho_pp_flat = rho_pp.reshape(nmo_external, -1)

    for w in range(nw):
        if ne is not None:
            ne.evaluate(
                "omega + base_hh",
                local_dict={"omega": omega[w], "base_hh": base_hh},
                out=ediff_hh,
            )
            ne.evaluate(
                "omega + base_pp",
                local_dict={"omega": omega[w], "base_pp": base_pp},
                out=ediff_pp,
            )
            ne.evaluate(
                "rho_hh / ediff_hh",
                local_dict={"rho_hh": rho_hh,
                            "ediff_hh": ediff_hh[None]},
                out=rho_hh_tmp,
            )
            ne.evaluate(
                "rho_pp / ediff_pp",
                local_dict={"rho_pp": rho_pp,
                            "ediff_pp": ediff_pp[None]},
                out=rho_pp_tmp,
            )
        else:
            ediff_hh[:] = omega[w] + base_hh
            ediff_pp[:] = omega[w] + base_pp
            rho_hh_tmp[:] = rho_hh / ediff_hh[None]
            rho_pp_tmp[:] = rho_pp / ediff_pp[None]

        sigma[w] += (
            rho_hh_flat
            @ rho_hh_tmp.reshape(nmo_external, -1).T)
        sigma[w] += (
            rho_pp_flat
            @ rho_pp_tmp.reshape(nmo_external, -1).T)

    return sigma


def _pair_channel_specs(tm, spin):
    """Return pair channels oriented for one external spin."""
    nocc_a, nocc_b = tm.nocc
    mo_energy_ref = np.asarray(tm._scf.mo_energy)
    oo_dim_aa = (nocc_a - 1) * nocc_a // 2
    oo_dim_bb = (nocc_b - 1) * nocc_b // 2
    oo_dim_ab = nocc_a * nocc_b

    if spin == 0:
        return [
            (nocc_a, mo_energy_ref[0], tm.exci_aa, tm.rho_aa,
             oo_dim_aa),
            (nocc_b, mo_energy_ref[1], tm.exci_ab, tm.rho_ab,
             oo_dim_ab),
        ]
    if spin == 1:
        return [
            (nocc_b, mo_energy_ref[1], tm.exci_bb, tm.rho_bb,
             oo_dim_bb),
            (nocc_a, mo_energy_ref[0], tm.exci_ab,
             tm.rho_ab.transpose(0, 2, 1), oo_dim_ab),
        ]
    raise ValueError("spin must be 0 (alpha) or 1 (beta)")


def _get_sigma_for_spin(tm, spin, mo_energy):
    """Get the static diagonal self-energy for one external spin."""
    nmo = len(mo_energy)
    sigma = np.zeros((nmo, nmo))
    for nocc, mo_energy_ref, exci, rho, oo_dim in _pair_channel_specs(
            tm, spin):
        sigma += get_sigma(
            nocc=nocc,
            mo_energy=mo_energy,
            mo_energy_ref=mo_energy_ref,
            exci=exci,
            rho=rho,
            oo_dim=oo_dim,
            mu=tm.mu,
            eta=tm.eta,
            fullsigma=False,
        )
    return sigma


def _get_sigma_derivative_for_spin(tm, spin, mo_energy):
    """Get the self-energy derivative for one external spin."""
    derivative = np.zeros(len(mo_energy))
    for nocc, mo_energy_ref, exci, rho, oo_dim in _pair_channel_specs(
            tm, spin):
        derivative += get_sigma_derivative(
            nocc=nocc,
            mo_energy=mo_energy,
            mo_energy_ref=mo_energy_ref,
            exci=exci,
            rho=rho,
            oo_dim=oo_dim,
            mu=tm.mu,
            eta=tm.eta,
        )
    return derivative


def _get_sigma_dynamic_for_spin(tm, spin, omega, eta, fullsigma):
    """Get the dynamical self-energy for one external spin."""
    nmo = tm.nmo[spin]
    sigma = np.zeros((len(omega), nmo, nmo), dtype=np.complex128)
    for nocc, mo_energy_ref, exci, rho, oo_dim in _pair_channel_specs(
            tm, spin):
        sigma += get_sigma_dynamic(
            nocc=nocc,
            mo_energy_ref=mo_energy_ref,
            exci=exci,
            rho=rho,
            oo_dim=oo_dim,
            mu=tm.mu,
            omega=omega,
            eta=eta,
            fullsigma=fullsigma,
        )
    return sigma


def kernel(tm):
    """Run an unrestricted T-matrix calculation."""
    nocc = tm.nocc
    nmo = tm.nmo
    nvir = (nmo[0] - nocc[0], nmo[1] - nocc[1])
    mf = tm._scf
    mo_energy = np.asarray(tm.mo_energy)
    mo_coeff = np.asarray(tm.mo_coeff)
    mf_mo_energy = np.asarray(mf.mo_energy)

    if tm.Lpq is None:
        tm.Lpq = tm.ao2mo(mo_coeff)

    tm.mu = get_chemical_potential(
        nocc=nocc, mo_energy=mf_mo_energy)

    dm = np.asarray(mf.make_rdm1())
    veff = np.asarray(mf.get_veff(dm=dm))
    vj = np.asarray(mf.get_j(dm=dm))
    vj_total = np.sum(vj, axis=0) if vj.ndim == 3 else vj
    vxc_ao = veff - vj_total[None, :, :]
    tm.vxc = np.asarray([
        mo_coeff[s].T @ vxc_ao[s] @ mo_coeff[s]
        for s in range(2)
    ])

    if tm.vhf_df:
        tm.vk = np.asarray([
            -np.einsum(
                'Lpi,Liq->pq',
                tm.Lpq[s][:, :, :nocc[s]],
                tm.Lpq[s][:, :nocc[s], :],
                optimize=True,
            )
            for s in range(2)
        ])
    else:
        if (isinstance(mf, scf.uhf.UHF)
                and not isinstance(mf, dft.uks.UKS)):
            uhf = mf
        else:
            uhf = scf.UHF(tm.mol)
        vk_ao = -np.asarray(uhf.get_k(dm=dm))
        tm.vk = np.asarray([
            mo_coeff[s].T @ vk_ao[s] @ mo_coeff[s]
            for s in range(2)
        ])

    start_clock("U-ppRPA diagonalization: alpha-alpha")
    exci_aa, xy_aa, _ = diagonalize_pprpa_subspace_same_spin(
        nocc=nocc[0], mo_energy=mf_mo_energy[0], Lpq=tm.Lpq[0],
        mu=tm.mu)
    stop_clock("U-ppRPA diagonalization: alpha-alpha")

    start_clock("U-ppRPA diagonalization: beta-beta")
    exci_bb, xy_bb, _ = diagonalize_pprpa_subspace_same_spin(
        nocc=nocc[1], mo_energy=mf_mo_energy[1], Lpq=tm.Lpq[1],
        mu=tm.mu)
    stop_clock("U-ppRPA diagonalization: beta-beta")

    start_clock("U-ppRPA diagonalization: alpha-beta")
    exci_ab, xy_ab, _ = diagonalize_pprpa_subspace_diff_spin(
        nocc=nocc, mo_energy=[mf_mo_energy[0], mf_mo_energy[1]],
        Lpq=tm.Lpq, mu=tm.mu)
    stop_clock("U-ppRPA diagonalization: alpha-beta")

    tm.exci_aa = exci_aa
    tm.exci_bb = exci_bb
    tm.exci_ab = exci_ab
    tm.xy_aa = xy_aa
    tm.xy_bb = xy_bb
    tm.xy_ab = xy_ab

    start_clock("U-T-matrix transition density: alpha-alpha")
    tm.rho_aa = get_transition_density_same_spin(
        nocc=nocc[0], nvir=nvir[0], xy=xy_aa, Lpq=tm.Lpq[0])
    stop_clock("U-T-matrix transition density: alpha-alpha")

    start_clock("U-T-matrix transition density: beta-beta")
    tm.rho_bb = get_transition_density_same_spin(
        nocc=nocc[1], nvir=nvir[1], xy=xy_bb, Lpq=tm.Lpq[1])
    stop_clock("U-T-matrix transition density: beta-beta")

    start_clock("U-T-matrix transition density: alpha-beta")
    tm.rho_ab = get_transition_density_diff_spin(
        nocc=nocc, nvir=nvir, xy=xy_ab, Lpq=tm.Lpq)
    stop_clock("U-T-matrix transition density: alpha-beta")

    sigma = np.zeros((2, nmo[0], nmo[0]))
    derivative = np.zeros((2, nmo[0]))
    for spin, label in enumerate(('alpha', 'beta')):
        start_clock("U-T-matrix self-energy: %s" % label)
        sigma[spin] = _get_sigma_for_spin(
            tm, spin, mo_energy[spin])
        stop_clock("U-T-matrix self-energy: %s" % label)

        if tm.qpe_linearized:
            derivative[spin] = _get_sigma_derivative_for_spin(
                tm, spin, mo_energy[spin])

    if tm.qpe_linearized:
        z = 1.0 / (1.0 - derivative)
        if tm.qpe_linearized_range is not None:
            z = np.where(
                (z < tm.qpe_linearized_range[0])
                | (z > tm.qpe_linearized_range[1]),
                1.0,
                z,
            )
        mo_energy = (
            mf_mo_energy
            + z * np.diagonal(
                tm.vk + sigma - tm.vxc, axis1=1, axis2=2))
    else:
        for spin, label in enumerate(('alpha', 'beta')):
            def quasiparticle(qp_energy):
                sigma_spin = _get_sigma_for_spin(
                    tm, spin, qp_energy)
                return qp_energy - (
                    mf_mo_energy[spin]
                    + (sigma_spin + tm.vk[spin]
                       - tm.vxc[spin]).diagonal())

            try:
                mo_energy[spin] = scipy.optimize.newton(
                    quasiparticle,
                    mf_mo_energy[spin],
                    tol=tm.qpe_tol * nmo[spin],
                    maxiter=tm.qpe_max_iter,
                )
            except RuntimeError:
                print(
                    'WARNING: %s quasiparticle equation fails to converge!'
                    % label)

    tm.mo_energy = mo_energy
    print('\n  Unrestricted T-matrix QP energies (Hartree):')
    for spin, label in enumerate(('alpha', 'beta')):
        print('    %s spin:' % label)
        for i in range(nmo[spin]):
            marker = ' (HOMO)' if i == nocc[spin] - 1 else \
                     ' (LUMO)' if i == nocc[spin] else ''
            print('      MO %4d:  MF = %12.6f  QP = %12.6f%s' %
                  (i + 1, mf_mo_energy[spin, i],
                   mo_energy[spin, i], marker))
    print('')


# =============================================================================
# UTMatrix class
# =============================================================================

class UTMatrix(lib.StreamObject):
    """Unrestricted T-matrix method based on ppRPA.

    The mean-field reference must be a collinear UHF or UKS object with real
    orbitals and occupied orbitals ordered before virtual orbitals.
    """

    def __init__(self, mf, auxbasis=None):
        mf_mo_energy = np.asarray(mf.mo_energy)
        mf_mo_coeff = np.asarray(mf.mo_coeff)
        if mf_mo_energy.ndim != 2 or mf_mo_energy.shape[0] != 2:
            raise TypeError("UTMatrix requires a UHF or UKS reference.")
        if mf_mo_coeff.ndim != 3 or mf_mo_coeff.shape[0] != 2:
            raise TypeError("UTMatrix requires alpha and beta MO coefficients.")
        if mf_mo_energy.shape[1] != mf_mo_coeff.shape[2]:
            raise ValueError("MO energy and coefficient dimensions disagree.")

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        self.auxbasis = auxbasis

        self.eta = 5.0e-3
        self.vhf_df = True
        self.qpe_linearized = False
        self.qpe_linearized_range = [0.5, 1.5]
        self.qpe_max_iter = 100
        self.qpe_tol = 1.0e-6

        self._nocc = None
        self._nmo = None
        self.mo_energy = np.array(mf_mo_energy, copy=True)
        self.mo_coeff = np.array(mf_mo_coeff, copy=True)
        self.Lpq = None

        self.mu = None
        self.vk = None
        self.vxc = None
        self.exci_aa = None
        self.exci_bb = None
        self.exci_ab = None
        self.xy_aa = None
        self.xy_bb = None
        self.xy_ab = None
        self.rho_aa = None
        self.rho_bb = None
        self.rho_ab = None

    @property
    def nocc(self):
        if self._nocc is not None:
            return self._nocc
        return tuple(int(n) for n in self._scf.nelec)

    @nocc.setter
    def nocc(self, n):
        self._nocc = tuple(int(x) for x in n)

    @property
    def nmo(self):
        if self._nmo is not None:
            return self._nmo
        return tuple(len(e) for e in self._scf.mo_energy)

    @nmo.setter
    def nmo(self, n):
        self._nmo = tuple(int(x) for x in n)

    def dump_flags(self):
        nocc = self.nocc
        nmo = self.nmo
        print('')
        print('******** %s ********' % self.__class__)
        print('method = %s' % self.__class__.__name__)
        print('U-T-matrix nocc = %d (%d alpha, %d beta)' %
              (sum(nocc), nocc[0], nocc[1]))
        print('U-T-matrix nvir = %d (%d alpha, %d beta)' %
              (sum(nmo) - sum(nocc), nmo[0] - nocc[0],
               nmo[1] - nocc[1]))
        print('density-fitting for exchange = %s' % self.vhf_df)
        print('broadening parameter = %.3e' % self.eta)
        print('use perturbative linearized QP eqn = %s' %
              self.qpe_linearized)
        if self.qpe_linearized:
            print('linearized factor range = %s' %
                  self.qpe_linearized_range)
        else:
            print('QPE max iter = %d' % self.qpe_max_iter)
            print('QPE tolerance = %.1e' % self.qpe_tol)
        print('')

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

    def ao2mo(self, mo_coeff=None):
        """Transform density-fitting integrals for alpha and beta MOs."""
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = np.asarray(mo_coeff)

        if not hasattr(self, 'with_df'):
            self.initialize_df(auxbasis=self.auxbasis)

        naux = self.with_df.get_naoaux()
        Lpq = []
        for spin in range(2):
            nmo = self.nmo[spin]
            mo = np.asarray(mo_coeff[spin], order='F')
            ijslice = (0, nmo, 0, nmo)
            Lpq_spin = _ao2mo.nr_e2(
                self.with_df._cderi,
                mo,
                ijslice,
                aosym='s2',
                out=None,
            )
            Lpq.append(Lpq_spin.reshape(naux, nmo, nmo))
        return Lpq

    def kernel(self):
        """Run the unrestricted T-matrix calculation."""
        if self.Lpq is None:
            self.initialize_df(auxbasis=self.auxbasis)

        self.dump_flags()
        cput0 = (time.process_time(), time.perf_counter())
        kernel(self)
        cpu_time = time.process_time() - cput0[0]
        wall_time = time.perf_counter() - cput0[1]
        print('U-T-matrix CPU time: %.2f s, wall time: %.2f s' %
              (cpu_time, wall_time))

    def make_gf(self, omega, eta, fullsigma=True, mode='linear'):
        r"""Get the spin-resolved dynamical Green's function.

        Parameters match :meth:`lib_pprpa.tmatrix.TMatrix.make_gf`.  The
        returned arrays have shape ``(2, nw, nmo, nmo)`` with alpha first.
        """
        omega = np.asarray(omega)
        mo_energy = np.asarray(self._scf.mo_energy)
        sigma = np.asarray([
            _get_sigma_dynamic_for_spin(
                self, spin, omega, eta, fullsigma)
            for spin in range(2)
        ])
        gf0 = np.asarray([
            get_g0(omega, mo_energy[spin], eta)
            for spin in range(2)
        ])

        sigma_diff = np.array(sigma, copy=True)
        if fullsigma:
            sigma_diff += (self.vk - self.vxc)[:, None, :, :]
        else:
            for spin in range(2):
                for w in range(len(omega)):
                    for i in range(self.nmo[spin]):
                        sigma_diff[spin, w, i, i] += (
                            self.vk[spin, i, i]
                            - self.vxc[spin, i, i])

        if mode == 'linear':
            gf = gf0 + gf0 @ sigma_diff @ gf0
        elif mode == 'dyson':
            gf = np.linalg.inv(np.linalg.inv(gf0) - sigma_diff)
        else:
            raise ValueError("mode must be 'linear' or 'dyson'")

        return gf, gf0, sigma

    def make_diag_dos(self, omega, eta):
        """Get spin- and orbital-resolved DOS using diagonal self-energy."""
        omega = np.asarray(omega)
        mo_energy = np.asarray(self._scf.mo_energy)
        nw = len(omega)
        eta2 = (3.0 * eta) ** 2
        dos = np.zeros((2, self.nmo[0], nw))

        for spin in range(2):
            nmo = self.nmo[spin]
            sigma_real = np.zeros((nmo, nw))
            sigma_imag = np.zeros((nmo, nw))

            for nocc, energy_ref, exci, rho, oo_dim in \
                    _pair_channel_specs(self, spin):
                nroot = len(exci)
                chunk_size = 500

                if oo_dim > 0:
                    exci_hh = exci[:oo_dim]
                    rho_hh = rho[:oo_dim, :, nocc:]
                    for q_idx in range(len(energy_ref) - nocc):
                        q = nocc + q_idx
                        rho_q = rho_hh[:, :, q_idx]
                        for start in range(0, oo_dim, chunk_size):
                            end = min(start + chunk_size, oo_dim)
                            ediff = (
                                omega[None, :]
                                + energy_ref[q]
                                - 2.0 * self.mu
                                - exci_hh[start:end, None])
                            denom = ediff ** 2 + eta2
                            rho_sq = rho_q[start:end] ** 2
                            sigma_real += (
                                rho_sq.T @ (ediff / denom))
                            sigma_imag += (
                                rho_sq.T @ (-eta / denom))

                if nroot > oo_dim:
                    exci_pp = exci[oo_dim:]
                    rho_pp = rho[oo_dim:, :, :nocc]
                    vv_dim = nroot - oo_dim
                    for q in range(nocc):
                        rho_q = rho_pp[:, :, q]
                        for start in range(0, vv_dim, chunk_size):
                            end = min(start + chunk_size, vv_dim)
                            ediff = (
                                omega[None, :]
                                + energy_ref[q]
                                - 2.0 * self.mu
                                - exci_pp[start:end, None])
                            denom = ediff ** 2 + eta2
                            rho_sq = rho_q[start:end] ** 2
                            sigma_real += (
                                rho_sq.T @ (ediff / denom))
                            sigma_imag += (
                                rho_sq.T @ (eta / denom))

            vk_minus_vxc = (
                self.vk[spin] - self.vxc[spin]).diagonal()
            ereal = (
                omega[None, :]
                - mo_energy[spin, :, None]
                - sigma_real
                - vk_minus_vxc[:, None])
            dos[spin] = (
                np.abs(sigma_imag)
                / (ereal ** 2 + sigma_imag ** 2)
                / np.pi)

        return dos

    def energy_tot(self, nw=60):
        r"""Calculate the unrestricted T-matrix Galitskii-Migdal energy."""
        from pyscf.lib import temporary_env

        pts, wts = np.polynomial.legendre.leggauss(nw)
        freqs = (1.0 + pts) / (1.0 - pts)
        weights = wts * 2.0 / (1.0 - pts) ** 2
        omega = 1j * freqs + self.mu

        _, gf0, sigma = self.make_gf(
            omega=omega, eta=0, fullsigma=True, mode='linear')
        g0_sigma = np.einsum(
            'swpq,swqp,w->', gf0, sigma, weights)
        e_c = (g0_sigma / (2.0 * np.pi)).real

        dm = self._scf.make_rdm1()
        if (isinstance(self._scf, scf.uhf.UHF)
                and not isinstance(self._scf, dft.uks.UKS)):
            uhf = self._scf
        else:
            uhf = scf.UHF(self.mol)
        with temporary_env(uhf, verbose=0):
            e_hf = (
                uhf.energy_elec(dm=dm)[0]
                + self._scf.energy_nuc())
        e_tot = e_hf + e_c

        print('UHF energy@U-T-matrix density  = %.8f' % e_hf)
        print('U-T-matrix correlation energy = %.8f' % e_c)
        print('U-T-matrix total energy       = %.8f' % e_tot)
        return e_tot, e_hf, e_c


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 3
    mol.atom = [
        ['Be', (0.0, 0.0, 0.0)],
        ['H', (0.0, 0.0, 1.342)],
    ]
    mol.basis = 'def2-svp'
    mol.spin = 1
    mol.build()

    mf = dft.UKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()

    tm = UTMatrix(mf)
    tm.eta = 1.0e-5
    tm.qpe_linearized = True
    tm.kernel()
