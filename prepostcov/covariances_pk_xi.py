import numpy as np
from typing import Dict, Iterable, List, Optional
from scipy.special import spherical_jn

from .multipoles import LegendreMultipoleExtractor, integrate_legendre_product_4


class PkXiCovariance:
    """
    Compute disconnected covariance between P_ell(k) and xi_ell(r) using a P_cross(k, mu).

    The covariance block for given (ell1, ell2) is
        Cov[ P_{ell1}(k), xi_{ell2}(r) ] = ((2*ell1+1)*(2*ell2+1) / V) * jbar_{ell2}(k, r) *
                                            sum_{L1,L2 in poles} i^{ell2} I_{ell1,ell2,L1,L2} * P_{L1}(k) * P_{L2}(k)
    where I is the integral of four Legendre polynomials over mu in [-1, 1] and
    jbar_{ell2}(k, r) is the r-bin-averaged spherical Bessel j_{ell2}(k r).

    Parameters
    - k: 1D np.ndarray of k values (nk,)
    - r: 1D np.ndarray of r-bin centers (nr,)
    - V: survey volume
    - poles_Pk: list of P multipoles to include (even ells), e.g., [0, 2, 4]
    - poles_Xi: list of xi multipoles to include (even ells), e.g., [0, 2, 4]
    - ngauss_leg_product: Gauss-Legendre order (half-range) for 4-Legendre integral; 2*ngauss used total
    - r_avg_nsamp: samples for r averaging per bin
    """

    def __init__(
        self,
        k: np.ndarray,
        r: np.ndarray,
        V: float,
        poles_Pk: Iterable[int] = (0, 2, 4),
        poles_Xi: Iterable[int] = (0, 2, 4),
        ngauss_leg_product: int = 32,
        r_avg_nsamp: int = 100,
    ) -> None:
        self.k = np.asarray(k)
        self.r = np.asarray(r)
        self.V = float(V)
        self.poles_Pk = list(poles_Pk)
        self.poles_Xi = list(poles_Xi)
        self.ngauss_leg_product = int(ngauss_leg_product)
        self.r_avg_nsamp = int(r_avg_nsamp)

        # Storage for P_cross multipoles: dict ell -> array(nk,)
        self.Pcross_ells: Optional[Dict[int, np.ndarray]] = None

        # Precompute 4-Legendre integrals for all combinations
        self._I4 = self._precompute_I4()

        # Precompute r-averaging grids per bin
        self._rr_grid = self._build_r_window_grid()

    # ----- Public API -----
    def set_Pcross_from_callable(self, P_of_mu, ells: Iterable[int] = (0, 2, 4), ngauss_eval: int = 8) -> None:
        """Compute and store P_cross multipoles from a callable P(k, mu)."""
        ext = LegendreMultipoleExtractor(ngauss=ngauss_eval)
        self.Pcross_ells = ext.project(P_of_mu, ells=tuple(ells))

    def set_Pcross_from_dict(self, Pcross_ells: Dict[int, np.ndarray]) -> None:
        """Provide precomputed P_cross multipoles as a dict ell -> P_ell(k)."""
        # Basic validation
        nk = self.k.shape[0]
        for ell, arr in Pcross_ells.items():
            arr = np.asarray(arr)
            if arr.shape[0] != nk:
                raise ValueError(f"P_cross ell={ell} has length {arr.shape[0]} != nk={nk}")
        self.Pcross_ells = {int(k): np.asarray(v) for k, v in Pcross_ells.items()}

    def compute_covariance(self) -> np.ndarray:
        """
        Return cov_Pkell_Xiell with shape (len(poles_Pk), len(poles_Xi), nk, nr).
        Requires Pcross_ells to be set.
        """
        if self.Pcross_ells is None:
            raise RuntimeError("Pcross_ells not set. Call set_Pcross_from_callable or set_Pcross_from_dict first.")

        nk = self.k.shape[0]
        nr = self.r.shape[0]
        cov = np.zeros((len(self.poles_Pk), len(self.poles_Xi), nk, nr))

        # Precompute P_ell arrays for all Pk poles
        P_arrays = {ell: np.asarray(self.Pcross_ells[ell]) for ell in self.poles_Pk}

        # Precompute jmean_kr for each xi multipole (returns (nr,))
        jmean_cache: Dict[int, np.ndarray] = {}
        for ell2 in self.poles_Xi:
            jmean_cache[ell2] = self._jmean_kr(ell2)  # (nr,)

        # Loop over desired covariance blocks
        for i, ell1 in enumerate(self.poles_Pk):
            for j, ell2 in enumerate(self.poles_Xi):
                # Sum over L1, L2
                S_k = np.zeros(nk)
                for L1 in self.poles_Pk:
                    P1 = P_arrays[L1]
                    for L2 in self.poles_Pk:
                        P2 = P_arrays[L2]
                        I = self._I4[(ell1, ell2, L1, L2)]  # scalar
                        S_k += I * (P1 * P2)
                # Prefactor depends on r only (via jmean), and on ells, V
                pref = 1.j**ell2 * ((2 * ell1 + 1) * (2 * ell2 + 1) / self.V) * jmean_cache[ell2]  # (nr,)
                cov[i, j, :, :] = S_k[:, None] * pref[None, :]

        return cov

    # ----- Internal helpers -----
    def _precompute_I4(self) -> Dict[tuple, float]:
        I4: Dict[tuple, float] = {}
        for ell1 in self.poles_Pk:
            for ell2 in self.poles_Xi:
                for L1 in self.poles_Pk:
                    for L2 in self.poles_Pk:
                        I = integrate_legendre_product_4((ell1, ell2, L1, L2), ngauss=self.ngauss_leg_product)
                        I4[(ell1, ell2, L1, L2)] = I
        return I4

    def _build_r_window_grid(self) -> np.ndarray:
        """Build an r-grid for averaging around each r-bin center, shape (ns, nr)."""
        r = self.r  # (nr,)
        nr = r.shape[0]
        ns = self.r_avg_nsamp
        if nr > 1:
            r_step = float(np.mean(np.diff(r)))
        else:
            r_step = 0.0
        # Uniform samples in [r_j - r_step, r_j + r_step] for each r_j
        t = np.linspace(-1.0, 1.0, ns)[:, None]  # (ns,1)
        rr = r[None, :] + t * r_step  # (ns, nr)
        return rr

    def _jmean_kr(self, ell: int) -> np.ndarray:
        """Compute ⟨ j_ell(k r) ⟩_r over the r-window; returns array (nk, nr)."""
        k = self.k  # (nk,)
        rr = self._rr_grid  # (ns, nr)
        ns, nr = rr.shape

        # Denominator per r-bin: ∫ rr^2 drr over the window
        den = np.trapz(rr ** 2, x=rr, axis=0)  # (nr,)

        # Numerator per r-bin and k-bin
        num = np.zeros((k.shape[0], nr))
        for j in range(nr):
            r_col = rr[:, j]  # (ns,)
            y = spherical_jn(ell, k[:, None] * r_col[None, :]) * (r_col[None, :] ** 2)  # (nk, ns)
            num[:, j] = np.trapz(y, x=r_col, axis=1)  # (nk,)

        jmean = num / den[None, :]
        return jmean  # (nk, nr)
