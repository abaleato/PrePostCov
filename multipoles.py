import numpy as np
from typing import Callable, Dict, Iterable, Optional


class LegendreMultipoleExtractor:
    """
    Helper to project an anisotropic P(k, mu) onto Legendre multipoles P_ell(k).

    Usage patterns:
    - Preferred (accurate): pass a callable P(mu) -> Pk (1D array over k). The class will
      evaluate it at Gauss-Legendre nodes and use quadrature to integrate over mu in [-1, 1].
    - Fallback (data grid): pass a 2D array Pkmu with a corresponding mu_array. The class will
      integrate using the trapezoidal rule (assuming P is even in mu if mu_array in [0, 1]).

    Normalization: P_ell(k) = (2*ell+1)/2 * int_{-1}^{1} dmu P(k, mu) L_ell(mu)
    """

    def __init__(
        self,
        ngauss: int = 8,
    ) -> None:
        """
        Parameters
        - ngauss: number of positive-side Gauss-Legendre nodes to use (total nodes = 2*ngauss)
        """
        assert ngauss >= 2, "ngauss should be >= 2"
        self.ngauss = ngauss
        # Full-range Gauss-Legendre nodes/weights on [-1, 1]
        self.mu_full, self.w_full = np.polynomial.legendre.leggauss(2 * ngauss)
        # Positive half [0, 1] subset of nodes (ascending)
        self.mu_pos = self.mu_full[ngauss:]

    @staticmethod
    def _legendre_L(ell: int, mu):
        return np.polynomial.legendre.Legendre.basis(ell)(mu)

    def project(
        self,
        Pkmu,
        ells: Iterable[int] = (0, 2, 4),
        mu_array: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Project P(k, mu) onto Legendre multipoles.

        Parameters
        - Pkmu: either a callable f(mu) -> Pk (shape (nk,)) or a 2D array with shape (nmu, nk)
        - ells: iterable of even multipoles to compute, e.g., (0, 2, 4)
        - mu_array: if Pkmu is an array, the corresponding mu grid. If None and Pkmu is callable,
                    Gauss-Legendre nodes are used automatically.

        Returns
        - dict mapping ell -> P_ell(k) as a 1D array of length nk
        """
        ells = list(ells)
        if any(ell < 0 or int(ell) != ell for ell in ells):
            raise ValueError("Multipoles must be non-negative integers")

        # Case 1: callable -> use Gauss-Legendre quadrature for best accuracy
        if callable(Pkmu):
            # Evaluate P at positive mu nodes (assumes evenness in mu)
            P_pos = np.atleast_2d(np.array([Pkmu(mu) for mu in self.mu_pos]))  # (nmu_pos, nk)
            if P_pos.ndim != 2:
                raise ValueError("Callable must return a 1D array over k for each mu")
            # Build full-range values using evenness: P(-mu) = P(+mu)
            P_full = np.vstack((P_pos[::-1], P_pos))  # (2*ngauss, nk), order matches mu_full

            results: Dict[int, np.ndarray] = {}
            for ell in ells:
                L_full = self._legendre_L(ell, self.mu_full)[:, None]  # (2*ngauss, 1)
                pref = (2 * ell + 1) / 2.0
                P_ell = pref * np.sum(self.w_full[:, None] * L_full * P_full, axis=0)
                results[ell] = P_ell
            return results

        # Case 2: grid data -> use provided mu grid and trapz integration
        if mu_array is None:
            raise ValueError("mu_array must be provided when Pkmu is an array")

        mu_array = np.asarray(mu_array)
        P_arr = np.asarray(Pkmu)
        if P_arr.ndim != 2:
            raise ValueError("Pkmu array must be 2D with shape (nmu, nk)")
        if P_arr.shape[0] != mu_array.shape[0]:
            raise ValueError("First dimension of Pkmu must match length of mu_array")

        # Determine if mu_array spans [0,1] (half-range) or [-1,1] (full-range)
        half_range = (mu_array.min() >= 0) and (mu_array.max() <= 1)
        full_range = (mu_array.min() >= -1) and (mu_array.max() <= 1) and not half_range
        if not (half_range or full_range):
            raise ValueError("mu_array must lie within [0,1] or [-1,1]")

        results: Dict[int, np.ndarray] = {}
        for ell in ells:
            L_vals = self._legendre_L(ell, mu_array)[:, None]  # (nmu, 1)
            if half_range:
                # Assume even P(k, mu) and integrate over [0,1] with factor 2
                pref = (2 * ell + 1) / 2.0
                P_ell = 2 * pref * np.trapz(P_arr * L_vals, x=mu_array, axis=0)
            else:
                # Full-range integration over [-1,1]
                pref = (2 * ell + 1) / 2.0
                P_ell = pref * np.trapz(P_arr * L_vals, x=mu_array, axis=0)
            results[ell] = P_ell
        return results


def integrate_legendre_product_4(ells, ngauss: int = 16) -> float:
    """
    Compute I = ∫_{-1}^{1} dμ ∏_{i=1}^4 L_{ℓ_i}(μ), where L_ℓ is the Legendre polynomial.

    Uses Gauss–Legendre quadrature (accurate for smooth polynomials) and
    cheaply enforces obvious parity selection: if sum(ells) is odd, the integral is 0.

    Parameters
    - ells: iterable of 4 non-negative integers (ℓ1, ℓ2, ℓ3, ℓ4)
    - ngauss: number of Gauss nodes per half-range (uses 2*ngauss total)

    Returns
    - float integral value

    Notes
    - An exact analytic expression exists in terms of Wigner symbols via product expansions,
      but high-order closed forms are cumbersome. For EFT use-cases, Gauss rules are exact for
      polynomials up to degree 4*max(ells) with sufficiently large order; 2*ngauss nodes exactly
      integrate polynomials up to order 4*ngauss-1. Choose ngauss >= max(ells)+1 for exactness.
    """
    ells = list(ells)
    if len(ells) != 4:
        raise ValueError("ells must contain exactly four integers")
    if any((not isinstance(L, int)) or L < 0 for L in ells):
        raise ValueError("All ell must be non-negative integers")

    # Parity selection: product of four Legendre polynomials is even if sum(ell) is even.
    if (sum(ells) % 2) == 1:
        return 0.0

    # Quadrature
    mu, w = np.polynomial.legendre.leggauss(2 * ngauss)
    prod = np.ones_like(mu)
    for L in ells:
        prod *= np.polynomial.legendre.Legendre.basis(L)(mu)
    return float(np.sum(w * prod))
