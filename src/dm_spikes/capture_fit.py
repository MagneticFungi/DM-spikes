"""Utilities for fitted capture-factor reconstructions."""

import numpy as np
from numpy.polynomial.chebyshev import chebval


def gamma_hat(gamma, gamma_domain):
    """Scale gamma values to the Chebyshev domain [-1, 1]."""
    gamma = np.asarray(gamma, dtype=float)
    gamma_min, gamma_max = np.asarray(gamma_domain, dtype=float)
    if not np.isfinite(gamma_min) or not np.isfinite(gamma_max):
        raise ValueError("gamma_domain must contain finite values.")
    if gamma_max <= gamma_min:
        raise ValueError("gamma_domain must satisfy gamma_max > gamma_min.")
    return 2.0 * (gamma - gamma_min) / (gamma_max - gamma_min) - 1.0


def _get_fit_value(fit_parameters, key, default=None):
    if hasattr(fit_parameters, "files") and key in fit_parameters.files:
        return fit_parameters[key]
    if isinstance(fit_parameters, dict) and key in fit_parameters:
        return fit_parameters[key]
    return default


def fitted_A_B(gamma, fit_parameters):
    """Evaluate A(gamma), B(gamma), and optional C(gamma)."""
    gamma_domain = _get_fit_value(fit_parameters, "gamma_domain")
    coeffs_logA = _get_fit_value(fit_parameters, "cheb_coeffs_logA")
    coeffs_logB = _get_fit_value(fit_parameters, "cheb_coeffs_logB")
    coeffs_C = _get_fit_value(fit_parameters, "cheb_coeffs_C", None)

    if gamma_domain is None or coeffs_logA is None or coeffs_logB is None:
        raise ValueError(
            "fit_parameters must contain gamma_domain, cheb_coeffs_logA, "
            "and cheb_coeffs_logB."
        )

    gh = gamma_hat(gamma, gamma_domain)
    A = np.exp(chebval(gh, np.asarray(coeffs_logA, dtype=float)))
    B = np.exp(chebval(gh, np.asarray(coeffs_logB, dtype=float)))
    C = None
    if coeffs_C is not None and np.asarray(coeffs_C).size:
        C = chebval(gh, np.asarray(coeffs_C, dtype=float))
    return A, B, C


def fitted_capture_factor(r, gamma, R_S, fit_parameters):
    """Return the fitted capture factor g_fit(r, gamma).

    Parameters
    ----------
    r, gamma : scalar or array_like
        Radius in pc and initial cusp slope. Inputs are broadcast together.
    R_S : float
        Schwarzschild radius in pc. Must be positive.
    fit_parameters : mapping or numpy.lib.npyio.NpzFile
        Parameters containing ``gamma_domain``, ``cheb_coeffs_logA`` and
        ``cheb_coeffs_logB``. If ``cheb_coeffs_C`` is present, the alternative
        model A(gamma) x**B(gamma) exp(C(gamma) * (1 - x)) is evaluated.

    Returns
    -------
    numpy.ndarray or numpy scalar
        Fitted capture factor. Values at r <= 4 R_S are set to zero.
    """
    if not np.isfinite(R_S) or R_S <= 0.0:
        raise ValueError("R_S must be a positive finite scalar.")

    r_arr = np.asarray(r, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    x = 1.0 - 4.0 * float(R_S) / r_arr
    A, B, C = fitted_A_B(gamma_arr, fit_parameters)
    x, A, B = np.broadcast_arrays(x, A, B)
    if C is None:
        C = np.zeros_like(x, dtype=float)
    else:
        C = np.broadcast_to(C, x.shape)

    out = np.zeros_like(x, dtype=float)
    mask = np.isfinite(x) & np.isfinite(A) & np.isfinite(B) & np.isfinite(C) & (x > 0.0)
    out[mask] = np.exp(np.log(A[mask]) + B[mask] * np.log(x[mask]) + C[mask] * (1.0 - x[mask]))
    return out
