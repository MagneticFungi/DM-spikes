"""Exact analytical profile generated from a pseudo-isothermal sphere."""

import numpy as np

from .constants import G


def _as_float_array(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _return_scalar_if_scalar(value, scalar):
    if scalar:
        return float(np.asarray(value))
    return value


def isothermal_matching_radius(M, sigma_v, G=G):
    """Return the matching radius of the analytical isothermal profile."""
    scalar = all(np.asarray(value).shape == () for value in (M, sigma_v, G))
    M = _as_float_array(M, "M")
    sigma_v = _as_float_array(sigma_v, "sigma_v")
    G = _as_float_array(G, "G")
    if np.any(M <= 0.0):
        raise ValueError("M must be positive.")
    if np.any(sigma_v <= 0.0):
        raise ValueError("sigma_v must be positive.")
    if np.any(G <= 0.0):
        raise ValueError("G must be positive.")

    result = (
        G
        * M
        / sigma_v**2
        * np.power(4.0 / (3.0 * np.sqrt(np.pi)), 2.0 / 3.0)
    )
    return _return_scalar_if_scalar(result, scalar)


def isothermal_profile(r_values, M, sigma_v, rho0, R_S, G=G):
    """Return the exact analytical isothermal profile in Msun pc^-3."""
    scalar = np.isscalar(r_values) or np.asarray(r_values).shape == ()
    r_values = _as_float_array(r_values, "r_values")
    if np.any(r_values <= 0.0):
        raise ValueError("r_values must be positive.")
    if M <= 0.0:
        raise ValueError("M must be positive.")
    if sigma_v <= 0.0:
        raise ValueError("sigma_v must be positive.")
    if rho0 < 0.0:
        raise ValueError("rho0 must be non-negative.")
    if R_S <= 0.0:
        raise ValueError("R_S must be positive.")
    if G <= 0.0:
        raise ValueError("G must be positive.")

    capture_factor = np.maximum(1.0 - 4.0 * R_S / r_values, 0.0)
    result = (
        4.0
        * rho0
        / (3.0 * np.sqrt(np.pi))
        * np.power(G * M / (r_values * sigma_v**2), 1.5)
        * np.power(capture_factor, 1.5)
    )
    return _return_scalar_if_scalar(result, scalar)
