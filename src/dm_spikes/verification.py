"""Verification routines for numerical and analytical density profiles."""
import numpy as np
from scipy.interpolate import CubicSpline

D = 8.5e3
# Cubic interpolation of the tabulated Gondolo-Silk alpha_gamma values.
alfa_spline = CubicSpline(np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 2.0]), np.array([0.0733, 0.12, 0.140, 0.142, 0.135, 0.122, 0.103, 0.0818, 0.0177]))

def isothermal_profile(r_values, M, sigma_v, rho0, R_S, G=4.30091e-3):
    """Analytical isothermal reference profile in Msun pc^-3."""
    r_values = np.asarray(r_values, dtype=float)

    return (
        4.0 * rho0 / (3.0 * np.sqrt(np.pi))
        * np.power(G * M / (r_values * sigma_v**2), 1.5)
        * np.power(np.maximum(1.0 - 4.0 * R_S / r_values, 0.0), 1.5)
    )

def alpha_gamma(gamma):
    """Interpolated alpha_gamma value for a given gamma."""

    if not (0.0 < gamma < 2.0):
        raise ValueError("Se requiere 0 < gamma < 2.")

    if gamma < 0.05:
        return 0.293 * np.power(gamma, 4.0/9.0)
    else:
        return float(alfa_spline(gamma))

def spike_radius(M, gamma, rho0, r0):
    """Spike radius for a given gamma, M, rho0, and r0."""
    alfa = alpha_gamma(gamma)
    return alfa * r0 * np.power(M / (rho0 * r0**3), 1.0 / (3.0 - gamma))

def rho_R(gamma, rho0, r0, R_sp):
    """Density at the spike radius for a given gamma."""
    return rho0 * np.power(R_sp / r0, -gamma)

def gamma_spike(gamma):
    """Spike slope for a given gamma."""
    return (9.0 - 2.0 * gamma) / (4.0 - gamma)

def rho_D(gamma):
    """Density at the solar radius for a given gamma."""
    return 0.0062 * (1.0 - gamma / 3.0)

def cusp_profile(r_values, M, gamma, R_S):
    """Analytical Gondolo-Silk spike approximation profile in Msun pc^-3."""
    r_values = np.asarray(r_values, dtype=float)

    rhoD = rho_D(gamma)
    R_sp = spike_radius(M, gamma, rhoD, D)
    gamma_sp = gamma_spike(gamma)
    rho_r = rho_R(gamma, rhoD, D, R_sp)
    g_gamma = np.maximum(1.0 - 4.0 * R_S / r_values,0.0,)**3

    return rho_r* g_gamma* np.power(R_sp / r_values, gamma_sp)

def initial_cusp_profile(r_values, gamma):
    """Initial GS cusp profile in Msun pc^-3 for radii in pc."""
    r_values = np.asarray(r_values, dtype=float)

    if np.any(r_values <= 0.0):
        raise ValueError("Todos los radios deben ser positivos.")
    if not (0.0 < gamma < 2.0):
        raise ValueError("Se requiere 0 < gamma < 2.")

    rhoD = rho_D(gamma)

    return rhoD * np.power(r_values / D, -gamma)

def isothermal_abs_error(numerical, analytical):
    """Absolute difference between numerical and analytical profiles."""
    return np.abs(numerical - analytical)

def cusp_rel_error(numerical, analytical):
    """Relative error, masked with NaN where the analytical profile is zero."""
    numerical = np.asarray(numerical, dtype=float)
    analytical = np.asarray(analytical, dtype=float)

    positive_mask = analytical > 0
    error = np.full_like(analytical, np.nan, dtype=float)
    error[positive_mask] = np.abs((numerical[positive_mask] - analytical[positive_mask])/analytical[positive_mask])
    return error
