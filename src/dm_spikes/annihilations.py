"""Dark-matter spike saturation due to self-annihilation."""
import numpy as np
from .verification import spike_radius, rho_R, gamma_spike

def rho_core(m, observable_sigma_v, bh_age=1e10):
    """Core density due to self-annihilation in Msun pc^-3."""
    if m <= 0.0:
        raise ValueError("m must be positive.")
    if observable_sigma_v <= 0.0:
        raise ValueError("observable_sigma_v must be positive.")
    if bh_age <= 0.0:
        raise ValueError("bh_age must be positive.")

    return m / (observable_sigma_v * bh_age)

def rho_spike(rho_cusp, m, observable_sigma_v, bh_age=1e10):
    rho_cusp = np.asarray(rho_cusp, dtype=float)

    rho_c = rho_core(m, observable_sigma_v, bh_age)

    return rho_cusp / (1.0 + rho_cusp / rho_c)

def core_radius(M, m, gamma, observable_sigma_v, bh_age=1e10):
    """Core radius due to self-annihilation in pc."""

    D = 8.5e3
    rhoD = 0.0062 * (1.0 - gamma / 3.0)
    R_sp = spike_radius(M, gamma, rhoD, D)
    gamma_sp = gamma_spike(gamma)
    rho_r = rho_R(gamma, rhoD, D, R_sp)
    rho_c = rho_core(m, observable_sigma_v, bh_age)
    
    return R_sp * (rho_r / rho_c) ** (1.0 / gamma_sp)
