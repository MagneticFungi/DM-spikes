import numpy as np
from .verification import rho_D, rho_R, spike_radius, gamma_spike

def g_gamma(density_profile, r_values, gamma, M=2.6e6, D=8.5e3):
    density_profile = np.asarray(density_profile, dtype=float)
    r_values = np.asarray(r_values, dtype=float)
    rhoD = rho_D(gamma)
    Rsp = spike_radius(M, gamma, rhoD, D)
    rhoR = rho_R(gamma, rhoD, D, Rsp)
    gamma_sp = gamma_spike(gamma)

    return np.asarray(density_profile / ( rhoR * (Rsp / r_values)**gamma_sp))