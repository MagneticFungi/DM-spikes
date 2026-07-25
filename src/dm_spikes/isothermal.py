"""Finite-core isothermal model with a constant distribution function."""

import numpy as np

from .density import rho_prime_profile


def make_isothermal_distribution(rho0, sigma_v):
    """Return constant f'(E', L') in Msun pc^-3 (km/s)^-3."""
    f0 = rho0 * (2 * np.pi * sigma_v**2) ** (-1.5)

    def f_prime(Ep, Lp):
        return f0

    return f_prime


def compute_isothermal_profile(r_values, M, rho0, sigma_v, **integration_kwargs):
    """Return the isothermal rho'(r) profile in Msun pc^-3 for radii in pc."""
    f_prime = make_isothermal_distribution(rho0, sigma_v)
    return rho_prime_profile(f_prime, r_values, M, **integration_kwargs)
