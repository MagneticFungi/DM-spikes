"""Shared density integration for final dark-matter profiles."""

import numpy as np
from scipy.integrate import quad

G = 4.30091e-3  # pc (km/s)^2 / Msun
C_LIGHT = 299792.458  # km/s


def schwarzschild_radius(M, G=G, clight=C_LIGHT):
    """Return the Schwarzschild radius in pc for mass M in Msun."""
    return 2.0 * G * M / clight**2


def energy_lower_bound(r, M, G=G, clight=C_LIGHT):
    """Return E'_m in (km/s)^2 at radius r in pc."""
    return -G * M / r * (1.0 - 4.0 * schwarzschild_radius(M, G=G, clight=clight) / r)


def angular_momentum_capture(M, G=G, clight=C_LIGHT):
    """Return L'_c in pc km/s for mass M in Msun."""
    return 2.0 * clight * schwarzschild_radius(M, G=G, clight=clight)


def angular_momentum_max(Ep, r, M, G=G):
    """Return L'_m in pc km/s for E' in (km/s)^2 and r in pc."""
    return np.sqrt(np.maximum(2.0 * r**2 * (Ep + G * M / r), 0.0))


def radial_velocity(Ep, Lp, r, M, G=G):
    """Return v_r in km/s for E', L', and r."""
    return np.sqrt(np.maximum(2.0 * (Ep + G * M / r - Lp**2 / (2.0 * r**2)), 0.0))


def rho_prime_at_r(f_prime, r, M, G=G, clight=C_LIGHT, epsabs=0.0, epsrel=1e-6, limit=100):
    """Return rho'(r) in Msun pc^-3 at radius r in pc."""
    if r <= 0.0:
        raise ValueError("r must be positive.")

    R_S = schwarzschild_radius(M, G=G, clight=clight)
    if r <= 4.0 * R_S:
        return 0.0

    E_m = energy_lower_bound(r, M, G=G, clight=clight)
    L_c = angular_momentum_capture(M, G=G, clight=clight)
    L_c2 = L_c**2

    def inner(Ep):
        L_m = angular_momentum_max(Ep, r, M, G=G)
        L_m2 = L_m**2
        if L_m2 <= L_c2:
            return 0.0

        u_max = np.sqrt(L_m2 - L_c2)

        def u_integrand(u):
            return f_prime(Ep, np.sqrt(np.maximum(L_m2 - u**2, 0.0)))

        value, _ = quad(u_integrand, 0.0, u_max, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return 4.0 * np.pi / r * value

    rho, _ = quad(inner, E_m, 0.0, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return rho


def rho_prime_profile(f_prime, r_array, M, **kwargs):
    """Return rho'(r) in Msun pc^-3 over radii r_array in pc."""
    return np.array([rho_prime_at_r(f_prime, r, M, **kwargs) for r in r_array])
