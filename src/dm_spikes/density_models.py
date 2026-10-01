"""Importable factories for JSON profile jobs; all callables are built locally.

These are initial densities, not replacements for Eddington inversion or the
adiabatic map. Units are those of the scalar API (pc, Msun, km/s).
"""

import math


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def power_law(*, rho0, r0, gamma):
    """rho(r) = rho0 * (r/r0)**(-gamma), with 0 < gamma < 3."""
    rho0, r0 = _positive(rho0, "rho0"), _positive(r0, "r0")
    gamma = _positive(gamma, "gamma")
    if gamma >= 3:
        raise ValueError("gamma must be below 3 for locally integrable mass")

    def density(r):
        return rho0 * (r / r0) ** (-gamma)

    density.prime = lambda r: -gamma * density(r) / r
    density.second = lambda r: gamma * (gamma + 1) * density(r) / r / r
    return density


def nfw(*, rho_s, r_s):
    """rho(r) = rho_s / [x (1+x)^2], x = r/r_s."""
    rho_s, r_s = _positive(rho_s, "rho_s"), _positive(r_s, "r_s")

    def density(r):
        x = r / r_s
        return rho_s / x / (1 + x) / (1 + x)

    density.prime = lambda r: density(r) * (-1 / r - 2 / (r_s + r))
    density.second = lambda r: density(r) * (
        2 / r / r + 4 / r / (r_s + r) + 6 / (r_s + r) ** 2
    )
    return density


def hernquist(*, total_mass, scale_radius):
    """rho(r) = total_mass*a / [2*pi*r*(r+a)^3], a = scale_radius."""
    total_mass = _positive(total_mass, "total_mass")
    a = _positive(scale_radius, "scale_radius")

    def density(r):
        return total_mass / (2 * math.pi) / r * (a / (r + a)) / (r + a) / (r + a)

    density.prime = lambda r: density(r) * (-1 / r - 3 / (a + r))
    density.second = lambda r: density(r) * (
        2 / r / r + 6 / r / (a + r) + 12 / (a + r) ** 2
    )
    return density
