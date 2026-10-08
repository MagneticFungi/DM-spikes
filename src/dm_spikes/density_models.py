"""Importable factories for JSON profile jobs; all callables are built locally.

These are initial densities, not replacements for Eddington inversion or the
adiabatic map. Units are those of the scalar API (pc, Msun, km/s).
"""

import math

from .constants import G


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


class NFWPotential:
    """Finite-escape potential of an untruncated NFW halo in pc and km/s."""

    radial_center_allowed = True

    def __init__(self, *, rho_s, r_s, G=G):
        self.rho_s = _positive(rho_s, "rho_s")
        self.r_s = _positive(r_s, "r_s")
        self.gravity = _positive(G, "G")
        self.scale = 4.0 * math.pi * self.gravity * self.rho_s * self.r_s**2
        self.central_potential = -self.scale

    def _x(self, radius):
        radius = float(radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        return radius / self.r_s

    def __call__(self, radius):
        x = self._x(radius)
        if x < 1e-3:
            ratio = 1.0 + x * (-0.5 + x * (1.0 / 3.0 + x * (
                -0.25 + x * (0.2 + x * (-1.0 / 6.0 + x / 7.0)))))
        else:
            ratio = math.log1p(x) / x
        return -self.scale * ratio

    def offset_from_center(self, radius):
        """Return Phi(r) - Phi(0) without subtracting two large potentials."""
        x = self._x(radius)
        if x < 1e-3:
            return self.scale * x * (
                0.5 + x * (-1.0 / 3.0 + x * (0.25 + x * (
                    -0.2 + x * (1.0 / 6.0 - x / 7.0)
                )))
            )
        return self.scale * (1.0 - math.log1p(x) / x)

    def prime(self, radius):
        x = self._x(radius)
        if x < 1e-3:
            derivative = 0.5 + x * (-2.0 / 3.0 + x * (
                0.75 + x * (-0.8 + x * (5.0 / 6.0 - 6.0 * x / 7.0))))
        else:
            derivative = (math.log1p(x) - x / (1.0 + x)) / x**2
        return self.scale * derivative / self.r_s

    def second(self, radius):
        x = self._x(radius)
        if x < 1e-3:
            derivative = -2.0 / 3.0 + x * (1.5 + x * (
                -2.4 + x * (10.0 / 3.0 - 30.0 * x / 7.0)))
        else:
            moment = math.log1p(x) - x / (1.0 + x)
            derivative = 1.0 / (x * (1.0 + x)**2) - 2.0 * moment / x**3
        return self.scale * derivative / self.r_s**2


class HernquistPotential:
    """Initial Hernquist potential, with a stable offset from its center."""

    radial_center_allowed = True

    def __init__(self, *, total_mass, scale_radius, G=G):
        self.total_mass = _positive(total_mass, "total_mass")
        self.scale_radius = _positive(scale_radius, "scale_radius")
        self.gravity = _positive(G, "G")
        self.gm = self.gravity * self.total_mass
        self.central_potential = -self.gm / self.scale_radius

    def __call__(self, radius):
        radius = _positive(radius, "radius")
        return -self.gm / (radius + self.scale_radius)

    def offset_from_center(self, radius):
        radius = _positive(radius, "radius")
        return -self.central_potential * (radius / (radius + self.scale_radius))

    def prime(self, radius):
        denominator = _positive(radius, "radius") + self.scale_radius
        return self.gm / denominator / denominator

    def second(self, radius):
        denominator = _positive(radius, "radius") + self.scale_radius
        return -2.0 * self.gm / denominator / denominator / denominator


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
