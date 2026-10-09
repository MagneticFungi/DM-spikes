"""Final dark-matter density from an adiabatically mapped isotropic DF.

The final Kepler potential, E'=-GM_bh/r+v²/2, and the approximate capture
cut L'_c=4GM_bh/c are the same phase-space model used by the former density
module. The initial potential is supplied separately and contains no black
hole. Its energy zero must match the supplied initial distribution function.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from .adiabatic_map import solve_initial_energy
from .constants import C_LIGHT, G


def _positive_finite(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return value


def schwarzschild_radius(M: float, G: float = G, clight: float = C_LIGHT) -> float:
    """Schwarzschild radius 2GM/c² in pc for M in Msun."""
    mass = _positive_finite(M, "M")
    gravity = _positive_finite(G, "G")
    light_speed = _positive_finite(clight, "clight")
    result = (2.0 * gravity * mass / light_speed) / light_speed
    if not math.isfinite(result) or result <= 0.0:
        raise ArithmeticError("Schwarzschild radius is outside the float64 range")
    return result


def energy_lower_bound(r: float, M: float, G: float = G, clight: float = C_LIGHT) -> float:
    """Lowest allowed final Kepler energy E'_m at a radius outside 4R_S."""
    radius = _positive_finite(r, "r")
    mass = _positive_finite(M, "M")
    gravity = _positive_finite(G, "G")
    capture_radius = 4.0 * schwarzschild_radius(mass, G=gravity, clight=clight)
    return -gravity * mass / radius * (1.0 - capture_radius / radius)


def angular_momentum_capture(M: float, G: float = G, clight: float = C_LIGHT) -> float:
    """Approximate capture threshold L'_c=4GM/c in pc km/s."""
    mass = _positive_finite(M, "M")
    gravity = _positive_finite(G, "G")
    light_speed = _positive_finite(clight, "clight")
    result = 4.0 * gravity * mass / light_speed
    if not math.isfinite(result):
        raise ArithmeticError("capture angular momentum is outside the float64 range")
    return result


def angular_momentum_max(Ep: float, r: float, M: float, G: float = G) -> float:
    """Largest local L' for final energy E' at radius r."""
    return np.sqrt(np.maximum(2.0 * r**2 * (Ep + G * M / r), 0.0))


def radial_velocity(Ep: float, Lp: float, r: float, M: float, G: float = G) -> float:
    """Radial speed for final Kepler energy and angular momentum."""
    return np.sqrt(np.maximum(2.0 * (Ep + G * M / r - Lp**2 / (2.0 * r**2)), 0.0))


def make_final_distribution(
    initial_df: Callable[[float], float],
    initial_potential: Callable[[float], float],
    M_bh: float,
    *,
    G: float = G,
    map_kwargs: dict | None = None,
) -> Callable[[float, float], float]:
    """Return f'(E',L') = f_i(E(E',L')) using the general adiabatic map.

    The initial DF receives the mapped *mechanical* energy directly, with
    the same energy zero as ``initial_potential``. ``map_kwargs`` passes
    solver settings to ``solve_initial_energy``; it cannot replace G or any
    of the four required inputs. Mapping errors propagate rather than being
    interpreted as zero phase-space density.
    """
    if not callable(initial_df) or not callable(initial_potential):
        raise TypeError("initial_df and initial_potential must be callable")
    mass = _positive_finite(M_bh, "M_bh")
    gravity = _positive_finite(G, "G")
    if map_kwargs is None:
        options = {}
    elif isinstance(map_kwargs, dict):
        options = dict(map_kwargs)
    else:
        raise TypeError("map_kwargs must be a dictionary or None")
    if any(key in options for key in ("G", "E_prime", "L_prime", "M_bh", "potential")):
        raise ValueError("map_kwargs cannot replace G or the mapping inputs")

    def final_distribution(E_prime: float, L_prime: float) -> float:
        initial_energy = solve_initial_energy(
            E_prime, L_prime, mass, initial_potential, G=gravity, **options
        )
        try:
            value = float(initial_df(initial_energy))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ArithmeticError(
                f"initial DF cannot be evaluated at mapped energy E={initial_energy:g}"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ArithmeticError(
                f"initial DF must be finite and nonnegative at E={initial_energy:g}"
            )
        return value

    return final_distribution


def rho_prime_at_r(
    f_prime: Callable[[float, float], float],
    r: float,
    M: float,
    G: float = G,
    clight: float = C_LIGHT,
    epsabs: float = 0.0,
    epsrel: float = 1e-6,
    limit: int = 100,
) -> float:
    """Integrate a supplied final DF over noncaptured, bound Kepler orbits."""
    if not callable(f_prime):
        raise TypeError("f_prime must be callable")
    radius = _positive_finite(r, "r")
    mass = _positive_finite(M, "M")
    gravity = _positive_finite(G, "G")
    light_speed = _positive_finite(clight, "clight")
    if not (math.isfinite(epsabs) and epsabs >= 0.0):
        raise ValueError("epsabs must be finite and nonnegative")
    if not (math.isfinite(epsrel) and epsrel > 5e-14):
        raise ValueError("epsrel must be finite and exceed the float64 quadrature floor")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    R_S = schwarzschild_radius(mass, G=gravity, clight=light_speed)
    if radius <= 4.0 * R_S:
        return 0.0

    E_m = energy_lower_bound(radius, mass, G=gravity, clight=light_speed)
    L_c = angular_momentum_capture(mass, G=gravity, clight=light_speed)
    L_c2 = L_c * L_c

    def integrate(integrand: Callable[[float], float], lower: float, upper: float) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                result, error = quad(
                    integrand, lower, upper, epsabs=epsabs,
                    epsrel=epsrel, limit=limit,
                )
            except IntegrationWarning as exc:
                raise ArithmeticError("final-density phase-space quadrature did not converge") from exc
        if not (math.isfinite(result) and math.isfinite(error)):
            raise ArithmeticError("final-density quadrature returned a non-finite result")
        if error > max(epsabs, epsrel * abs(result)):
            raise ArithmeticError("final-density quadrature error exceeds the requested tolerance")
        return result

    def inner(Ep: float) -> float:
        L_m2 = 2.0 * radius**2 * (Ep + gravity * mass / radius)
        excess = L_m2 - L_c2
        if excess <= 0.0:
            return 0.0
        u_max = math.sqrt(excess)

        def u_integrand(u: float) -> float:
            angular_momentum = math.sqrt(max(L_m2 - u * u, 0.0))
            value = float(f_prime(Ep, angular_momentum))
            if not math.isfinite(value) or value < 0.0:
                raise ArithmeticError("final DF must be finite and nonnegative")
            return value

        return 4.0 * math.pi / radius * integrate(u_integrand, 0.0, u_max)

    result = integrate(inner, E_m, 0.0)
    if result < 0.0:
        raise ArithmeticError("final density is negative")
    return result


def rho_prime_profile(
    f_prime: Callable[[float, float], float],
    r_array,
    M: float,
    **kwargs,
) -> np.ndarray:
    """Evaluate the final density on an array of positive radii."""
    radii = np.asarray(r_array, dtype=float)
    return np.asarray(
        [rho_prime_at_r(f_prime, float(radius), M, **kwargs) for radius in radii.flat]
    ).reshape(radii.shape)


def make_final_profile(
    initial_df: Callable[[float], float],
    initial_potential: Callable[[float], float],
    M_bh: float,
    *,
    G: float = G,
    clight: float = C_LIGHT,
    epsabs: float = 0.0,
    epsrel: float = 1e-6,
    limit: int = 100,
    map_kwargs: dict | None = None,
) -> Callable[[float], float]:
    """Return rho'(r) by composing the adiabatic map and final DF integral.

    Build ``initial_potential`` with ``make_initial_potential`` and
    ``initial_df`` with ``make_eddington_df`` first. This direct inversion is
    intentionally untabulated and can be expensive at many radii.
    """
    f_prime = make_final_distribution(
        initial_df, initial_potential, M_bh, G=G, map_kwargs=map_kwargs
    )

    def final_density(radius: float) -> float:
        return rho_prime_at_r(
            f_prime, radius, M_bh, G=G, clight=clight,
            epsabs=epsabs, epsrel=epsrel, limit=limit,
        )

    return final_density


__all__ = [
    "schwarzschild_radius",
    "energy_lower_bound",
    "angular_momentum_capture",
    "angular_momentum_max",
    "radial_velocity",
    "make_final_distribution",
    "rho_prime_at_r",
    "rho_prime_profile",
    "make_final_profile",
]
