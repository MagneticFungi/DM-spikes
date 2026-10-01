"""Direct isotropic Eddington inversion for a supplied density--potential pair.

``potential(r)`` must be monotone increasing, numerically stable, and use the
same energy zero as energies subsequently passed to the returned callable.
The density and potential must describe a smooth, physically admissible
isotropic system. Radii passed to either input callable are positive scalars.
No potential is inferred from the density, and no profile-specific formula is
used here.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq


_LOG_MIN = math.log(np.finfo(float).tiny) + 0.02
_LOG_MAX = math.log(np.finfo(float).max) - 0.02
_COEFFICIENT = 1.0 / (math.sqrt(8.0) * math.pi**2)


def make_eddington_df(
    density: Callable[[float], float],
    potential: Callable[[float], float],
    boundary: str,
    *,
    density_prime: Callable[[float], float] | None = None,
    density_second: Callable[[float], float] | None = None,
    potential_prime: Callable[[float], float] | None = None,
    potential_second: Callable[[float], float] | None = None,
    escape_slope: float = 0.0,
    central_potential: float | None = None,
    tail_survey_log_radius: float = 64.0,
) -> Callable[[float], float]:
    """Return ``f(E)`` for mechanical energy in the zero of ``potential``.

    ``finite_escape`` assumes ``potential(infinity) = 0`` and supports E < 0.
    Its surface term is ``-escape_slope/sqrt(-E)``, where ``escape_slope`` is
    ``(d density/d potential)`` at potential zero. The default zero is valid
    only when this derivative vanishes. ``confining`` assumes the potential
    tends to positive infinity, its outer derivative vanishes, and supports
    E > potential(0). A known finite ``potential(0)`` may be supplied as
    ``central_potential``; only then can energies at/below that endpoint be
    classified as outside the domain. Otherwise an unbracketed inner root
    raises an error, not a potentially incorrect zero. E >= 0 for
    ``finite_escape`` always returns zero.

    Pairs of optional first/second radial derivatives can improve accuracy.
    Otherwise five-point differences in log radius are used and checked at
    two step sizes. A failed derivative, root, quadrature, or tail-convergence
    check raises a descriptive error instead of returning an unreliable DF.
    The inversion is performed afresh at every E; no interpolation is used.
    The outer-tail check scans at least ``tail_survey_log_radius`` units in
    log(r/r_E), then requires two consecutive negligible segments. The
    default is 64. This is a convergence heuristic, not a proof against an
    arbitrarily remote feature in a generic density; increase it if the
    supplied profile has a known distant scale.
    """
    if boundary not in ("finite_escape", "confining"):
        raise ValueError("boundary must be 'finite_escape' or 'confining'")
    if not callable(density) or not callable(potential):
        raise TypeError("density and potential must be scalar callables")
    if (density_prime is None) != (density_second is None):
        raise ValueError("provide both density_prime and density_second, or neither")
    if (potential_prime is None) != (potential_second is None):
        raise ValueError("provide both potential_prime and potential_second, or neither")
    for derivative in (density_prime, density_second, potential_prime, potential_second):
        if derivative is not None and not callable(derivative):
            raise TypeError("optional derivatives must be callables")
    escape_slope = float(escape_slope)
    if not math.isfinite(escape_slope):
        raise ValueError("escape_slope must be finite")
    if boundary == "confining" and escape_slope != 0.0:
        raise ValueError("escape_slope applies only to finite_escape")
    if central_potential is not None:
        central_potential = float(central_potential)
        if not math.isfinite(central_potential):
            raise ValueError("central_potential must be finite")
    tail_survey_log_radius = float(tail_survey_log_radius)
    if not (math.isfinite(tail_survey_log_radius) and tail_survey_log_radius > 0.0):
        raise ValueError("tail_survey_log_radius must be positive and finite")

    def scalar(fn: Callable[[float], float], radius: float, name: str) -> float:
        if not (math.isfinite(radius) and radius > 0.0):
            raise ArithmeticError(f"{name}: radius left the positive float64 range")
        try:
            value = float(fn(radius))
        except (ArithmeticError, ValueError, TypeError, OverflowError) as exc:
            raise ArithmeticError(f"{name} could not be evaluated at r={radius:g}") from exc
        if not math.isfinite(value):
            raise ArithmeticError(f"{name} is non-finite at r={radius:g}")
        return value

    phi_at_one = scalar(potential, 1.0, "potential")
    if boundary == "finite_escape" and phi_at_one >= 0.0:
        raise ValueError("finite_escape requires a negative potential tending to zero")
    if scalar(density, 1.0, "density") < 0.0:
        raise ValueError("density must be nonnegative")
    if central_potential is not None and central_potential > phi_at_one:
        raise ValueError("central_potential cannot exceed potential(1)")

    def numerical_log_derivatives(
        fn: Callable[[float], float], radius: float, step: float, name: str
    ) -> tuple[float, float]:
        x = math.log(radius)
        if x - 2.0 * step <= _LOG_MIN or x + 2.0 * step >= _LOG_MAX:
            raise ArithmeticError(f"{name} derivatives exceed the float64 radius range")
        fm2 = scalar(fn, math.exp(x - 2.0 * step), name)
        fm1 = scalar(fn, math.exp(x - step), name)
        f0 = scalar(fn, radius, name)
        fp1 = scalar(fn, math.exp(x + step), name)
        fp2 = scalar(fn, math.exp(x + 2.0 * step), name)
        first = (fm2 - 8.0 * fm1 + 8.0 * fp1 - fp2) / (12.0 * step)
        second = (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * step**2)
        return first, second

    def log_derivatives(
        fn: Callable[[float], float],
        first_fn: Callable[[float], float] | None,
        second_fn: Callable[[float], float] | None,
        radius: float,
        step: float,
        name: str,
    ) -> tuple[float, float]:
        if first_fn is None:
            return numerical_log_derivatives(fn, radius, step, name)
        first = radius * scalar(first_fn, radius, name + "_prime")
        second = first + radius * radius * scalar(second_fn, radius, name + "_second")
        if not (math.isfinite(first) and math.isfinite(second)):
            raise ArithmeticError(f"{name} derivatives are non-finite at r={radius:g}")
        return first, second

    def second_density_in_potential(radius: float) -> tuple[float, float]:
        if density_prime is not None and potential_prime is not None:
            # Keep supplied radial derivatives in radial form. Converting them
            # to log-radius derivatives first can subtract two enormous,
            # nearly equal terms even when rho''(r) itself is well resolved.
            rho_r = scalar(density_prime, radius, "density_prime")
            rho_rr = scalar(density_second, radius, "density_second")
            phi_r = scalar(potential_prime, radius, "potential_prime")
            phi_rr = scalar(potential_second, radius, "potential_second")
            if phi_r <= 0.0:
                raise ArithmeticError(f"potential must increase with radius at r={radius:g}")
            correction = (rho_r / phi_r) * phi_rr
            if not math.isfinite(correction):
                raise ArithmeticError(f"radial derivative combination overflows at r={radius:g}")
            denominator = phi_r * phi_r
            if denominator == 0.0:
                raise ArithmeticError(f"potential derivative is below float64 resolution at r={radius:g}")
            second = math.fsum((rho_rr, -correction)) / denominator
            phi_x = radius * phi_r
            if not (math.isfinite(second) and math.isfinite(phi_x) and phi_x > 0.0):
                raise ArithmeticError(f"d²rho/dphi² is not resolved at r={radius:g}")
            return second, phi_x

        def quantities(step: float) -> tuple[float, float, float]:
            rho_x, rho_xx = log_derivatives(
                density, density_prime, density_second, radius, step, "density"
            )
            phi_x, phi_xx = log_derivatives(
                potential, potential_prime, potential_second, radius, step, "potential"
            )
            if not math.isfinite(phi_x) or phi_x <= 0.0:
                raise ArithmeticError(
                    f"potential must increase with radius; unresolved derivative at r={radius:g}"
                )
            numerator = rho_xx * phi_x - rho_x * phi_xx
            if not math.isfinite(numerator):
                raise ArithmeticError(f"density-potential derivatives overflow at r={radius:g}")
            return phi_x, numerator, abs(rho_xx * phi_x) + abs(rho_x * phi_xx)

        coarse = quantities(0.003)
        fine = quantities(0.0015)
        if abs(coarse[0] - fine[0]) > 0.003 * max(abs(coarse[0]), abs(fine[0])):
            raise ArithmeticError(
                f"numerical potential derivative is unreliable at r={radius:g}; "
                "supply derivative callables or a more stable potential"
            )
        scale = max(coarse[2], fine[2])
        resolved_numerator = max(abs(coarse[1]), abs(fine[1]), 1e-10 * scale)
        if abs(coarse[1] - fine[1]) > 0.003 * resolved_numerator:
            raise ArithmeticError(
                f"numerical density derivatives are unreliable at r={radius:g}; "
                "supply derivative callables or a smoother density"
            )
        phi_x, numerator, _ = coarse
        second = numerator / phi_x**3
        if not math.isfinite(second):
            raise ArithmeticError(f"d²rho/dphi² is non-finite at r={radius:g}")
        return second, phi_x

    def radius_at_energy(energy: float) -> float:
        residual = phi_at_one - energy
        if residual == 0.0:
            return 1.0
        direction = -1.0 if residual > 0.0 else 1.0
        previous_x = 0.0
        previous_phi = phi_at_one
        flat_steps = 0
        for index in range(1, 1025):
            x = direction * index * math.log(2.0)
            if x <= _LOG_MIN or x >= _LOG_MAX:
                if direction < 0.0:
                    raise ArithmeticError(
                        "could not resolve phi(r)=E toward the center; "
                        "provide central_potential to classify an out-of-domain energy"
                    )
                raise ArithmeticError("could not locate phi(r)=E before the float64 radius limit")
            current_phi = scalar(potential, math.exp(x), "potential")
            current_residual = current_phi - energy
            if current_residual == 0.0:
                if direction < 0.0 and x - math.log(2.0) > _LOG_MIN:
                    inward_phi = scalar(potential, math.exp(x - math.log(2.0)), "potential")
                    if inward_phi == energy:
                        raise ArithmeticError(
                            "phi(r)=E is unresolved on a central potential plateau at float64 precision"
                        )
                return math.exp(x)
            if (current_residual > 0.0) != (residual > 0.0):
                a, b = sorted((previous_x, x))
                root = brentq(
                    lambda log_r: scalar(potential, math.exp(log_r), "potential") - energy,
                    a, b, xtol=5e-15, rtol=4 * np.finfo(float).eps,
                )
                return math.exp(root)
            if direction < 0.0:
                scale = max(abs(current_phi), abs(energy), np.finfo(float).tiny)
                flat_steps = flat_steps + 1 if abs(current_phi - previous_phi) < 2e-15 * scale else 0
                if flat_steps >= 8:
                    raise ArithmeticError(
                        "potential values no longer resolve an inner root; "
                        "provide central_potential if E is outside the domain"
                    )
            previous_x, previous_phi = x, current_phi
        raise ArithmeticError("could not bracket phi(r)=E by geometric radius expansion")

    def distribution(energy: float) -> float:
        """Evaluate the initial isotropic DF at mechanical energy ``energy``."""
        energy = float(energy)
        if not math.isfinite(energy):
            raise ValueError("energy must be finite")
        if boundary == "finite_escape" and energy >= 0.0:
            return 0.0
        if central_potential is not None and energy <= central_potential:
            return 0.0
        radius_e = radius_at_energy(energy)
        log_radius_e = math.log(radius_e)
        max_x = _LOG_MAX - log_radius_e - 0.01
        if max_x <= 0.0:
            raise ArithmeticError("insufficient float64 radius range above phi(r)=E")
        at_root = second_density_in_potential(radius_e)
        root_phi = scalar(potential, radius_e, "potential")
        root_roundoff = np.finfo(float).eps * max(abs(root_phi), abs(energy), np.finfo(float).tiny)
        if max(root_roundoff, abs(root_phi - energy)) > 1e-5 * at_root[1]:
            raise ArithmeticError(
                "energy and the inner root are not resolved relative to the local potential slope"
            )

        def potential_log_slope(log_radius: float) -> float:
            radius = math.exp(log_radius)
            if potential_prime is not None:
                slope = radius * scalar(potential_prime, radius, "potential_prime")
            else:
                coarse, _ = numerical_log_derivatives(potential, radius, 0.003, "potential")
                fine, _ = numerical_log_derivatives(potential, radius, 0.0015, "potential")
                if abs(coarse - fine) > 0.003 * max(abs(coarse), abs(fine)):
                    raise ArithmeticError(
                        f"potential derivative is unresolved at r={radius:g}; "
                        "supply potential_prime or a more stable potential"
                    )
                slope = coarse
            if not (math.isfinite(slope) and slope > 0.0):
                raise ArithmeticError(f"potential slope is unresolved at r={radius:g}")
            return slope

        def local_potential_gap(delta_log_radius: float) -> float:
            # Integrate d phi / d(log r), instead of subtracting two nearby
            # values of phi. Three-point Gauss panels are used only where the
            # direct subtraction is ill-conditioned; this is not a DF model.
            panels = max(1, math.ceil(delta_log_radius / 0.05))
            if panels > 128:
                raise ArithmeticError(
                    "potential(r)-E remains unresolved too far from the turning radius"
                )
            panel_width = delta_log_radius / panels
            gauss_offset = math.sqrt(3.0 / 5.0)
            contributions = []
            for index in range(panels):
                midpoint = log_radius_e + (index + 0.5) * panel_width
                half_width = 0.5 * panel_width
                contributions.append(
                    half_width * (
                        5.0 / 9.0 * potential_log_slope(midpoint - gauss_offset * half_width)
                        + 8.0 / 9.0 * potential_log_slope(midpoint)
                        + 5.0 / 9.0 * potential_log_slope(midpoint + gauss_offset * half_width)
                    )
                )
            gap = math.fsum(contributions)
            if not (math.isfinite(gap) and gap > 0.0):
                raise ArithmeticError("local potential increment is not resolved")
            return gap

        # With log(r/r_E) = t², the 1/sqrt(phi(r)-E) endpoint singularity
        # becomes finite without evaluating potential at r=0 or r=infinity.
        def integrand(t: float) -> float:
            if t == 0.0:
                return 2.0 * math.sqrt(at_root[1]) * at_root[0]
            delta_log_radius = t * t
            radius = math.exp(log_radius_e + delta_log_radius)
            phi_at_radius = scalar(potential, radius, "potential")
            direct_gap = phi_at_radius - energy
            scale = max(abs(phi_at_radius), abs(energy), np.finfo(float).tiny)
            if direct_gap < -1e-10 * scale:
                raise ArithmeticError(
                    "potential(r)-E is negative beyond roundoff above its turning radius"
                )
            gap = (
                direct_gap if direct_gap > 1e-10 * scale
                else local_potential_gap(delta_log_radius)
            )
            second, phi_x = second_density_in_potential(radius)
            return 2.0 * t * phi_x * second / math.sqrt(gap)

        integral = 0.0
        quadrature_error = 0.0
        previous_x = 0.0
        small_segments = 0
        last_segment = 0.0
        last_x = 0.0
        for index in range(12):
            next_x = min(float(2**index), max_x)
            if next_x <= previous_x:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                try:
                    segment, error = quad(
                        integrand, math.sqrt(previous_x), math.sqrt(next_x),
                        epsabs=0.0, epsrel=2e-7, limit=120,
                    )
                except IntegrationWarning as exc:
                    raise ArithmeticError(
                        f"Eddington quadrature did not converge at E={energy:g}; "
                        "numerical derivatives or potential resolution may be insufficient"
                    ) from exc
            integral += segment
            quadrature_error += error
            last_segment = segment
            last_x = next_x
            if index >= 2 and abs(segment) <= 2e-7 * abs(integral):
                small_segments += 1
                if small_segments >= 2 and next_x >= tail_survey_log_radius:
                    break
            else:
                small_segments = 0
            previous_x = next_x
        else:
            raise ArithmeticError(f"Eddington integral tail did not converge at E={energy:g}")
        if small_segments < 2 or last_x < tail_survey_log_radius:
            raise ArithmeticError(
                f"Eddington integral tail was not resolved over the required "
                f"log-radius survey at E={energy:g}"
            )
        if boundary == "finite_escape":
            integral -= escape_slope / math.sqrt(-energy)
        if quadrature_error + 2.0 * abs(last_segment) > 2e-5 * abs(integral):
            raise ArithmeticError(
                f"Eddington integral or boundary-term cancellation is unresolved at E={energy:g}"
            )
        answer = _COEFFICIENT * integral
        if not math.isfinite(answer):
            raise ArithmeticError(f"Eddington DF is non-finite at E={energy:g}")
        return answer

    return distribution


__all__ = ["make_eddington_df"]
