"""Spherical initial gravitational potential built from a supplied density.

This module contains no black-hole contribution and no profile-specific
formula. The density must be nonnegative, locally integrable at the center,
and defined at all positive radii visited by the quadratures. For a finite
escape boundary, the outer shell integral must converge; for a confining
boundary, the resulting potential must grow without bound. These asymptotic
properties cannot be proved from finitely many evaluations of a generic
callable. In particular, a remote narrow feature can escape any numerical
tail survey, so the returned potential is only as trustworthy as the supplied
density's stated global behavior.
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.interpolate import CubicHermiteSpline

from .constants import G as DEFAULT_G


_LOG_MAX = math.log(np.finfo(float).max) - 0.02
_LOG_MIN = math.log(np.finfo(float).tiny) + 0.02
_QUAD_RTOL = 2e-9
_TAIL_RTOL = 1e-8
_MIN_TAIL_LOG_SPAN = 64.0
_MAX_TAIL_LOG_STEP = 32.0
_MASS_TAIL_RTOL = 2e-10
_MASS_LOG_STEP = 2.0
_MASS_MIN_LOG_SPAN = 64.0
_LOG_SMALLEST_RADIUS = math.log(np.nextafter(0.0, 1.0)) + 2.0
_SECOND_RTOL = 1e-5


def _positive_radius(value: float) -> float:
    try:
        radius = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("radius must be a positive finite scalar") from exc
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("radius must be a positive finite scalar")
    return radius


class _InitialPotential:
    """Callable potential with first and second radial derivatives."""

    radial_center_allowed = True

    def __init__(
        self,
        density: Callable[[float], float],
        boundary: str,
        gravitational_constant: float,
        reference_radius: float | None,
    ) -> None:
        self._density = density
        self._boundary = boundary
        self._G = gravitational_constant
        self._r_ref = reference_radius
        # A represented density is smooth within each interpolation interval.
        # Share its knots with every Poisson quadrature, including direct
        # evaluations outside the potential table in a fresh radial worker.
        self._density_breakpoints = tuple(sorted({
            _positive_radius(radius)
            for radius in getattr(density, "integration_breakpoints", ())
        }))
        self._log_density_breakpoints = tuple(
            math.log(radius) for radius in self._density_breakpoints
        )
        self._mass = lru_cache(maxsize=4096)(self._mass_uncached)
        self._outer_shell = lru_cache(maxsize=2048)(self._outer_shell_uncached)
        self._mass_anchor_radii: list[float] = []
        self._mass_anchor_values: list[float] = []
        self._shell_anchor_radii: list[float] = []
        self._shell_anchor_values: list[float] = []
        self._m_ref = self._mass_direct(reference_radius) if reference_radius is not None else None
        if reference_radius is not None:
            self._remember_mass(reference_radius, self._m_ref)

    def _rho(self, radius: float) -> float:
        if not (math.isfinite(radius) and radius > 0.0):
            raise ArithmeticError("density quadrature left the positive float64 radius range")
        try:
            value = float(self._density(radius))
        except (ArithmeticError, TypeError, ValueError, OverflowError) as exc:
            raise ArithmeticError(f"density cannot be evaluated at r={radius:g}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ArithmeticError(f"density must be finite and nonnegative at r={radius:g}")
        return value

    @staticmethod
    def _integrate(
        integrand: Callable[[float], float], a: float, b: float, name: str,
        *, breakpoints: tuple[float, ...] = (),
    ) -> float:
        """Integrate separately on pieces delimited by sorted breakpoints."""
        if a == b:
            return 0.0
        if b < a:
            return -_InitialPotential._integrate(
                integrand, b, a, name, breakpoints=breakpoints,
            )
        start = bisect_right(breakpoints, a)
        stop = bisect_left(breakpoints, b)
        edges = (a, *breakpoints[start:stop], b)
        terms = []
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            for left, right in zip(edges[:-1], edges[1:]):
                if left == right:
                    continue
                try:
                    value, error = quad(
                        integrand, left, right, epsabs=0.0, epsrel=_QUAD_RTOL, limit=200,
                    )
                except IntegrationWarning as exc:
                    raise ArithmeticError(
                        f"{name} did not converge on integration coordinates "
                        f"[{left:.17g}, {right:.17g}]: {exc}"
                    ) from exc
                if not (math.isfinite(value) and math.isfinite(error)):
                    raise ArithmeticError(f"{name} returned a non-finite value or error estimate")
                if error > 1e-7 * abs(value):
                    raise ArithmeticError(f"{name} has an unresolved quadrature error")
                terms.append(value)
        return math.fsum(terms)

    def _unit_density_breakpoints(self, a: float, b: float) -> tuple[float, ...]:
        """Density knots mapped onto t in [0, 1], where r = a + (b-a)*t."""
        if a == b:
            return ()
        start = bisect_right(self._density_breakpoints, min(a, b))
        stop = bisect_left(self._density_breakpoints, max(a, b))
        knots = self._density_breakpoints[start:stop]
        if b < a:
            knots = knots[::-1]
        return tuple((radius - a) / (b - a) for radius in knots)

    def _log_mass_moment(self, a: float, b: float) -> float:
        """Positive integral of s² rho(s) ds, with bounded log-radius panels."""
        if a == b:
            return 0.0
        log_a, log_b = math.log(a), math.log(b)
        count = max(1, math.ceil((log_b - log_a) / _MASS_LOG_STEP))
        width = (log_b - log_a) / count

        def integrand(log_radius: float) -> float:
            shell_radius = math.exp(log_radius)
            return shell_radius * (shell_radius * self._rho(shell_radius)) * shell_radius

        return math.fsum(
            self._integrate(
                integrand, log_a + index * width, log_a + (index + 1) * width,
                "enclosed mass shell",
                breakpoints=self._log_density_breakpoints,
            )
            for index in range(count)
        )

    def _mass_direct(self, radius: float) -> float:
        # Survey inward from r in short log-radius panels. A single integral
        # after s=r*t² misses mass concentrated many decades below r.
        log_radius = math.log(radius)
        total = 0.0
        small_panels = 0
        upper_log = log_radius
        for _ in range(512):
            lower_log = max(_LOG_SMALLEST_RADIUS, upper_log - _MASS_LOG_STEP)
            if lower_log >= upper_log:
                break
            contribution = self._log_mass_moment(math.exp(lower_log), math.exp(upper_log))
            total = math.fsum((total, contribution))
            if contribution <= _MASS_TAIL_RTOL * total:
                small_panels += 1
            else:
                small_panels = 0
            upper_log = lower_log
            if small_panels >= 2 and log_radius - upper_log >= min(
                _MASS_MIN_LOG_SPAN, log_radius - _LOG_SMALLEST_RADIUS
            ):
                break
        else:
            raise ArithmeticError("enclosed mass did not converge over the inner log-radius survey")
        if small_panels < 2:
            raise ArithmeticError("enclosed mass cannot be resolved before the float64 radius limit")
        result = 4.0 * math.pi * total
        if not math.isfinite(result) or result < 0.0:
            raise ArithmeticError(f"enclosed mass is not finite at r={radius:g}")
        if result == 0.0 and self._rho(radius) > 0.0:
            raise ArithmeticError(f"enclosed mass underflowed at r={radius:g}")
        return result

    def _remember_mass(self, radius: float, mass: float) -> None:
        index = bisect_left(self._mass_anchor_radii, radius)
        if index < len(self._mass_anchor_radii) and self._mass_anchor_radii[index] == radius:
            return
        self._mass_anchor_radii.insert(index, radius)
        self._mass_anchor_values.insert(index, mass)
        if len(self._mass_anchor_radii) > 2048:
            # The reference mass is stored independently in self._m_ref.
            del self._mass_anchor_radii[0]
            del self._mass_anchor_values[0]

    def _local_moment(self, a: float, b: float, power: int) -> float:
        delta = b - a
        if delta == 0.0:
            return 0.0
        return delta * self._integrate(
            lambda t: (a + delta * t) ** power * self._rho(a + delta * t),
            0.0, 1.0, "local density moment",
            breakpoints=self._unit_density_breakpoints(a, b),
        )

    def _mass_uncached(self, radius: float) -> float:
        if self._r_ref is not None and 0.5 <= radius / self._r_ref <= 2.0:
            result = self._m_ref + 4.0 * math.pi * self._local_moment(self._r_ref, radius, 2)
            if not math.isfinite(result) or result < 0.0:
                raise ArithmeticError(f"enclosed mass is unresolved at r={radius:g}")
        else:
            index = bisect_left(self._mass_anchor_radii, radius)
            if index and radius / self._mass_anchor_radii[index - 1] <= math.exp(4.0):
                anchor = self._mass_anchor_radii[index - 1]
                result = self._mass_anchor_values[index - 1]
                # Log-radius endpoints can differ by only a few ulps during
                # a turning-point solve. Integrate that short interval on
                # [0, 1] to retain its width and avoid QUADPACK roundoff.
                moment = (self._local_moment(anchor, radius, 2)
                          if radius / anchor <= 2.0 else
                          self._log_mass_moment(anchor, radius))
                result += 4.0 * math.pi * moment
            else:
                result = self._mass_direct(radius)
        if not math.isfinite(result) or result < 0.0:
            raise ArithmeticError(f"enclosed mass is unresolved at r={radius:g}")
        self._remember_mass(radius, result)
        return result

    def _log_moment(self, a: float, b: float) -> float:
        """Oriented integral of s*rho(s) from a to b, in log-radius panels."""
        if a == b:
            return 0.0
        log_a, log_b = math.log(a), math.log(b)
        count = max(1, math.ceil(abs(log_b - log_a) / 2.0))
        width = (log_b - log_a) / count

        def integrand(log_radius: float) -> float:
            radius = math.exp(log_radius)
            return (radius * self._rho(radius)) * radius

        terms = [
            self._integrate(
                integrand, log_a + index * width, log_a + (index + 1) * width,
                "density shell integral",
                breakpoints=self._log_density_breakpoints,
            )
            for index in range(count)
        ]
        return math.fsum(terms)

    def _remember_shell(self, radius: float, shell: float) -> None:
        index = bisect_left(self._shell_anchor_radii, radius)
        if index < len(self._shell_anchor_radii) and self._shell_anchor_radii[index] == radius:
            return
        self._shell_anchor_radii.insert(index, radius)
        self._shell_anchor_values.insert(index, shell)
        if len(self._shell_anchor_radii) > 2048:
            del self._shell_anchor_radii[0]
            del self._shell_anchor_values[0]

    def _outer_shell_uncached(self, radius: float) -> float:
        """Reuse a neighboring shell integral and integrate only between radii."""
        index = bisect_left(self._shell_anchor_radii, radius)

        def between(a: float, b: float) -> float:
            return (self._local_moment(a, b, 1) if b / a <= 2.0
                    else self._log_moment(a, b))

        if index < len(self._shell_anchor_radii):
            anchor = self._shell_anchor_radii[index]
            if anchor / radius <= math.exp(4.0):
                value = self._shell_anchor_values[index] + between(radius, anchor)
                self._remember_shell(radius, value)
                return value
        if index:
            anchor = self._shell_anchor_radii[index - 1]
            if radius / anchor <= math.exp(4.0):
                anchor_value = self._shell_anchor_values[index - 1]
                value = anchor_value - between(anchor, radius)
                if value > 1e-10 * anchor_value:
                    self._remember_shell(radius, value)
                    return value

        value = self._outer_shell_direct(radius)
        self._remember_shell(radius, value)
        return value

    def _outer_shell_direct(self, radius: float) -> float:
        """Integral from r to infinity of s*rho(s), with a surveyed tail."""
        log_radius = math.log(radius)
        max_span = _LOG_MAX - log_radius - 0.01
        if max_span <= 0.0:
            raise ArithmeticError("outer shell integral exceeds the float64 radius range")
        total = 0.0
        previous_span = 0.0
        previous_contribution = None
        small_segments = 0
        last_span = 0.0
        for index in range(64):
            # Check the tail frequently after the initial geometric survey.
            # Doubling a span from 128 to 256 log units can drive an already
            # negligible NFW shell below float64 resolution before the second
            # small-segment check has a chance to stop the survey.
            next_span = min(float(2**index), previous_span + _MAX_TAIL_LOG_STEP,
                            max_span)
            if next_span <= previous_span:
                break
            contribution = self._log_moment(
                math.exp(log_radius + previous_span),
                math.exp(log_radius + next_span),
            )
            total += contribution
            last_span = next_span
            if (
                contribution == 0.0
                and previous_contribution is not None
                and previous_contribution > _TAIL_RTOL * total
            ):
                raise ArithmeticError(
                    "outer shell contribution vanished abruptly before its tail was negligible"
                )
            if index >= 2 and contribution <= _TAIL_RTOL * total:
                small_segments += 1
                if small_segments >= 2 and next_span >= _MIN_TAIL_LOG_SPAN:
                    break
            else:
                small_segments = 0
            previous_contribution = contribution
            previous_span = next_span
        else:
            raise ArithmeticError("outer shell integral did not converge before the radius limit")
        if small_segments < 2 or last_span < _MIN_TAIL_LOG_SPAN:
            raise ArithmeticError(
                "outer shell tail could not be resolved over the log-radius survey"
            )
        if not math.isfinite(total) or total < 0.0:
            raise ArithmeticError("outer shell integral is non-finite")
        return total

    def __call__(self, radius: float) -> float:
        """Return phi(r) in the boundary's energy convention."""
        radius = _positive_radius(radius)
        if self._boundary == "finite_escape":
            value = -self._G * self._mass(radius) / radius
            value -= 4.0 * math.pi * self._G * self._outer_shell(radius)
        else:
            reference = self._r_ref
            if radius == reference:
                return 0.0
            if 0.5 <= radius / reference <= 2.0:
                # Algebraically equivalent to the mass-plus-shell formula,
                # but the first term is linear in delta and the second is
                # quadratic: no subtraction of nearly equal large values.
                delta = radius - reference
                shell = self._integrate(
                    lambda t: (reference + delta * t) * (1.0 - t)
                    * self._rho(reference + delta * t),
                    0.0, 1.0, "local potential shell integral",
                    breakpoints=self._unit_density_breakpoints(reference, radius),
                )
                value = self._G * self._m_ref * delta / (radius * reference)
                value += 4.0 * math.pi * self._G * delta * delta * shell / radius
            else:
                value = self._G * (self._m_ref / reference - self._mass(radius) / radius)
                value += 4.0 * math.pi * self._G * self._log_moment(reference, radius)
        if not math.isfinite(value):
            raise ArithmeticError(f"initial potential is non-finite at r={radius:g}")
        if value == 0.0 and self._rho(radius) > 0.0:
            raise ArithmeticError(f"initial potential underflowed at r={radius:g}")
        return value

    def prime(self, radius: float) -> float:
        """Return phi'(r) = G M(r)/r²."""
        radius = _positive_radius(radius)
        value = (self._G * self._mass(radius) / radius) / radius
        if not math.isfinite(value):
            raise ArithmeticError(f"initial potential derivative is non-finite at r={radius:g}")
        return value

    def second(self, radius: float) -> float:
        """Return phi''(r), or raise if cancellation exceeds float64 precision."""
        radius = _positive_radius(radius)
        density_term = 4.0 * math.pi * self._G * self._rho(radius)
        mass_term = 2.0 * self.prime(radius) / radius
        value = density_term - mass_term
        # If the two terms nearly cancel, integrate their difference *before*
        # summing. With q(s)=s*rho(s), phi''=8*pi*G/r * int_0^1
        # t [q(r)-q(r*t)] dt. This does not require a model-specific cusp.
        if abs(value) <= 0.01 * (abs(density_term) + abs(mass_term)):
            q_radius = radius * self._rho(radius)
            roundoff_scale = abs(q_radius)

            def contrast(t: float) -> float:
                nonlocal roundoff_scale
                if t == 0.0:
                    return 0.0
                inner_radius = radius * t
                if inner_radius == 0.0:
                    raise ArithmeticError("second derivative reached the float64 radius limit")
                q_inner = inner_radius * self._rho(inner_radius)
                roundoff_scale = max(roundoff_scale, t * abs(q_inner))
                return t * (q_radius - q_inner)

            try:
                contrast_integral = self._integrate(
                    contrast, 0.0, 1.0, "second derivative contrast",
                    breakpoints=self._unit_density_breakpoints(0.0, radius),
                )
            except ArithmeticError as exc:
                raise ArithmeticError(
                    f"second derivative lacks float64 precision at r={radius:g} "
                    "(density contrast cannot be resolved)"
                ) from exc
            value = 8.0 * math.pi * self._G * contrast_integral / radius
            rounding_floor = (
                32.0 * np.finfo(float).eps * 8.0 * math.pi * self._G
                * roundoff_scale / radius
            )
            if abs(value) <= rounding_floor / _SECOND_RTOL:
                raise ArithmeticError(
                    f"second derivative lacks float64 precision at r={radius:g}"
                )
        if not math.isfinite(value):
            raise ArithmeticError(f"initial potential second derivative is non-finite at r={radius:g}")
        return value


def make_initial_potential(
    density: Callable[[float], float],
    boundary: str,
    *,
    G: float = DEFAULT_G,
    r_ref: float | None = None,
) -> _InitialPotential:
    """Build the spherical initial potential for a generic density.

    ``finite_escape`` sets phi(infinity)=0 and rejects ``r_ref``. It requires
    a convergent integral of r*rho(r) at infinity, but *not* finite total
    mass. ``confining`` requires finite positive ``r_ref`` and sets
    phi(r_ref)=0. Both density and the returned potential receive positive
    scalar radii. The potential also exposes ``prime`` and ``second`` for
    derivative-aware Eddington inversion.

    A piecewise density can expose ``integration_breakpoints``, a sequence
    of positive finite radii. All density quadratures are split at those
    radii so an adaptive integral never has to discover interpolation knots.

    The quadrature error estimate excludes error in ``density`` itself and
    cannot certify that an unobserved remote feature is absent. The inward
    mass survey likewise cannot rule out an arbitrarily narrow or distant
    central mass concentration. ``second`` uses a density-contrast integral
    when direct subtraction is ill-conditioned, and raises ArithmeticError
    if float64 density evaluations cannot resolve the result. A truly zero
    second derivative may therefore be numerically uncertifiable.
    """
    if not callable(density):
        raise TypeError("density must be a scalar callable")
    if boundary not in ("finite_escape", "confining"):
        raise ValueError("boundary must be 'finite_escape' or 'confining'")
    try:
        gravitational_constant = float(G)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("G must be positive and finite") from exc
    if not (math.isfinite(gravitational_constant) and gravitational_constant > 0.0):
        raise ValueError("G must be positive and finite")
    if boundary == "finite_escape":
        if r_ref is not None:
            raise ValueError("r_ref is not used with finite_escape")
        reference_radius = None
    else:
        if r_ref is None:
            raise ValueError("confining requires a positive finite r_ref")
        reference_radius = _positive_radius(r_ref)
    result = _InitialPotential(density, boundary, gravitational_constant, reference_radius)
    result._rho(1.0)
    return result


class TabulatedPotential:
    """Hermite table in log radius, with the direct Poisson potential outside."""

    def __init__(self, direct, log_radii, values, log_slopes, unsafe_intervals=None):
        self.direct = direct
        self.log_radii = np.asarray(log_radii, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.log_slopes = np.asarray(log_slopes, dtype=float)
        self.unsafe_intervals = (np.zeros(len(self.log_radii) - 1, dtype=bool)
                                 if unsafe_intervals is None else
                                 np.asarray(unsafe_intervals, dtype=bool))
        if (not callable(direct) or not callable(getattr(direct, "prime", None))
                or len(self.log_radii) < 3 or np.any(np.diff(self.log_radii) <= 0.0)
                or self.values.shape != self.log_radii.shape
                or self.log_slopes.shape != self.log_radii.shape
                or self.unsafe_intervals.shape != (len(self.log_radii) - 1,)
                or not np.all(np.isfinite(self.values))
                or not np.all(np.isfinite(self.log_slopes))):
            raise ValueError("invalid potential table")
        self._interpolator = CubicHermiteSpline(
            self.log_radii, self.values, self.log_slopes,
        )
        self.radial_center_allowed = getattr(direct, "radial_center_allowed", False)

    def _coordinate(self, radius):
        return math.log(_positive_radius(radius))

    def __call__(self, radius):
        coordinate = self._coordinate(radius)
        if self.log_radii[0] <= coordinate <= self.log_radii[-1]:
            index = min(np.searchsorted(self.log_radii, coordinate, side="right") - 1,
                        len(self.unsafe_intervals) - 1)
            if not self.unsafe_intervals[index]:
                return float(self._interpolator(coordinate))
        return float(self.direct(radius))

    def prime(self, radius):
        # Differentiating nearly constant values of a deep central potential
        # loses digits; the mass-based direct derivative remains well resolved.
        return float(self.direct.prime(radius))

    def second(self, radius):
        return float(self.direct.second(radius))

    def table_data(self):
        return (self.log_radii, self.values, self.log_slopes, self.unsafe_intervals)


def tabulate_potential(direct, minimum_radius, maximum_radius, *,
                       rtol=1e-7, max_points=4097):
    """Sample a halo potential and validate values at interval midpoints."""
    minimum_radius = _positive_radius(minimum_radius)
    maximum_radius = _positive_radius(maximum_radius)
    if maximum_radius <= minimum_radius:
        raise ValueError("potential table radii must increase")
    if not math.isfinite(rtol) or not 0.0 < rtol < 0.01:
        raise ValueError("rtol must lie between zero and 0.01")
    if type(max_points) is not int or max_points < 3:
        raise ValueError("max_points must be at least three")
    span = math.log(maximum_radius / minimum_radius)
    count = max(3, math.ceil(span / 0.5) + 1)
    if count > max_points:
        raise ValueError("potential table domain exceeds max_points")
    nodes = np.linspace(math.log(minimum_radius), math.log(maximum_radius), count)

    @lru_cache(maxsize=None)
    def sample(coordinate):
        radius = math.exp(coordinate)
        value = float(direct(radius))
        slope = radius * float(direct.prime(radius))
        if not math.isfinite(value) or not math.isfinite(slope) or slope < 0.0:
            raise ArithmeticError("potential table requires finite nondecreasing potential")
        return value, slope

    while True:
        values = np.array([sample(float(node))[0] for node in nodes])
        slopes = np.array([sample(float(node))[1] for node in nodes])
        table = TabulatedPotential(direct, nodes, values, slopes)
        midpoints = 0.5 * (nodes[:-1] + nodes[1:])
        refine = []
        unsafe = []
        for midpoint in midpoints:
            exact, exact_slope = sample(float(midpoint))
            interpolated = float(table._interpolator(midpoint))
            # Near a finite central potential, the local energy scale is much
            # smaller than |Phi|. Keep an absolute float64 roundoff floor.
            scale = max(abs(exact_slope), 1e-9 * abs(exact))
            desired = rtol * scale
            roundoff_floor = 64.0 * np.finfo(float).eps * abs(exact)
            unsafe.append(roundoff_floor > desired)
            refine.append(roundoff_floor <= desired and
                          abs(interpolated - exact) > desired)
        refine = np.asarray(refine, dtype=bool)
        if not np.any(refine):
            return TabulatedPotential(direct, nodes, values, slopes, unsafe)
        if len(nodes) + int(np.count_nonzero(refine)) > max_points:
            raise ArithmeticError("halo potential table did not meet its interpolation tolerance")
        nodes = np.sort(np.r_[nodes, midpoints[refine]])


__all__ = ["make_initial_potential", "TabulatedPotential", "tabulate_potential"]
