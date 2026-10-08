"""Adiabatic final distribution and density in a spherical total potential.

The potential supplied here includes the black hole and the halo. The final
density integral starts at L_c = 2 c R_S for each trial potential. The initial
density, potential and isotropic distribution remain fixed while the final
potential changes.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.interpolate import PchipInterpolator
from numpy.polynomial.legendre import leggauss

from .adiabatic_map import solve_initial_energy
from .constants import C_LIGHT, G
from .eddington import make_eddington_df
from .initial_profile import make_initial_potential


class _DensityQuadratureError(ArithmeticError):
    """A density rule failed; distinct from errors evaluating its DF or map."""


@lru_cache(maxsize=8)
def _unit_rule(order):
    abscissas, weights = leggauss(order)
    return 0.5 * (abscissas + 1.0), 0.5 * weights


@lru_cache(maxsize=8)
def _angular_rule(order):
    nodes, weights = _unit_rule(order)
    theta = 0.5 * math.pi * nodes
    return theta, 0.5 * math.pi * weights * np.sin(theta)


def schwarzschild_radius(M: float, G: float = G, clight: float = C_LIGHT) -> float:
    """Return 2 G M / c² in pc for M in solar masses."""
    mass, gravity, light = float(M), float(G), float(clight)
    if not all(math.isfinite(v) and v > 0 for v in (mass, gravity, light)):
        raise ValueError("M, G and clight must be positive and finite")
    return 2.0 * gravity * mass / light**2


def _capture_threshold(potential, override):
    raw = (getattr(potential, "capture_angular_momentum", 0.0)
           if override is None else override)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("capture angular momentum must be nonnegative and finite") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("capture angular momentum must be nonnegative and finite")
    return value


def make_final_distribution(
    initial_df: Callable[[float], float],
    initial_potential: Callable[[float], float],
    final_potential: Callable[[float], float],
    *,
    map_kwargs: dict | None = None,
) -> Callable[[float, float], float]:
    """Return the final DF by conserving radial action and L."""
    if not all(callable(fn) for fn in (initial_df, initial_potential, final_potential)):
        raise TypeError("the distribution and both potentials must be callable")
    options = dict(map_kwargs or {})
    action_table = options.pop("initial_action_table", None)
    if set(options) & {"E_prime", "L_prime", "potential", "final_potential",
                       "_target_action"}:
        raise ValueError("map_kwargs cannot replace the mapping inputs")

    def distribution(energy: float, angular_momentum: float) -> float:
        angular_momentum = float(angular_momentum)
        if not math.isfinite(angular_momentum) or angular_momentum < 0.0:
            raise ValueError("angular momentum must be nonnegative and finite")
        # The unchanged system is an exact fixed point, including at L = 0.
        if final_potential is initial_potential:
            initial_energy = energy
        elif action_table is None:
            initial_energy = solve_initial_energy(
                energy, angular_momentum, initial_potential, final_potential,
                **options,
            )
        else:
            initial_energy = action_table.map(
                energy, angular_momentum, final_potential, options=options,
            )
        value = float(initial_df(initial_energy))
        if not math.isfinite(value) or value < 0.0:
            raise ArithmeticError("mapped distribution must be finite and nonnegative")
        return value

    return distribution


def rho_prime_at_r(
    f_prime: Callable[[float, float], float],
    r: float,
    potential: Callable[[float], float],
    boundary: str,
    *,
    epsabs: float = 0.0,
    epsrel: float = 1e-6,
    limit: int = 100,
    energy_scale: float | None = None,
    capture_angular_momentum: float | None = None,
    tail_survey_scales: float = 32.0,
    max_tail_segments: int = 64,
) -> float:
    """Integrate the final DF over velocities in the supplied total potential.

    For ``finite_escape``, the energy interval is
    [Phi(r) + L_c²/(2r²), 0] when capture is enabled. For
    ``confining``, it extends to infinity and is integrated in growing panels
    until the observed tail is negligible. That tail check is a numerical
    diagnostic, not a proof about an arbitrary DF at remote energies. The
    angular integral and finite-escape energy integral use successive Gauss
    rules. For capture, the allowed domain starts at L_c and the velocity
    substitution regularizes its narrow boundary near 4 R_S. A finite-escape
    integral unresolved by order 256 is checked with adaptive quadrature and,
    if needed, one further Gauss refinement.
    """
    if not callable(f_prime) or not callable(potential):
        raise TypeError("distribution and potential must be callable")
    if boundary not in ("finite_escape", "confining"):
        raise ValueError("boundary must be finite_escape or confining")
    radius = float(r)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    phi = float(potential(radius))
    if not math.isfinite(phi):
        raise ArithmeticError("final potential is non-finite")
    capture_l = _capture_threshold(potential, capture_angular_momentum)
    if not math.isfinite(epsabs) or epsabs < 0.0 or not math.isfinite(epsrel) or epsrel <= 2e-13:
        raise ValueError("invalid density quadrature tolerances")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if type(max_tail_segments) is not int or max_tail_segments < 4:
        raise ValueError("max_tail_segments must be at least four")
    if not math.isfinite(tail_survey_scales) or tail_survey_scales <= 0.0:
        raise ValueError("tail_survey_scales must be positive and finite")
    if boundary == "finite_escape":
        if phi >= 0.0:
            raise ValueError("finite_escape requires a negative total potential")
        scale = -phi
    else:
        if energy_scale is None:
            slope = radius * potential.prime(radius) if callable(getattr(potential, "prime", None)) else 0.0
            scale = max(abs(float(slope)), abs(phi), 1.0)
        else:
            scale = float(energy_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("energy_scale must be positive and finite")

    speed_scale = math.sqrt(2.0 * scale)
    capture_speed = capture_l / radius
    capture_t = capture_speed / speed_scale
    if not math.isfinite(capture_t):
        raise ArithmeticError("capture threshold exceeds the velocity range")
    # At this radius and in the current trial potential, L >= L_c requires
    # E >= E_min = Phi_f(r) + L_c²/(2r²).
    energy_min = math.fsum((phi, 0.5 * capture_speed**2))
    if boundary == "finite_escape" and energy_min >= 0.0:
        return 0.0
    finite_span = ((-energy_min / scale)
                   if boundary == "finite_escape" else None)

    def integrate(fn, a, b, absolute):
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                value, error = quad(fn, a, b, epsabs=absolute, epsrel=epsrel / 4, limit=limit)
            except IntegrationWarning as exc:
                raise _DensityQuadratureError("final density quadrature did not converge") from exc
        if not (math.isfinite(value) and math.isfinite(error)):
            raise _DensityQuadratureError("final density quadrature returned a non-finite value")
        if error > max(absolute, epsrel * abs(value) / 4):
            raise _DensityQuadratureError("final density quadrature exceeds its error estimate")
        return value, error

    def energy_integrand(u):
        if u == 0.0:
            return 0.0
        if boundary == "finite_escape":
            t_squared = capture_t**2 + finite_span * u**2
            energy = energy_min * (1.0 - u**2)
            jacobian = finite_span * u / math.sqrt(t_squared)
        else:
            t_squared = capture_t**2 + u**2
            energy = math.fsum((energy_min, scale * u**2))
            jacobian = u / math.sqrt(t_squared)
        speed = speed_scale * math.sqrt(t_squared)
        angular_scale = radius * speed

        # mu = cos(theta): the sqrt(1-mu²) endpoint in L becomes sin(theta),
        # so successive Gauss rules converge without repeated action solves
        # near the tangential/radial endpoints.
        angular = None
        for order in (8, 16, 32, 64, 128, 256):
            terms = []
            if capture_l == 0.0:
                nodes, weights = _angular_rule(order)
                for theta, weight in zip(nodes, weights):
                    angular_momentum = angular_scale * math.sin(float(theta))
                    value = float(f_prime(energy, angular_momentum))
                    if not math.isfinite(value) or value < 0.0:
                        raise ArithmeticError("final distribution must be finite and nonnegative")
                    terms.append(float(weight) * value)
            else:
                radial_momentum = math.sqrt(max(
                    (angular_scale - capture_l) * (angular_scale + capture_l), 0.0,
                ))
                angular_width = math.atan2(radial_momentum, capture_l)
                nodes, weights = _unit_rule(order)
                for node, weight in zip(nodes, weights):
                    angle = angular_width * float(node)
                    cosine = math.cos(angle)
                    angular_momentum = max(capture_l, angular_scale * cosine)
                    value = float(f_prime(energy, angular_momentum))
                    if not math.isfinite(value) or value < 0.0:
                        raise ArithmeticError("final distribution must be finite and nonnegative")
                    terms.append(angular_width * float(weight) * cosine * value)
            refined = math.fsum(terms)
            if angular is not None and abs(refined - angular) <= epsrel * max(refined, angular) / 4:
                angular = refined
                break
            angular = refined
        else:
            raise ArithmeticError("angular density quadrature did not converge")
        return 4.0 * math.pi * speed**2 * speed_scale * jacobian * angular

    # v² = 2*scale*(capture_t² + span*u²), making the capture edge smooth.
    if boundary == "finite_escape":
        panels = [0.0, 1.0]
        if isinstance(potential, _TotalPotential) and potential.gm > 0.0:
            # The mapped DF can drop sharply where the kinetic energy reaches
            # the point-mass part of the potential. Keep that transition
            # inside a short panel, away from either quadrature endpoint.
            point_mass_fraction = (potential.gm / radius) / scale
            transition_squared = (point_mass_fraction - capture_t**2) / finite_span
            if 0.01 < point_mass_fraction < 1.0 and 0.0 < transition_squared < 1.0:
                transition = math.sqrt(transition_squared)
                half_width = min(0.02, max(0.005, 0.25 * (1.0 - transition)))
                panels = sorted({
                    0.0, max(0.0, transition - half_width),
                    min(1.0, transition + half_width), 1.0,
                })

        def energy_gauss(order):
            nodes, weights = _unit_rule(order)
            return math.fsum(
                (end - start) * math.fsum(
                    float(weight) * energy_integrand(
                        start + (end - start) * float(node)
                    )
                    for node, weight in zip(nodes, weights)
                )
                for start, end in zip(panels[:-1], panels[1:])
            )

        previous = None
        for order in (8, 16, 32, 64, 128, 256):
            refined = energy_gauss(order)
            if previous is not None and abs(refined - previous) <= max(
                epsabs, epsrel * max(abs(refined), abs(previous))
            ) / 4:
                density = refined
                break
            previous = refined
        else:
            # Narrow features need more resolution than one global Gauss rule.
            # Keep the order-256 estimate as an independent check on QUADPACK.
            try:
                adaptive, _ = integrate(energy_integrand, 0.0, 1.0, epsabs)
            except _DensityQuadratureError:
                adaptive = None
            if adaptive is not None and abs(adaptive - previous) <= max(
                epsabs, epsrel * max(abs(adaptive), abs(previous))
            ) / 4:
                density = adaptive
            else:
                refined = energy_gauss(512)
                tolerance = max(epsabs, epsrel * max(abs(refined), abs(previous))) / 4
                if abs(refined - previous) <= tolerance or (
                    adaptive is not None and abs(refined - adaptive) <= tolerance
                ):
                    density = refined
                else:
                    raise ArithmeticError(
                        "energy density quadrature did not converge: "
                        f"order 256={previous:.17g}, order 512={refined:.17g}, "
                        f"relative change={abs(refined - previous) / max(abs(refined), abs(previous)):.6g}, "
                        f"required={epsrel / 4:.6g}"
                    )
    else:
        density = 0.0
        segments = []
        previous = 0.0
        small = 0
        for index in range(max_tail_segments):
            end = math.sqrt(2.0**index)
            segment, _ = integrate(energy_integrand, previous, end, epsabs / max_tail_segments)
            density += segment
            segments.append(segment)
            previous = end
            if len(segments) < 4:
                continue
            ratios = [
                newer / older if older > 0.0 else (0.0 if newer == 0.0 else math.inf)
                for older, newer in zip(segments[-3:-1], segments[-2:])
            ]
            ratio = max(ratios)
            remainder = segment * ratio / (1.0 - ratio) if ratio < 1.0 else math.inf
            if 2.0**index >= tail_survey_scales and remainder <= max(epsabs, epsrel * density) / 4:
                small += 1
                if small == 2:
                    break
            else:
                small = 0
        else:
            raise ArithmeticError("confining final density tail did not converge")
    if not math.isfinite(density) or density < 0.0:
        raise ArithmeticError("final density is invalid")
    return density


def rho_prime_profile(f_prime, r_array, potential, boundary, **kwargs) -> np.ndarray:
    """Evaluate a final density on an array of positive radii."""
    radii = np.asarray(r_array, dtype=float)
    return np.asarray([
        rho_prime_at_r(f_prime, float(radius), potential, boundary, **kwargs)
        for radius in radii.flat
    ]).reshape(radii.shape)


def make_final_profile(
    initial_df: Callable[[float], float],
    initial_potential: Callable[[float], float],
    final_potential: Callable[[float], float],
    boundary: str,
    *,
    epsabs: float = 0.0,
    epsrel: float = 1e-6,
    limit: int = 100,
    map_kwargs: dict | None = None,
    capture_angular_momentum: float | None = None,
) -> Callable[[float], float]:
    """Return the density implied by action mapping in a final potential."""
    capture_l = _capture_threshold(final_potential, capture_angular_momentum)
    distribution = make_final_distribution(
        initial_df, initial_potential, final_potential, map_kwargs=map_kwargs,
    )

    def final_density(radius: float) -> float:
        return rho_prime_at_r(
            distribution, radius, final_potential, boundary,
            epsabs=epsabs, epsrel=epsrel, limit=limit,
            capture_angular_momentum=capture_l,
        )

    final_density.distribution = distribution
    final_density.potential = final_potential
    return final_density


class _TotalPotential:
    """The halo Poisson potential and one central point-mass contribution."""

    radial_center_allowed = True

    def __init__(self, halo, mass, gravity, clight=C_LIGHT):
        self.halo = halo
        self.gm = gravity * mass
        light = float(clight)
        if not math.isfinite(light) or light <= 0.0:
            raise ValueError("clight must be positive and finite")
        self.capture_angular_momentum = 4.0 * self.gm / light

    def __call__(self, radius):
        return math.fsum((self.halo(radius), -self.gm / radius))

    def prime(self, radius):
        return self.halo.prime(radius) + self.gm / radius**2


class _RadialDensity:
    """Smooth correction to the original density passed to Poisson.

    The logarithmic density ratio is interpolated on the radial grid. If
    capture produces zeros, interpolate the nonnegative ratio itself. Its
    slope vanishes at either endpoint and the endpoint ratio is held fixed
    outside the grid. Thus an unchanged profile is represented exactly and
    the original inner and outer asymptotics are retained up to a scale.
    """

    def __init__(self, radii, values, original):
        self.radii = np.asarray(radii, dtype=float)
        self.values = np.asarray(values, dtype=float)
        if np.any(self.values < 0.0) or not np.all(np.isfinite(self.values)):
            raise ValueError("Poisson grid density must be nonnegative and finite")
        self.logr = np.log(self.radii)
        reference = np.array([float(original(float(r))) for r in self.radii])
        if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
            raise ValueError("initial density must be positive and finite on the grid")
        ratio = self.values / reference
        self.logarithmic = bool(np.all(ratio > 0.0))
        correction = np.log(ratio) if self.logarithmic else ratio
        if not np.all(np.isfinite(correction)):
            raise ValueError("density correction is not finite")
        left = self.logr[0] - (self.logr[1] - self.logr[0])
        right = self.logr[-1] + (self.logr[-1] - self.logr[-2])
        self.correction = PchipInterpolator(
            np.r_[left, self.logr, right],
            np.r_[correction[0], correction, correction[-1]],
        )
        self.inner_scale = float(ratio[0])
        self.outer_scale = float(ratio[-1])
        self.original = original
        # Include the grid boundaries: they join the interpolated correction
        # to its constant inner/outer continuation. Retain any knots supplied
        # by the original density as well.
        self.integration_breakpoints = (
            tuple(float(radius) for radius in self.radii)
            + tuple(getattr(original, "integration_breakpoints", ()))
        )

    def __call__(self, radius):
        radius = float(radius)
        if radius <= 0.0 or not math.isfinite(radius):
            raise ValueError("radius must be positive and finite")
        if radius < self.radii[0]:
            factor = self.inner_scale
        elif radius > self.radii[-1]:
            factor = self.outer_scale
        else:
            value = float(self.correction(math.log(radius)))
            factor = math.exp(value) if self.logarithmic else max(0.0, value)
        return float(factor * self.original(radius)) if factor > 0.0 else 0.0


class _InnerContinuedPotential:
    """Exact inner Poisson continuation for a constant density ratio.

    Below the first density-grid radius, ``_RadialDensity`` is
    ``inner_scale * original``. Enclosed mass and the potential gradient
    therefore have the same ratio to the initial system there. The potential
    value at the grid boundary fixes the additive contribution of all outer
    shells without reintegrating them at every deep turning point.
    """

    def __init__(self, direct, initial, inner_radius, inner_scale):
        self.direct = direct
        self.initial = initial
        self.inner_radius = float(inner_radius)
        self.inner_scale = float(inner_scale)
        if (not math.isfinite(self.inner_radius) or self.inner_radius <= 0.0
                or not math.isfinite(self.inner_scale) or self.inner_scale < 0.0):
            raise ValueError("invalid inner potential continuation")
        self.boundary_value = float(direct(self.inner_radius))
        self.initial_boundary_value = float(initial(self.inner_radius))
        if not (math.isfinite(self.boundary_value)
                and math.isfinite(self.initial_boundary_value)):
            raise ArithmeticError("inner potential boundary is non-finite")
        self.radial_center_allowed = getattr(direct, "radial_center_allowed", False)

    def __call__(self, radius):
        radius = float(radius)
        if radius >= self.inner_radius:
            return self.direct(radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        if self.inner_scale == 0.0:
            return self.boundary_value
        offset = getattr(self.initial, "offset_from_center", None)
        difference = (offset(radius) - offset(self.inner_radius) if callable(offset)
                      else self.initial(radius) - self.initial_boundary_value)
        return math.fsum((
            self.boundary_value,
            self.inner_scale * difference,
        ))

    def prime(self, radius):
        radius = float(radius)
        if radius >= self.inner_radius:
            return self.direct.prime(radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        return self.inner_scale * self.initial.prime(radius) if self.inner_scale else 0.0

    def second(self, radius):
        radius = float(radius)
        if radius >= self.inner_radius:
            return self.direct.second(radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        return self.inner_scale * self.initial.second(radius) if self.inner_scale else 0.0


@dataclass
class SelfConsistentResult:
    """Density/DF use density_potential; potential is its next Poisson update."""

    radii: np.ndarray
    density: np.ndarray
    density_profile: Callable[[float], float]
    potential: Callable[[float], float]
    distribution: Callable[[float, float], float]
    converged: bool
    history: list[dict]
    density_potential: Callable[[float], float] | None = None


class SelfConsistentConvergenceError(RuntimeError):
    """Iteration limit reached; ``result`` contains the final trial state."""

    def __init__(self, result):
        change = result.history[-1]["max_relative_density_change"]
        super().__init__(
            f"final density did not converge on the radial grid after "
            f"{len(result.history)} iterations (max relative change {change:.6g})"
        )
        self.result = result


def _initial_state(initial_density, boundary, r_ref, gravity,
                   initial_potential=None, initial_df=None, eddington_kwargs=None):
    """Build the fixed initial potential and DF shared by each iteration."""
    phi_initial = initial_potential if initial_potential is not None else make_initial_potential(
        initial_density, boundary, G=gravity, r_ref=r_ref,
    )
    if not callable(phi_initial) or not callable(getattr(phi_initial, "prime", None)):
        raise TypeError("initial potential must expose its radial derivative")
    if initial_df is None:
        options = dict(eddington_kwargs or {})
        for prefix, source in (("density", initial_density), ("potential", phi_initial)):
            if callable(getattr(source, "prime", None)) and callable(getattr(source, "second", None)):
                options.setdefault(prefix + "_prime", source.prime)
                options.setdefault(prefix + "_second", source.second)
        if initial_potential is None:
            options["potential_prime"] = phi_initial.prime
            options["potential_second"] = lambda radius: math.fsum((
                4.0 * math.pi * gravity * float(initial_density(radius)),
                -2.0 * phi_initial.prime(radius) / radius,
            ))
        initial_df = make_eddington_df(initial_density, phi_initial, boundary, **options)
    elif not callable(initial_df):
        raise TypeError("initial_df must be callable")
    return phi_initial, initial_df


def solve_self_consistent(
    initial_density,
    radii,
    M_bh,
    *,
    boundary="finite_escape",
    r_ref=None,
    G=G,
    clight=C_LIGHT,
    initial_potential=None,
    initial_df=None,
    eddington_kwargs=None,
    map_kwargs=None,
    epsrel=1e-5,
    limit=100,
    rtol=1e-4,
    max_iterations=100,
    iteration_callback=None,
    return_on_max_iterations=False,
    _potential_tabulator=None,
    _density_evaluator=None,
):
    """Iterate the appendix-A action map, density integral and halo Poisson solve.

    Convergence means ``max(abs(rho_new/rho_previous - 1)) < rtol`` on every
    supplied radius. No mixing enters this loop. For M_bh > 0, the final
    density integral starts at L_c = 2 c R_S = 4 G M_bh / c. Optional
    interpolation tables accelerate evaluations while checking action residuals.
    The finite grid represents the correction to the initial density;
    its inner and outer continuations are documented by ``_RadialDensity``.
    Vary the radial domain to assess their effect on the result. An optional
    ``iteration_callback`` receives the complete trial result after each
    Poisson update, before the convergence check. By default, exhausting the
    iteration limit raises; ``return_on_max_iterations=True`` returns the last
    state with ``converged=False`` instead.
    """
    if not callable(initial_density):
        raise TypeError("initial_density must be callable")
    grid = np.array(radii, dtype=float, copy=True)
    if (grid.ndim != 1 or len(grid) < 3 or not np.all(np.isfinite(grid))
            or np.any(grid <= 0.0) or np.any(np.diff(grid) <= 0.0)):
        raise ValueError("radii must be at least three increasing positive points")
    mass, gravity, light = float(M_bh), float(G), float(clight)
    if (not math.isfinite(mass) or mass < 0.0
            or not math.isfinite(gravity) or gravity <= 0.0
            or not math.isfinite(light) or light <= 0.0):
        raise ValueError("M_bh must be nonnegative, G and clight positive")
    if not math.isfinite(rtol) or not 0.0 < rtol < 1.0:
        raise ValueError("rtol must be between zero and one")
    if not math.isfinite(epsrel) or not 2e-13 < epsrel <= rtol / 8.0:
        raise ValueError("density epsrel must be at most one eighth of rtol")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if iteration_callback is not None and not callable(iteration_callback):
        raise TypeError("iteration_callback must be callable")
    if type(return_on_max_iterations) is not bool:
        raise TypeError("return_on_max_iterations must be boolean")
    if _potential_tabulator is not None and not callable(_potential_tabulator):
        raise TypeError("_potential_tabulator must be callable")
    previous = np.array([float(initial_density(float(r))) for r in grid])
    if not np.all(np.isfinite(previous)) or np.any(previous <= 0.0):
        raise ValueError("initial density must be positive and finite on the grid")

    phi_initial, initial_df = _initial_state(
        initial_density, boundary, r_ref, gravity,
        initial_potential, initial_df, eddington_kwargs,
    )

    map_options = dict(map_kwargs or {})
    map_options.setdefault("action_match_rtol", min(1e-5, epsrel / 10.0))
    action_options = dict(map_options.get("radial_action_kwargs", {}))
    action_options.setdefault("epsrel", min(1e-6, epsrel / 100.0))
    action_options.setdefault("epsabs", 0.0)
    map_options["radial_action_kwargs"] = action_options

    current_potential = (phi_initial if mass == 0.0 else
                         _TotalPotential(phi_initial, mass, gravity, light))
    history = []
    for iteration in range(1, max_iterations + 1):
        profile = make_final_profile(
            initial_df, phi_initial, current_potential, boundary,
            epsrel=epsrel, limit=limit, map_kwargs=map_options,
        )
        updated = (np.array([profile(float(r)) for r in grid])
                   if _density_evaluator is None else
                   np.asarray(_density_evaluator(iteration, grid, previous, profile,
                                                 map_options, epsrel, limit), dtype=float))
        if updated.shape != grid.shape:
            raise ValueError("updated density must match the radial grid")
        if not np.all(np.isfinite(updated)) or np.any(updated < 0.0):
            raise ArithmeticError("updated density must be nonnegative and finite on the grid")
        relative_change = np.zeros_like(updated)
        positive = previous > 0.0
        relative_change[positive] = np.abs(updated[positive] / previous[positive] - 1.0)
        # A reappearing density cannot be converged (rtol < 1); keep a finite
        # diagnostic for JSON snapshots. Zero -> zero has change zero.
        relative_change[(~positive) & (updated > 0.0)] = 1.0
        change = float(np.max(relative_change))
        history.append({"iteration": iteration, "max_relative_density_change": change})
        represented = _RadialDensity(grid, updated, initial_density)
        # The potential produced by this density is the next trial, including
        # on the converged iteration. The point mass is never passed to Poisson.
        if mass == 0.0 and iteration == 1 and change < rtol:
            candidate = phi_initial
        else:
            halo = _InnerContinuedPotential(
                make_initial_potential(represented, boundary, G=gravity, r_ref=r_ref),
                phi_initial, float(grid[0]), represented.inner_scale,
            )
            if _potential_tabulator is not None:
                halo = _potential_tabulator(halo, grid)
            candidate = (halo if mass == 0.0 else
                         _TotalPotential(halo, mass, gravity, light))
        result = SelfConsistentResult(
            grid.copy(), updated.copy(), represented, candidate,
            make_final_distribution(initial_df, phi_initial, current_potential,
                                    map_kwargs=map_options),
            change < rtol, list(history), density_potential=current_potential,
        )
        if iteration_callback is not None:
            iteration_callback(result)
        if result.converged:
            return result
        current_potential, previous = candidate, updated
    if return_on_max_iterations:
        return result
    raise SelfConsistentConvergenceError(result)


__all__ = [
    "schwarzschild_radius", "make_final_distribution", "rho_prime_at_r",
    "rho_prime_profile", "make_final_profile", "solve_self_consistent",
    "SelfConsistentResult", "SelfConsistentConvergenceError",
]
