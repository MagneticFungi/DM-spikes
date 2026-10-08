"""Numerical adiabatic mapping through conservation of radial action."""

from collections.abc import Callable
import math
from math import fsum
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq, minimize_scalar, root_scalar
from scipy.special import expit, logit



Potential = Callable[[float], float]

_FLOAT_EPS = np.finfo(float).eps
_MIN_BRENT_RTOL = 4.0 * _FLOAT_EPS
_LOG_RADIUS_STEP = np.log(2.0)
_LOG_RADIUS_MIN = np.log(np.finfo(float).tiny)
_LOG_RADIUS_MAX = np.log(np.finfo(float).max) - 1e-12


def _finite_scalar(value, name):
    array = np.asarray(value)
    if array.shape != ():
        raise TypeError(f"{name} must be a scalar.")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _potential_value(potential, radius, *, allow_negative_infinity=False):
    """Evaluate a scalar potential, optionally allowing an attractive singularity.

    The callable must preserve the energy differences relevant to the orbit
    before returning its float value; this module cannot recover lost digits.
    """
    if not callable(potential):
        raise TypeError("potential must be callable.")
    try:
        value = potential(float(radius))
    except Exception as exc:
        raise ValueError(
            f"potential could not be evaluated at r={radius:.17g}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if allow_negative_infinity:
        array = np.asarray(value)
        if array.shape == () and float(array) == -np.inf:
            return -np.inf
    return _finite_scalar(value, "potential(r)")


def _validate_solver_tolerances(root_xtol, root_rtol, maxiter):
    root_xtol = _finite_scalar(root_xtol, "root_xtol")
    root_rtol = _finite_scalar(root_rtol, "root_rtol")
    if root_xtol <= 0.0:
        raise ValueError("root_xtol must be positive.")
    if root_rtol < _MIN_BRENT_RTOL:
        raise ValueError(
            f"root_rtol must be at least {_MIN_BRENT_RTOL:.17g}."
        )
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    return root_xtol, root_rtol, int(maxiter)


def _radius_from_log(log_radius):
    if not _LOG_RADIUS_MIN <= log_radius <= _LOG_RADIUS_MAX:
        raise OverflowError(
            "The logarithmic radius exceeded the safe float range."
        )
    return float(np.exp(log_radius))


def _orbit_energy_coordinates(energy, potential):
    """Use the central offset only when it reduces the energy scale.

    Near escape, E and Phi(r) are small already. Subtracting Phi(0) from
    both would manufacture a cancellation between large, rounded values.
    Return the same coordinate choice to turning points and action integrals.
    """
    offset = getattr(potential, "offset_from_center", None)
    if callable(offset):
        center = _finite_scalar(potential.central_potential, "central potential")
        reduced = fsum((energy, -center))
        if abs(reduced) < abs(energy):
            return offset, reduced
    return None, None


def _find_effective_minimum(L, potential, root_xtol, maxiter, max_expansions):
    """Bracket the unique minimum from r = 1, then refine it in log(r)."""
    has_center_offset = callable(getattr(potential, "offset_from_center", None))
    offset = None
    if has_center_offset:
        offset, _ = _orbit_energy_coordinates(effective_potential(1.0, L, potential), potential)

    def value(log_radius):
        radius = _radius_from_log(log_radius)
        if callable(offset):
            return fsum((
                _finite_scalar(offset(radius), "potential offset"),
                0.5 * (L / radius)**2,
            ))
        return effective_potential(radius, L, potential)

    center = 0.0
    center_value = value(center)
    left = -_LOG_RADIUS_STEP
    right = _LOG_RADIUS_STEP
    left_value = value(left)
    right_value = value(right)

    if center_value < left_value and center_value < right_value:
        bracket = (left, right)
        best_log, best_value = center, center_value
        flat_bracket = False
    else:
        if left_value <= center_value and left_value < right_value:
            direction = -1.0
        elif right_value <= center_value and right_value < left_value:
            direction = 1.0
        else:
            raise ValueError(
                "The effective potential is flat or not single-welled near "
                "r = 1 at the available numerical precision."
            )
        previous_log = center
        best_log = left if direction < 0.0 else right
        best_value = left_value if direction < 0.0 else right_value

        for _ in range(max_expansions):
            next_log = best_log + direction * _LOG_RADIUS_STEP
            try:
                next_value = value(next_log)
            except OverflowError as exc:
                raise RuntimeError(
                    "Could not bracket the effective-potential minimum before "
                    "exhausting the numerical radius range."
                ) from exc
            if next_value >= best_value:
                bracket = tuple(sorted((previous_log, next_log)))
                flat_bracket = next_value == best_value
                break
            previous_log, best_log, best_value = best_log, next_log, next_value
        else:
            raise RuntimeError(
                "Could not bracket the effective-potential minimum within "
                "max_expansions."
            )

    # The minimum may lie far from r=1. Choose the better energy zero there,
    # then hold it fixed throughout refinement. Reevaluate stored values in
    # that same convention before comparing them with the refined minimum.
    if has_center_offset:
        offset, _ = _orbit_energy_coordinates(
            effective_potential(_radius_from_log(best_log), L, potential), potential,
        )
        best_value = value(best_log)
        flat_bracket = any(value(endpoint) == best_value for endpoint in bracket)

    minimum = minimize_scalar(
        value,
        bounds=bracket,
        method="bounded",
        options={"xatol": root_xtol, "maxiter": maxiter},
    )
    if not minimum.success:
        raise RuntimeError(
            "Could not refine the effective-potential minimum: "
            f"{minimum.message}"
        )
    refined_value = value(minimum.x)
    if flat_bracket and refined_value >= best_value:
        raise FloatingPointError(
            "The effective-potential minimum is flat at the available "
            "numerical precision."
        )
    if refined_value < best_value:
        best_log, best_value = float(minimum.x), float(refined_value)

    neighboring_values = (
        value(best_log - _LOG_RADIUS_STEP),
        value(best_log + _LOG_RADIUS_STEP),
    )
    local_energy_scale = max(neighboring_values) - best_value
    if not np.isfinite(local_energy_scale) or local_energy_scale <= 0.0:
        raise FloatingPointError(
            "The effective-potential minimum cannot be resolved at the "
            "available numerical precision."
        )
    if callable(offset):
        best_value = fsum((
            _finite_scalar(potential.central_potential, "central potential"),
            best_value,
        ))
    return float(best_log), float(best_value), float(local_energy_scale)


def _expand_turning_root(
    log_allowed,
    direction,
    gap,
    root_xtol,
    root_rtol,
    maxiter,
    max_expansions,
    *,
    center_allowed=False,
):
    """Expand geometrically to the first forbidden radius, then solve F = 0."""
    previous_log = log_allowed
    side = "inner" if direction < 0.0 else "outer"
    possible_cause = (
        "the orbit may be unbound or the return may lie beyond the numerical range"
        if direction > 0.0
        else "the inner return may lie below the numerical radius range"
    )
    for _ in range(max_expansions):
        next_log = previous_log + direction * _LOG_RADIUS_STEP
        try:
            next_radius = _radius_from_log(next_log)
        except OverflowError as exc:
            if center_allowed:
                return 0.0
            raise RuntimeError(
                f"Could not bracket the {side} turning point before reaching "
                f"the numerical radius limit; {possible_cause}."
            ) from exc

        next_gap = gap(next_log, allow_singular_center=center_allowed)
        if center_allowed and next_gap == np.inf:
            return 0.0
        if next_gap <= 0.0:
            if next_gap == 0.0:
                return next_radius
            log_root = brentq(
                gap,
                *sorted((previous_log, next_log)),
                xtol=root_xtol,
                rtol=root_rtol,
                maxiter=maxiter,
            )
            return _radius_from_log(log_root)
        previous_log = next_log

    raise RuntimeError(
        f"Could not bracket the {side} turning point within max_expansions; "
        f"{possible_cause}."
    )


def effective_potential(r, L, potential):
    """Return phi(r) + L^2 / (2 r^2) for a scalar positive radius."""
    radius = _finite_scalar(r, "r")
    angular_momentum = _finite_scalar(L, "L")
    if radius <= 0.0:
        raise ValueError("r must be positive.")
    if angular_momentum < 0.0:
        raise ValueError("L must be non-negative.")

    centrifugal = 0.5 * (angular_momentum / radius) ** 2
    result = fsum((_potential_value(potential, radius), centrifugal))
    if not np.isfinite(result):
        raise FloatingPointError(
            f"The effective potential is not finite at r={radius:.17g}."
        )
    return float(result)


def find_turning_points(
    E,
    L,
    potential,
    *,
    root_xtol=1e-12,
    root_rtol=_MIN_BRENT_RTOL,
    maxiter=200,
    max_expansions=2048,
    orbit_atol=0.0,
    orbit_rtol=32.0 * _FLOAT_EPS,
):
    """Find the turning radii of a single bound orbit without radial bounds.

    The scalar potential must describe one effective-potential well over the
    relevant positive radii. Searches begin at r = 1 in the potential's radius
    units and expand by factors of two. For L = 0, a finite inner return is
    found if present; otherwise an allowed interval reaching the numerical
    center uses r_minus = 0. A negative-infinite central limit is accepted
    only during this inward radial search. Failure to find an outer return
    does not prove that the orbit is unbound for an arbitrary potential.

    ``root_xtol`` and ``root_rtol`` apply in log-radius. A positive energy
    gap is never classified as circular; ``orbit_atol`` and ``orbit_rtol``
    only identify an unresolved energy below the numerical minimum.
    """
    energy = _finite_scalar(E, "E")
    angular_momentum = _finite_scalar(L, "L")
    if angular_momentum < 0.0:
        raise ValueError("L must be non-negative.")
    if not callable(potential):
        raise TypeError("potential must be callable.")

    offset, reduced_energy = _orbit_energy_coordinates(energy, potential)

    root_xtol, root_rtol, maxiter = _validate_solver_tolerances(
        root_xtol, root_rtol, maxiter
    )
    if not isinstance(max_expansions, (int, np.integer)) or max_expansions < 1:
        raise ValueError("max_expansions must be a positive integer.")
    max_expansions = int(max_expansions)
    orbit_atol = _finite_scalar(orbit_atol, "orbit_atol")
    orbit_rtol = _finite_scalar(orbit_rtol, "orbit_rtol")
    if orbit_atol < 0.0 or orbit_rtol < 0.0:
        raise ValueError("orbit tolerances must be non-negative.")

    def gap(log_radius, *, allow_singular_center=False):
        radius = _radius_from_log(log_radius)
        centrifugal = 0.5 * (angular_momentum / radius) ** 2
        if not np.isfinite(centrifugal):
            raise FloatingPointError(
                f"The centrifugal energy is not finite at r={radius:.17g}."
            )
        if reduced_energy is None:
            potential_value = _potential_value(
                potential, radius, allow_negative_infinity=allow_singular_center
            )
            if potential_value == -np.inf:
                return np.inf
            result = fsum((energy, -potential_value, -centrifugal))
        else:
            result = fsum((
                reduced_energy,
                -_finite_scalar(offset(radius), "potential offset"),
                -centrifugal,
            ))
        if not np.isfinite(result):
            raise FloatingPointError(
                f"The radial energy gap is not finite at r={radius:.17g}."
            )
        return float(result)

    if angular_momentum > 0.0:
        log_peak, _, orbit_scale = _find_effective_minimum(
            angular_momentum,
            potential,
            root_xtol,
            maxiter,
            max_expansions,
        )
    else:
        log_peak = 0.0
        peak_gap = gap(log_peak)
        if peak_gap <= 0.0:
            found_allowed = False
            left_potential = _potential_value(potential, 0.5)
            right_potential = _potential_value(potential, 2.0)
            direction = -1.0 if left_potential < right_potential else 1.0
            for expansion in range(1, max_expansions + 1):
                trial_log = direction * expansion * _LOG_RADIUS_STEP
                if not _LOG_RADIUS_MIN <= trial_log <= _LOG_RADIUS_MAX:
                    break
                trial_gap = gap(trial_log)
                if trial_gap > 0.0:
                    log_peak = trial_log
                    found_allowed = True
                    break
            if not found_allowed:
                raise ValueError(
                    "No allowed radial region was found for L = 0 within "
                    "the numerical radius range."
                )
        orbit_scale = max(abs(gap(log_peak)), np.finfo(float).tiny)

    peak_gap = gap(log_peak)
    orbit_tolerance = orbit_atol + orbit_rtol * orbit_scale

    if peak_gap < 0.0:
        if -peak_gap <= orbit_tolerance and angular_momentum > 0.0:
            raise FloatingPointError(
                "E lies below the computed effective-potential minimum by "
                "an amount that cannot be resolved reliably."
            )
        raise ValueError(
            "E is below the minimum of the effective potential; no radial "
            "orbit exists."
        )
    if peak_gap == 0.0:
        if angular_momentum > 0.0:
            peak_radius = _radius_from_log(log_peak)
            return peak_radius, peak_radius
        raise ValueError("No allowed radial interval exists for L = 0.")
    if angular_momentum > 0.0:
        peak_radius = _radius_from_log(log_peak)
        peak_potential = (_potential_value(potential, peak_radius)
                          if reduced_energy is None else
                          _finite_scalar(offset(peak_radius), "potential offset"))
        peak_energy = energy if reduced_energy is None else reduced_energy
        centrifugal = 0.5 * (angular_momentum / peak_radius) ** 2
        gap_resolution = 8.0 * _FLOAT_EPS * max(
            abs(peak_energy), abs(peak_potential), centrifugal
        )
        if peak_gap <= gap_resolution:
            raise FloatingPointError(
                "The positive radial energy gap above the effective "
                "minimum is too small to resolve with float64 potential "
                "values; the orbit cannot be treated as circular."
            )

    if angular_momentum == 0.0 and getattr(potential, "radial_center_allowed", False) is True:
        r_minus = 0.0
    else:
        r_minus = _expand_turning_root(
            log_peak,
            -1.0,
            gap,
            root_xtol,
            root_rtol,
            maxiter,
            max_expansions,
            center_allowed=(angular_momentum == 0.0),
        )
    r_plus = _expand_turning_root(
        log_peak,
        1.0,
        gap,
        root_xtol,
        root_rtol,
        maxiter,
        max_expansions,
    )

    if not (0.0 <= r_minus < r_plus):
        raise RuntimeError(
            "The computed turning points are not ordered: "
            f"r_minus={r_minus}, r_plus={r_plus}."
        )
    return r_minus, r_plus


def radial_action(
    E,
    L,
    potential,
    *,
    epsabs=1e-10,
    epsrel=1e-10,
    limit=200,
    root_xtol=1e-12,
    root_rtol=_MIN_BRENT_RTOL,
    maxiter=200,
    max_expansions=2048,
    orbit_atol=0.0,
    orbit_rtol=32.0 * _FLOAT_EPS,
    radicand_atol=0.0,
    radicand_rtol=1e-10,
    return_error=False,
):
    """Evaluate the normalized radial action for a supplied potential.

    The returned quantity is

        I_r = (1 / pi) integral[r_minus, r_plus] p_r dr.

    A sin-squared change of variable regularizes both turning points. For a
    nonzero pericentre the transformation is applied to log(r), which also
    resolves highly eccentric orbits with r_minus << r_plus. With
    ``return_error=True``, the second value is only the quadrature's
    estimated error, not a bound on the total action error from the potential,
    turning radii, and floating-point arithmetic.
    """
    energy = _finite_scalar(E, "E")
    angular_momentum = _finite_scalar(L, "L")
    if angular_momentum < 0.0:
        raise ValueError("L must be non-negative.")
    epsabs = _finite_scalar(epsabs, "epsabs")
    epsrel = _finite_scalar(epsrel, "epsrel")
    radicand_atol = _finite_scalar(radicand_atol, "radicand_atol")
    radicand_rtol = _finite_scalar(radicand_rtol, "radicand_rtol")
    if epsabs < 0.0 or epsrel < 0.0 or (epsabs == 0.0 and epsrel <= 0.0):
        raise ValueError("Integration tolerances must be non-negative and nonzero.")
    if radicand_atol < 0.0 or radicand_rtol < 0.0:
        raise ValueError("radicand tolerances must be non-negative.")
    if not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError("limit must be a positive integer.")

    r_minus, r_plus = find_turning_points(
        energy,
        angular_momentum,
        potential,
        root_xtol=root_xtol,
        root_rtol=root_rtol,
        maxiter=maxiter,
        max_expansions=max_expansions,
        orbit_atol=orbit_atol,
        orbit_rtol=orbit_rtol,
    )

    if r_minus == r_plus:
        return (0.0, 0.0) if return_error else 0.0

    offset, reduced_energy = _orbit_energy_coordinates(energy, potential)

    if r_minus > 0.0:
        log_r_minus = np.log(r_minus)
        log_width = np.log(r_plus) - log_r_minus

        def radius_and_jacobian(sine, cosine):
            radius = np.exp(log_r_minus + log_width * sine**2)
            jacobian = 2.0 * log_width * radius * sine * cosine
            return radius, jacobian

    else:
        radial_width = r_plus

        def radius_and_jacobian(sine, cosine):
            radius = radial_width * sine**2
            jacobian = 2.0 * radial_width * sine * cosine
            return radius, jacobian

    def transformed_integrand(theta):
        sine = np.sin(theta)
        cosine = np.cos(theta)
        radius, jacobian = radius_and_jacobian(sine, cosine)
        if jacobian == 0.0:
            return 0.0

        centrifugal_twice = (angular_momentum / radius) ** 2
        if not np.isfinite(centrifugal_twice):
            raise FloatingPointError(
                f"The centrifugal energy is not finite at r={radius:.17g}."
            )
        if reduced_energy is None:
            potential_value = _potential_value(potential, radius)
            components = (energy, -potential_value, -0.5 * centrifugal_twice)
        else:
            potential_offset = _finite_scalar(offset(radius), "potential offset")
            components = (reduced_energy, -potential_offset,
                          -0.5 * centrifugal_twice)
        radicand = 2.0 * fsum(components)
        if not np.isfinite(radicand):
            raise FloatingPointError(
                f"The radial-action radicand is not finite at r={radius:.17g}."
            )
        cancellation_error = 16.0 * _FLOAT_EPS * sum(map(abs, components))
        radicand_tolerance = (
            radicand_atol + radicand_rtol * abs(radicand)
            + cancellation_error
        )
        if not np.isfinite(radicand_tolerance):
            raise FloatingPointError(
                "The radial-action error scale exceeded the float range."
            )
        if radicand < -radicand_tolerance:
            raise FloatingPointError(
                "The radial-action radicand became negative beyond roundoff: "
                f"r={radius:.17g}, value={radicand:.17g}."
            )
        return np.sqrt(max(radicand, 0.0)) * jacobian

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            integral, error = quad(
                transformed_integrand,
                0.0,
                0.5 * np.pi,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=int(limit),
            )
        except IntegrationWarning as exc:
            raise RuntimeError(
                "The radial-action quadrature did not converge at "
                f"E={energy:.17g}, L={angular_momentum:.17g}, "
                f"r_minus={r_minus:.17g}, r_plus={r_plus:.17g}; "
                f"quadrature warning: {exc}"
            ) from exc

    action = float(integral / np.pi)
    action_error = float(error / np.pi)
    if not np.isfinite(action) or not np.isfinite(action_error):
        raise FloatingPointError("The radial-action result is not finite.")
    if action < 0.0:
        raise FloatingPointError("The radial action cannot be negative.")
    if action == 0.0:
        raise FloatingPointError(
            "The radial-action integral collapsed to zero despite distinct "
            "turning points; the orbit is below numerical resolution."
        )
    return (action, action_error) if return_error else action


def _automatic_energy_bracket(
    angular_momentum,
    potential,
    target_action,
    action_tolerance,
    initial_action,
    max_expansions,
    root_xtol,
    minimum_maxiter,
):
    """Bracket the target action using energies at positive trial radii."""
    def trial_energy(log_radius):
        radius = _radius_from_log(log_radius)
        try:
            energy = effective_potential(radius, angular_momentum, potential)
        except (ValueError, FloatingPointError) as exc:
            raise FloatingPointError(
                f"Cannot evaluate a finite trial energy at r={radius:.17g}: {exc}"
            ) from exc
        return energy

    def trial_action(energy, log_radius):
        try:
            return initial_action(energy)
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            radius = _radius_from_log(log_radius)
            raise RuntimeError(
                "Cannot evaluate the bound radial action for trial "
                f"r={radius:.17g}, E={energy:.17g}: {exc}"
            ) from exc

    if angular_momentum > 0.0:
        log_minimum, energy_min, _ = _find_effective_minimum(
            angular_momentum, potential, root_xtol, minimum_maxiter,
            max_expansions,
        )
        action_min = 0.0
        outward_start = log_minimum
    else:
        energy_at_one = trial_energy(0.0)
        energy_at_half = trial_energy(-_LOG_RADIUS_STEP)
        energy_at_two = trial_energy(_LOG_RADIUS_STEP)

        if energy_at_one <= energy_at_half and energy_at_one <= energy_at_two:
            # A radial orbit may have a finite-radius minimum.
            log_minimum, energy_min, _ = _find_effective_minimum(
                0.0, potential, root_xtol, minimum_maxiter, max_expansions
            )
            action_min = 0.0
            outward_start = log_minimum
        elif energy_at_two < energy_at_one:
            if energy_at_half < energy_at_one:
                raise ValueError(
                    "The radial potential is not single-welled near r=1."
                )
            # r=1 is inside the finite-radius minimum; move to its outer side.
            log_minimum, energy_min, _ = _find_effective_minimum(
                0.0, potential, root_xtol, minimum_maxiter, max_expansions
            )
            action_min = 0.0
            outward_start = log_minimum
        else:
            # r=1 lies on the outward branch. If its action is too large,
            # reduce the radius until the target is crossed or a minimum appears.
            action_at_one = trial_action(energy_at_one, 0.0)
            if (action_at_one > 0.0 or target_action == 0.0) and (
                abs(action_at_one - target_action)
                <= action_tolerance(action_at_one)
            ):
                return (energy_at_one, action_at_one), (energy_at_one, action_at_one)
            if action_at_one >= target_action:
                energy_max, action_max = energy_at_one, action_at_one
                previous_energy = energy_at_one
                for expansion in range(1, max_expansions + 1):
                    log_trial = -expansion * _LOG_RADIUS_STEP
                    try:
                        energy_trial = trial_energy(log_trial)
                    except OverflowError as exc:
                        raise RuntimeError(
                            "Could not bracket the target action before reaching "
                            "the numerical center."
                        ) from exc
                    if energy_trial >= previous_energy:
                        log_minimum, energy_min, _ = _find_effective_minimum(
                            0.0, potential, root_xtol, minimum_maxiter,
                            max_expansions,
                        )
                        action_min = 0.0
                        break
                    action_trial = trial_action(energy_trial, log_trial)
                    if action_trial <= target_action:
                        return (energy_trial, action_trial), (energy_max, action_max)
                    energy_max, action_max = energy_trial, action_trial
                    previous_energy = energy_trial
                else:
                    raise RuntimeError(
                        "Could not bracket the target action toward the center "
                        "within max_bracket_expansions."
                    )
                return (energy_min, action_min), (energy_max, action_max)

            energy_min, action_min = energy_at_one, action_at_one
            outward_start = 0.0

    if target_action == 0.0:
        return (energy_min, action_min), (energy_min, action_min)

    distinct_trial_energy = False
    for expansion in range(1, max_expansions + 1):
        log_trial = outward_start + expansion * _LOG_RADIUS_STEP
        try:
            energy_trial = trial_energy(log_trial)
        except OverflowError as exc:
            raise RuntimeError(
                "Could not bracket the target action before reaching the "
                "numerical outer radius."
            ) from exc
        if energy_trial < energy_min:
            raise FloatingPointError(
                "An outer trial apocenter has energy below the current "
                "lower bracket; the numerical effective-potential minimum "
                "or its one-well assumption is not reliable."
            )
        if energy_trial == energy_min:
            continue
        distinct_trial_energy = True
        action_trial = trial_action(energy_trial, log_trial)
        if action_trial >= target_action:
            return (energy_min, action_min), (energy_trial, action_trial)
        energy_min, action_min = energy_trial, action_trial

    if not distinct_trial_energy:
        raise FloatingPointError(
            "Expanding the trial apocenter did not produce a distinct "
            "float64 energy; the potential or energy scale is unresolved."
        )
    raise RuntimeError(
        "Could not bracket the target action with outer trial radii within "
        "max_bracket_expansions; the initial bound orbit may not attain it."
    )


class _CenteredPotential:
    """Use a supplied analytic Phi(r)-Phi(0) as the potential's energy zero."""

    def __init__(self, potential):
        self._offset = potential.offset_from_center
        self.radial_center_allowed = getattr(potential, "radial_center_allowed", False)

    def __call__(self, radius):
        return self._offset(radius)


def solve_initial_energy(
    E_prime,
    L_prime,
    potential,
    final_potential,
    E_bracket=None,
    *,
    energy_xtol=1e-12,
    energy_rtol=1e-10,
    maxiter=100,
    action_match_atol=0.0,
    action_match_rtol=1e-7,
    radial_action_kwargs=None,
    max_bracket_expansions=2048,
    _target_action=None,
    _refinement_attempt=False,
):
    """Solve E(E_prime, L_prime) by conserving action in two potentials.

    By default, bound trial orbits at geometrically expanded radii bracket
    the initial energy. An explicit ``E_bracket=(E_min, E_max)`` remains
    available; both endpoints must correspond to bound initial orbits.
    Turning radii are found automatically for every trial energy. The
    potential must have a single relevant radial well. The geometric search
    is limited by ``max_bracket_expansions``. Brent solves on a fraction of
    the energy interval so its relative tolerance is unaffected by adding
    a constant to the potential; excessively large offsets can still erase
    energy differences at floating-point precision. ``potential(r)`` must
    return finite, numerically stable float values over the searched radii.
    Its energy zero must match the initial distribution function: the returned
    E uses exactly that potential's energy convention. If the initial
    potential supplies a stable ``offset_from_center``, deeply bound orbits
    are solved and checked in E-Phi(0). Only the returned energy is rounded
    back to the original convention for evaluating the DF; recomputing its
    action from that rounded scalar may lose the requested radial-action
    precision for a nearly circular orbit.

    If the residual check fails, one additional energy solve uses refined
    action quadrature and turning-point tolerances. Its bracket and endpoint
    actions are recomputed with those settings; checking only the old energy
    cannot correct a root displaced by the original action evaluations.
    """
    energy_prime = _finite_scalar(E_prime, "E_prime")
    angular_momentum_prime = _finite_scalar(L_prime, "L_prime")
    if not callable(potential) or not callable(final_potential):
        raise TypeError("both potentials must be callable.")
    if angular_momentum_prime < 0.0:
        raise ValueError("L_prime must be non-negative.")

    energy_xtol, energy_rtol, maxiter = _validate_solver_tolerances(
        energy_xtol, energy_rtol, maxiter
    )
    if (not isinstance(max_bracket_expansions, (int, np.integer))
            or max_bracket_expansions < 1):
        raise ValueError("max_bracket_expansions must be a positive integer.")
    max_bracket_expansions = int(max_bracket_expansions)
    if type(_refinement_attempt) is not bool:
        raise TypeError("_refinement_attempt must be boolean")
    action_match_atol = _finite_scalar(
        action_match_atol, "action_match_atol"
    )
    action_match_rtol = _finite_scalar(
        action_match_rtol, "action_match_rtol"
    )
    if action_match_atol < 0.0 or action_match_rtol < 0.0:
        raise ValueError("action-match tolerances must be non-negative.")
    if radial_action_kwargs is None:
        radial_action_kwargs = {}
    elif not isinstance(radial_action_kwargs, dict):
        raise TypeError("radial_action_kwargs must be a dictionary or None.")
    else:
        radial_action_kwargs = dict(radial_action_kwargs)
    if "return_error" in radial_action_kwargs:
        raise ValueError(
            "radial_action_kwargs cannot override return_error."
        )

    target_action = (radial_action(
        energy_prime, angular_momentum_prime, final_potential,
        **radial_action_kwargs,
    ) if _target_action is None else _finite_scalar(_target_action, "target_action"))
    if target_action < 0.0:
        raise ValueError("target_action must be nonnegative")

    action_kwargs = dict(radial_action_kwargs)

    def action_tolerance(action):
        return action_match_atol + action_match_rtol * max(
            abs(action), abs(target_action)
        )

    target_resolution = 8.0 * _FLOAT_EPS * max(
        angular_momentum_prime, target_action
    )
    if (0.0 < target_action <= target_resolution
            and action_tolerance(target_action) < target_resolution):
        raise FloatingPointError(
            "The positive final radial action is only a few float64 "
            "spacings below the circular angular momentum; its requested "
            "relative accuracy cannot be certified from these inputs."
        )

    # Only nearly circular final orbits need tighter turning-point settings
    # from the outset. Quadrature is tightened only when its error estimate
    # or the final residual requires it.
    if (target_action > 0.0 and angular_momentum_prime > 0.0
            and target_action <= 1e-4 * (
                target_action + angular_momentum_prime
            )):
        action_kwargs["root_xtol"] = min(
            action_kwargs.get("root_xtol", 1e-12), 1e-14
        )

    def initial_action(energy):
        return radial_action(
            energy,
            angular_momentum_prime,
            potential,
            **action_kwargs,
        )

    if E_bracket is None:
        minimum_xtol, _, minimum_maxiter = _validate_solver_tolerances(
            action_kwargs.get("root_xtol", 1e-12),
            _MIN_BRENT_RTOL,
            action_kwargs.get("maxiter", 200),
        )
        (energy_min, action_min), (energy_max, action_max) = (
            _automatic_energy_bracket(
                angular_momentum_prime,
                potential,
                target_action,
                action_tolerance,
                initial_action,
                max_bracket_expansions,
                minimum_xtol,
                minimum_maxiter,
            )
        )
    else:
        try:
            energy_min, energy_max = E_bracket
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "E_bracket must contain exactly two finite energies."
            ) from exc
        energy_min = _finite_scalar(energy_min, "E_bracket[0]")
        energy_max = _finite_scalar(energy_max, "E_bracket[1]")
        if energy_max <= energy_min:
            raise ValueError("E_bracket must be strictly increasing.")
        action_min = initial_action(energy_min)
        action_max = initial_action(energy_max)

    # Forming Phi(0) + epsilon inside every root evaluation quantizes epsilon
    # on the much coarser absolute-energy grid. Center the entire solve, not
    # just the subtraction in radial_action. Keep the original zero near
    # escape, where centering would instead lose the small binding energy.
    offset = getattr(potential, "offset_from_center", None)
    if callable(offset):
        center = _finite_scalar(potential.central_potential, "central potential")
        if center != 0.0 and abs(energy_max - center) < 0.5 * abs(center):
            centered = _CenteredPotential(potential)
            shifted_bracket = (None if E_bracket is None else (
                fsum((energy_min, -center)), fsum((energy_max, -center)),
            ))
            relative_energy = solve_initial_energy(
                0.0, angular_momentum_prime, centered, centered,
                E_bracket=shifted_bracket,
                energy_xtol=energy_xtol, energy_rtol=energy_rtol,
                maxiter=maxiter, action_match_atol=action_match_atol,
                action_match_rtol=action_match_rtol,
                radial_action_kwargs=action_kwargs,
                max_bracket_expansions=max_bracket_expansions,
                _target_action=target_action,
                _refinement_attempt=_refinement_attempt,
            )
            absolute_energy = fsum((center, relative_energy))
            if relative_energy != 0.0 and absolute_energy == center:
                raise FloatingPointError(
                    "The mapped energy rounds to the central potential; "
                    "it cannot be passed to the DF as an absolute float64 energy."
                )
            return absolute_energy

    residual_min = action_min - target_action
    residual_max = action_max - target_action

    matched_min = (action_min > 0.0 or target_action == 0.0) and (
        abs(residual_min) <= action_tolerance(action_min)
    )
    matched_max = (action_max > 0.0 or target_action == 0.0) and (
        abs(residual_max) <= action_tolerance(action_max)
    )
    if matched_min:
        energy = energy_min
    elif matched_max:
        energy = energy_max
    else:
        if not ((residual_min < 0.0 < residual_max)
                or (residual_max < 0.0 < residual_min)):
            raise ValueError(
                "The initial-energy interval does not bracket radial-action "
                "conservation: "
                f"residual(E_min)={residual_min:.17g}, "
                f"residual(E_max)={residual_max:.17g}."
            )

        energy_span = energy_max - energy_min
        if not np.isfinite(energy_span) or energy_span <= 0.0:
            raise FloatingPointError(
                "The initial-energy bracket has no finite, resolvable width."
            )
        action_span = abs(action_max - action_min)
        target_tolerance = action_tolerance(target_action)
        scaled_xtol = min(
            energy_xtol / energy_span,
            0.1 * target_tolerance / action_span,
        )
        scaled_xtol = max(np.finfo(float).tiny, scaled_xtol)
        scaled_rtol = max(
            _MIN_BRENT_RTOL,
            min(energy_rtol, 0.1 * target_tolerance / action_span),
        )

        def residual_at_fraction(fraction):
            if fraction == 0.0:
                return residual_min
            if fraction == 1.0:
                return residual_max
            return initial_action(energy_min + fraction * energy_span) - target_action

        solution = root_scalar(
            residual_at_fraction,
            bracket=(0.0, 1.0),
            method="brentq",
            xtol=scaled_xtol,
            rtol=scaled_rtol,
            maxiter=maxiter,
        )
        if not solution.converged:
            raise RuntimeError(
                "The initial-energy solve did not converge: "
                f"{solution.flag}."
            )
        energy = float(energy_min + solution.root * energy_span)

    if target_action == 0.0 and matched_min and action_min == 0.0:
        return energy

    def checked_action(candidate, kwargs):
        action, quadrature_error = radial_action(
            candidate,
            angular_momentum_prime,
            potential,
            return_error=True,
            **kwargs,
        )
        tolerance = action_tolerance(action)
        valid = (
            (target_action == 0.0 or action > 0.0)
            and abs(action - target_action) <= tolerance
            and quadrature_error <= tolerance
        )
        return valid, action, quadrature_error, tolerance

    try:
        valid, action, quadrature_error, tolerance = checked_action(
            energy, action_kwargs
        )
    except (FloatingPointError, RuntimeError) as exc:
        raise RuntimeError(
            "Insufficient precision while verifying the initial radial "
            f"action at E={energy:.17g}: {exc}"
        ) from exc
    if valid:
        return energy
    first_action = action

    # One bounded retry can expose inadequate quadrature or root resolution.
    # Agreement between retries is not treated as a proof of total accuracy.
    refined_kwargs = dict(action_kwargs)
    refined_kwargs["epsabs"] = min(
        refined_kwargs.get("epsabs", 1e-10) * 0.1,
        max(tolerance, np.finfo(float).tiny),
    )
    refined_kwargs["epsrel"] = min(
        refined_kwargs.get("epsrel", 1e-10) * 0.1, 1e-12
    )
    refined_kwargs["root_xtol"] = min(
        refined_kwargs.get("root_xtol", 1e-12) * 0.1, 1e-14
    )
    try:
        valid, action, quadrature_error, tolerance = checked_action(
            energy, refined_kwargs
        )
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "Insufficient precision while refining the initial radial "
            f"action at E={energy:.17g}: {exc}"
        ) from exc
    if valid and abs(action - first_action) <= tolerance:
        return energy

    if not _refinement_attempt:
        # Tightening a quadrature can move the root. Solve the conservation
        # equation again instead of searching only adjacent float64 values
        # of an energy computed with the previous quadrature. Rebuild an
        # automatic bracket, or reevaluate the caller's explicit endpoints;
        # never reuse cached endpoint actions from the coarser solve.
        return solve_initial_energy(
            energy_prime, angular_momentum_prime, potential, final_potential,
            E_bracket=E_bracket,
            energy_xtol=energy_xtol, energy_rtol=energy_rtol,
            maxiter=maxiter, action_match_atol=action_match_atol,
            action_match_rtol=action_match_rtol,
            radial_action_kwargs=refined_kwargs,
            max_bracket_expansions=max_bracket_expansions,
            _target_action=target_action,
            _refinement_attempt=True,
        )

    # Inspect neighboring representable energies only after a failed check.
    # Their actions cannot by themselves certify the accuracy of potential(r).
    for neighbor in (
        np.nextafter(energy, -np.inf),
        np.nextafter(energy, np.inf),
    ):
        if not energy_min <= neighbor <= energy_max:
            continue
        try:
            neighbor_valid, _, _, _ = checked_action(neighbor, refined_kwargs)
        except (FloatingPointError, RuntimeError, ValueError):
            continue
        if neighbor_valid:
            return float(neighbor)

    if target_action > 0.0 and action == 0.0:
        raise RuntimeError(
            "Insufficient precision: the positive target radial action "
            "collapsed to zero at the computed initial energy."
        )
    raise RuntimeError(
        "Radial-action conservation failed after a refined energy solve "
        "and checks of neighboring float64 energies; potential, turning-point "
        "and quadrature errors are not bounded by the quadrature estimate alone: "
        f"E={energy:.17g}, L={angular_momentum_prime:.17g}, "
        f"target_action={target_action:.17g}, "
        f"residual={action - target_action:.17g}, "
        f"residual_before_quadrature_refinement={first_action - target_action:.17g}, "
        f"quadrature_error_estimate={quadrature_error:.17g}, "
        f"tolerance={tolerance:.17g}."
    )


class InitialActionTable:
    """Interpolate E_i(I_r, L), certifying each query against the initial action.

    The table is fixed for the entire halo-growth calculation. An interpolated
    energy is accepted only when a direct radial-action evaluation satisfies
    the same action tolerance as ``solve_initial_energy``. Otherwise that
    solver is used with the already-computed final action.
    """

    def __init__(self, potential, central_potential, log_scales, circularities,
                 binding_logits):
        self.potential = potential
        self.binding_scale = -float(central_potential)
        self.log_scales = np.asarray(log_scales, dtype=float)
        self.circularities = np.asarray(circularities, dtype=float)
        self.binding_logits = np.asarray(binding_logits, dtype=float)
        if (not callable(potential) or not np.isfinite(self.binding_scale)
                or self.binding_scale <= 0.0 or len(self.log_scales) < 4
                or len(self.circularities) < 4
                or np.any(np.diff(self.log_scales) <= 0.0)
                or np.any(np.diff(self.circularities) <= 0.0)
                or self.binding_logits.shape != (
                    len(self.log_scales), len(self.circularities)
                ) or not np.all(np.isfinite(self.binding_logits))):
            raise ValueError("invalid initial action table")
        self._interpolator = RectBivariateSpline(
            self.log_scales, self.circularities, self.binding_logits,
        )
        self.accepted = 0
        self.fallback = 0

    def initial_energy(self, target_action, angular_momentum, *, options):
        total = target_action + angular_momentum
        if total > 0.0:
            log_scale = np.log(total)
            circularity = angular_momentum / total
            if self.log_scales[0] <= log_scale <= self.log_scales[-1]:
                logit_binding = float(self._interpolator(log_scale, circularity)[0, 0])
                candidate = -self.binding_scale * float(expit(logit_binding))
                action_kwargs = dict(options.get("radial_action_kwargs", {}))
                try:
                    action, error = radial_action(
                        candidate, angular_momentum, self.potential,
                        return_error=True, **action_kwargs,
                    )
                    tolerance = options.get("action_match_atol", 0.0) + (
                        options.get("action_match_rtol", 1e-7)
                        * max(abs(action), abs(target_action))
                    )
                    if (action > 0.0 or target_action == 0.0) and (
                        abs(action - target_action) <= tolerance and error <= tolerance
                    ):
                        self.accepted += 1
                        return candidate
                except (ArithmeticError, RuntimeError, ValueError):
                    pass
        self.fallback += 1
        return solve_initial_energy(
            0.0, angular_momentum, self.potential, self.potential,
            _target_action=target_action, **options,
        )

    def map(self, energy_prime, angular_momentum, final_potential, *, options):
        target_action = radial_action(
            energy_prime, angular_momentum, final_potential,
            **options.get("radial_action_kwargs", {}),
        )
        return self.initial_energy(target_action, angular_momentum, options=options)


def build_initial_action_table(potential, central_potential, minimum_scale,
                               maximum_scale, *, points_per_decade=6,
                               circularity_points=25, map_kwargs=None,
                               _energy_evaluator=None):
    """Sample the fixed inverse initial-action surface over a finite domain."""
    minimum_scale, maximum_scale = float(minimum_scale), float(maximum_scale)
    if (not np.isfinite(minimum_scale) or not np.isfinite(maximum_scale)
            or minimum_scale <= 0.0 or maximum_scale <= minimum_scale):
        raise ValueError("initial action scales must be positive and increasing")
    if type(points_per_decade) is not int or points_per_decade < 2:
        raise ValueError("points_per_decade must be at least two")
    if type(circularity_points) is not int or circularity_points < 4:
        raise ValueError("circularity_points must be at least four")
    if _energy_evaluator is not None and not callable(_energy_evaluator):
        raise TypeError("initial energy evaluator must be callable")
    options = dict(map_kwargs or {})
    options.pop("initial_action_table", None)
    logarithmic_span = math.log10(maximum_scale / minimum_scale)
    scale_count = max(4, math.ceil(logarithmic_span * points_per_decade) + 1)
    log_scales = np.linspace(math.log(minimum_scale), math.log(maximum_scale), scale_count)
    circularities = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, circularity_points)))
    binding_scale = -float(central_potential)
    if not np.isfinite(binding_scale) or binding_scale <= 0.0:
        raise ValueError("central_potential must be negative and finite")
    targets = []
    for log_scale in log_scales:
        scale = math.exp(float(log_scale))
        for circularity in circularities:
            angular_momentum = scale * float(circularity)
            target_action = scale - angular_momentum
            targets.append((target_action, angular_momentum))
    if _energy_evaluator is None:
        energies = [
            solve_initial_energy(
                0.0, angular_momentum, potential, potential,
                _target_action=target_action, **options,
            )
            for target_action, angular_momentum in targets
        ]
    else:
        energies = _energy_evaluator(targets, options)
    energies = np.asarray(energies, dtype=float)
    if energies.shape != (len(targets),) or not np.all(np.isfinite(energies)):
        raise ArithmeticError("initial action table returned invalid energies")
    binding = -energies / binding_scale
    if np.any(binding <= 0.0) or np.any(binding >= 1.0):
        raise ArithmeticError("initial action table energy left the bound interval")
    binding_logits = logit(binding).reshape(scale_count, circularity_points)
    return InitialActionTable(
        potential, central_potential, log_scales, circularities, binding_logits,
    )
