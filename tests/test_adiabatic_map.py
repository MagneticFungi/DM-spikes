"""Analytical checks for the automatic radial turning-point search."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import dm_spikes.adiabatic_map as adiabatic_map
from dm_spikes.adiabatic_map import (
    find_turning_points,
    radial_action,
    radial_action_final_kepler,
    solve_initial_energy,
)


def harmonic_potential(omega):
    return lambda radius: 0.5 * (omega * radius) ** 2


def test_no_radial_bounds_in_public_signatures():
    for function in (find_turning_points, radial_action, solve_initial_energy):
        assert "r_bounds" not in inspect.signature(function).parameters


def test_initial_energy_bracket_is_optional():
    assert inspect.signature(solve_initial_energy).parameters["E_bracket"].default is None


@pytest.mark.parametrize("angular_momentum", [0.0, 0.4, 0.999999])
def test_kepler_turning_points_and_action(angular_momentum):
    energy = -0.5
    eccentricity = np.sqrt(1.0 + 2.0 * energy * angular_momentum**2)
    expected_radii = (1.0 - eccentricity, 1.0 + eccentricity)
    potential = lambda radius: -1.0 / radius

    assert find_turning_points(energy, angular_momentum, potential) == pytest.approx(
        expected_radii, rel=2e-9, abs=2e-12
    )
    numerical_action = radial_action(energy, angular_momentum, potential)
    exact_action = radial_action_final_kepler(
        energy, angular_momentum, 1.0, G=1.0
    )
    assert numerical_action == pytest.approx(
        exact_action, rel=2e-9, abs=2e-12
    )


@pytest.mark.parametrize("angular_momentum", [0.0, 1e-8, 0.4, 1.499999999])
def test_harmonic_turning_points_and_action(angular_momentum):
    omega = 2.0
    energy = 3.0
    discriminant = np.sqrt(energy**2 - (omega * angular_momentum) ** 2)
    expected_radii = (
        np.sqrt((energy - discriminant) / omega**2),
        np.sqrt((energy + discriminant) / omega**2),
    )
    if angular_momentum < 1e-6:
        # This form avoids cancellation in the inner root.
        expected_radii = (
            angular_momentum / (omega * expected_radii[1]),
            expected_radii[1],
        )

    potential = harmonic_potential(omega)
    actual_radii = find_turning_points(energy, angular_momentum, potential)
    assert actual_radii == pytest.approx(expected_radii, rel=2e-7, abs=2e-12)
    expected_action = 0.5 * (energy / omega - angular_momentum)
    assert radial_action(energy, angular_momentum, potential) == pytest.approx(
        expected_action, rel=2e-7, abs=2e-12
    )


def test_circular_harmonic_orbit():
    potential = harmonic_potential(2.0)
    circular_radius = np.sqrt(0.4 / 2.0)
    assert find_turning_points(0.8, 0.4, potential) == pytest.approx(
        (circular_radius, circular_radius), rel=2e-9
    )
    assert radial_action(0.8, 0.4, potential) == 0.0


@pytest.mark.parametrize("radius_scale", [1e-100, 1e100])
def test_turning_points_across_radial_scales(radius_scale):
    omega = 1.0 / radius_scale
    angular_momentum = radius_scale
    energy = 2.0
    expected_inner = radius_scale * np.sqrt(2.0 - np.sqrt(3.0))
    expected_outer = radius_scale * np.sqrt(2.0 + np.sqrt(3.0))
    assert find_turning_points(
        energy, angular_momentum, harmonic_potential(omega)
    ) == pytest.approx((expected_inner, expected_outer), rel=2e-8)


def test_zero_angular_momentum_with_finite_inner_turning_point():
    potential = lambda radius: (radius - 1.0) ** 2
    assert find_turning_points(0.25, 0.0, potential) == pytest.approx(
        (0.5, 1.5), rel=2e-10
    )


def test_unbound_orbit_reports_missing_outer_turn():
    with pytest.raises(RuntimeError, match="outer turning point.*may be unbound"):
        find_turning_points(0.5, 0.4, lambda radius: -1.0 / radius)


def test_energy_below_effective_minimum_is_rejected():
    with pytest.raises(ValueError, match="below the minimum"):
        find_turning_points(0.5, 0.4, harmonic_potential(2.0))


def test_circular_classification_is_independent_of_potential_offset():
    offset = 1e8
    potential = lambda radius: offset + 2.0 * radius**2
    r_minus, r_plus = find_turning_points(offset + 0.80001, 0.4, potential)
    assert r_minus < r_plus


def test_flat_effective_potential_is_not_reported_as_circular():
    with pytest.raises((ValueError, RuntimeError, FloatingPointError), match="minimum|flat"):
        find_turning_points(0.0, 0.4, lambda radius: 0.0)


def test_nonfinite_potential_is_rejected():
    with pytest.raises(ValueError, match=r"potential\(r\) must be finite"):
        find_turning_points(1.0, 0.4, lambda radius: np.nan)


@pytest.mark.parametrize("mu", [1e-100, 1e100])
def test_radial_kepler_orbit_across_scales(mu):
    potential = lambda radius: -mu / radius
    assert find_turning_points(-0.5, 0.0, potential) == pytest.approx(
        (0.0, 2.0 * mu), rel=2e-8
    )
    assert radial_action(-0.5, 0.0, potential) / mu == pytest.approx(1.0, rel=2e-9)


def test_expansion_limit_is_reported():
    with pytest.raises(RuntimeError, match="max_expansions"):
        find_turning_points(
            3.0, 0.4, harmonic_potential(2.0), max_expansions=1
        )


def test_energy_mapping_conserves_action_without_radial_bounds():
    omega = 2.0
    energy_prime = -0.5
    angular_momentum = 0.4
    target_action = radial_action_final_kepler(
        energy_prime, angular_momentum, 1.0, G=1.0
    )
    expected_energy = omega * (2.0 * target_action + angular_momentum)
    actual_energy = solve_initial_energy(
        energy_prime,
        angular_momentum,
        1.0,
        harmonic_potential(omega),
        (0.800001, 5.0),
        G=1.0,
    )
    assert actual_energy == pytest.approx(expected_energy, rel=2e-9)


def test_energy_mapping_allows_negative_initial_energy():
    energy = solve_initial_energy(
        -0.5,
        0.4,
        1.0,
        lambda radius: -1.0 / radius,
        (-1.0, -0.25),
        G=1.0,
    )
    assert energy == pytest.approx(-0.5, rel=2e-9)


@pytest.mark.parametrize("radius_scale", [1e-100, 1.0, 1e100])
def test_automatic_energy_bracket_with_angular_momentum(radius_scale):
    potential = lambda radius: 0.5 * (radius / radius_scale) ** 2
    energy = solve_initial_energy(
        -0.5, 0.4 * radius_scale, radius_scale, potential, G=1.0
    )
    assert energy == pytest.approx(1.6, rel=2e-9)


@pytest.mark.parametrize("radius_scale", [1e-100, 1e100])
def test_automatic_radial_energy_bracket_across_scales(radius_scale):
    potential = lambda radius: 0.5 * (radius / radius_scale) ** 2
    energy = solve_initial_energy(-0.5, 0.0, radius_scale, potential, G=1.0)
    assert energy == pytest.approx(2.0, rel=2e-9)


def test_automatic_radial_kepler_mapping():
    energy = solve_initial_energy(-0.5, 0.0, 1.0, lambda r: -1.0 / r, G=1.0)
    assert energy == pytest.approx(-0.5, rel=2e-9)


@pytest.mark.parametrize("mu", [1e-100, 1e100])
def test_automatic_radial_kepler_mapping_across_scales(mu):
    energy = solve_initial_energy(
        -0.5, 0.0, mu, lambda radius: -mu / radius, G=1.0
    )
    assert energy == pytest.approx(-0.5, rel=2e-9)


def test_automatic_kepler_mapping_tightens_a_wide_energy_bracket():
    energy = solve_initial_energy(
        -0.5, 0.4, 1e100, lambda radius: -1e100 / radius, G=1.0
    )
    assert energy == pytest.approx(-0.5, rel=2e-9)


def test_automatic_radial_mapping_with_finite_radius_minimum():
    potential = lambda radius: (radius - 1.0) ** 2
    target_action = radial_action(0.25, 0.0, potential)
    energy_prime = -0.5 / target_action**2
    energy = solve_initial_energy(energy_prime, 0.0, 1.0, potential, G=1.0)
    assert energy == pytest.approx(0.25, rel=2e-9)


@pytest.mark.parametrize("offset", [-1000.0, 1000.0])
def test_automatic_bracket_is_invariant_under_potential_offset(offset):
    potential = lambda radius: offset + 2.0 * radius**2
    energy = solve_initial_energy(-0.5, 0.4, 1.0, potential, G=1.0)
    assert energy == pytest.approx(offset + 3.2, rel=2e-12)


@pytest.mark.parametrize("action_gap", [0.0, 1e-6])
def test_automatic_bracket_near_circular_action(action_gap):
    angular_momentum = 1.0 - action_gap
    energy = solve_initial_energy(
        -0.5, angular_momentum, 1.0, harmonic_potential(2.0), G=1.0
    )
    expected = 2.0 * (2.0 * action_gap + angular_momentum)
    assert energy == pytest.approx(expected, rel=2e-10)


def test_explicit_endpoint_matching_uses_its_own_action_scale():
    energy = solve_initial_energy(
        -0.5, 0.0, 1.0, harmonic_potential(2.0),
        E_bracket=(3.0, 1e12), G=1.0,
    )
    assert energy == pytest.approx(4.0, rel=2e-9)


def test_automatic_energy_bracket_expansion_limit():
    with pytest.raises(RuntimeError, match="max_bracket_expansions"):
        solve_initial_energy(
            -0.5, 0.4, 1.0, harmonic_potential(2.0),
            G=1.0, max_bracket_expansions=1,
        )


def test_automatic_energy_bracket_reports_nonfinite_potential():
    potential = lambda radius: np.nan if radius > 1.0 else 2.0 * radius**2
    with pytest.raises(FloatingPointError, match="finite trial energy"):
        solve_initial_energy(-0.5, 0.0, 1.0, potential, G=1.0)


def test_automatic_energy_bracket_reports_quadrature_failure(monkeypatch):
    def fail_quadrature(*args, **kwargs):
        raise RuntimeError("The radial-action quadrature did not converge.")

    monkeypatch.setattr(adiabatic_map, "radial_action", fail_quadrature)
    with pytest.raises(RuntimeError, match="quadrature did not converge"):
        solve_initial_energy(-0.5, 0.0, 1.0, harmonic_potential(2.0), G=1.0)


def test_initial_energy_root_checks_action_residual(monkeypatch):
    monkeypatch.setattr(
        adiabatic_map, "root_scalar",
        lambda *args, **kwargs: SimpleNamespace(converged=True, root=0.5),
    )
    with pytest.raises(RuntimeError, match="Insufficient precision to conserve radial action"):
        solve_initial_energy(
            -0.5, 0.0, 1.0, harmonic_potential(2.0),
            E_bracket=(1.0, 5.0), G=1.0,
        )


@pytest.mark.parametrize("action_gap", [1e-4, 1e-6, 1e-8, 1e-10])
def test_resolvable_near_circular_kepler_mapping(action_gap):
    potential = lambda radius: -1.0 / radius
    angular_momentum = 1.0 - action_gap
    target_action = radial_action_final_kepler(
        -0.5, angular_momentum, 1.0, G=1.0
    )
    r_minus, r_plus = find_turning_points(
        -0.5, angular_momentum, potential
    )
    assert target_action > 0.0
    assert 0.0 < r_minus < r_plus
    assert radial_action(-0.5, angular_momentum, potential) > 0.0
    assert solve_initial_energy(
        -0.5, angular_momentum, 1.0, potential, G=1.0
    ) == pytest.approx(-0.5, rel=2e-9)


def test_small_positive_kepler_action_is_not_called_circular():
    potential = lambda radius: -1.0 / radius
    angular_momentum = 1.0 - 1e-12
    r_minus, r_plus = find_turning_points(-0.5, angular_momentum, potential)
    assert r_minus < r_plus
    assert radial_action(-0.5, angular_momentum, potential) > 0.0
    with pytest.raises(RuntimeError, match="Insufficient precision"):
        solve_initial_energy(-0.5, angular_momentum, 1.0, potential, G=1.0)


def test_kepler_exactly_circular_mapping():
    potential = lambda radius: -1.0 / radius
    assert find_turning_points(-0.5, 1.0, potential) == (1.0, 1.0)
    assert radial_action(-0.5, 1.0, potential) == 0.0
    assert solve_initial_energy(-0.5, 1.0, 1.0, potential, G=1.0) == -0.5


def test_kepler_energy_below_effective_minimum():
    with pytest.raises(ValueError, match="below the minimum"):
        find_turning_points(-0.5001, 1.0, lambda radius: -1.0 / radius)


def test_kepler_angular_momentum_above_circular_is_not_clipped():
    with pytest.raises((FloatingPointError, ValueError), match="circular"):
        radial_action_final_kepler(-0.5, 1.0 + 1e-13, 1.0, G=1.0)


def test_float64_gap_too_small_to_resolve_is_reported():
    angular_momentum = np.nextafter(1.0, 0.0)
    with pytest.raises(FloatingPointError, match="too small to resolve"):
        find_turning_points(
            -0.5, angular_momentum, lambda radius: -1.0 / radius
        )


def test_final_action_at_input_float64_limit_is_reported():
    angular_momentum = np.nextafter(1.0, 0.0)
    with pytest.raises(FloatingPointError, match="final Kepler radial action"):
        solve_initial_energy(
            -0.5, angular_momentum, 1.0,
            lambda radius: -1.0 / radius, G=1.0,
        )


def test_matching_endpoint_still_checks_quadrature_error(monkeypatch):
    def inaccurate_action(energy, angular_momentum, potential, *, return_error=False, **kwargs):
        return (1.0, 0.1) if return_error else 1.0

    monkeypatch.setattr(adiabatic_map, "radial_action", inaccurate_action)
    with pytest.raises(RuntimeError, match="Insufficient precision"):
        solve_initial_energy(
            -0.5, 0.0, 1.0, harmonic_potential(2.0),
            E_bracket=(1.0, 2.0), G=1.0,
        )


def test_positive_target_cannot_match_a_zero_action_endpoint(monkeypatch):
    def zero_action(energy, angular_momentum, potential, *, return_error=False, **kwargs):
        return (0.0, 0.0) if return_error else 0.0

    monkeypatch.setattr(adiabatic_map, "radial_action", zero_action)
    with pytest.raises(ValueError, match="does not bracket"):
        solve_initial_energy(
            -0.5, 0.0, 1.0, harmonic_potential(2.0),
            E_bracket=(1.0, 2.0), G=1.0, action_match_atol=2.0,
        )
