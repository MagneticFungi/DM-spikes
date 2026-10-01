"""Composition and phase-space checks for the numerical final profile."""

import math

import pytest

import dm_spikes.final_profile as final_profile
from dm_spikes.final_profile import (
    energy_lower_bound,
    make_final_distribution,
    make_final_profile,
    rho_prime_at_r,
    rho_prime_profile,
    schwarzschild_radius,
)


def test_final_distribution_uses_mapped_energy_without_changing_its_zero():
    shifted_potential = lambda radius: 7.0 - 1.0 / radius
    final_df = make_final_distribution(
        lambda initial_energy: initial_energy,
        shifted_potential, 1.0, G=1.0,
    )
    assert final_df(-0.5, 0.5) == pytest.approx(6.5, rel=1e-9)


def test_constant_final_df_has_analytic_capture_density():
    gravity = mass = 1.0
    light_speed = 1000.0
    radius = 1.0
    lower_energy = energy_lower_bound(radius, mass, G=gravity, clight=light_speed)
    expected = 8.0 * math.pi * math.sqrt(2.0) / 3.0 * (-lower_energy) ** 1.5
    actual = rho_prime_at_r(
        lambda energy, angular_momentum: 1.0,
        radius, mass, G=gravity, clight=light_speed,
    )
    assert actual == pytest.approx(expected, rel=1e-9)
    capture_radius = 4.0 * schwarzschild_radius(mass, G=gravity, clight=light_speed)
    assert rho_prime_at_r(
        lambda energy, angular_momentum: 1.0,
        capture_radius, mass, G=gravity, clight=light_speed,
    ) == 0.0
    profile = rho_prime_profile(
        lambda energy, angular_momentum: 1.0,
        [capture_radius, radius], mass, G=gravity, clight=light_speed,
    )
    assert profile == pytest.approx([0.0, expected], rel=1e-9)


def test_make_final_profile_composes_map_df_and_phase_space(monkeypatch):
    seen = []

    def mapped_energy(energy, angular_momentum, mass, potential, *, G, **kwargs):
        seen.append((energy, angular_momentum, mass, potential, G))
        return energy + 7.0

    monkeypatch.setattr(final_profile, "solve_initial_energy", mapped_energy)
    initial_potential = lambda radius: radius
    final_density = make_final_profile(
        lambda initial_energy: initial_energy + 2.0,
        initial_potential, 1.0, G=1.0, clight=1000.0,
    )
    expected = rho_prime_at_r(
        lambda energy, angular_momentum: energy + 9.0,
        1.0, 1.0, G=1.0, clight=1000.0,
    )
    assert final_density(1.0) == pytest.approx(expected, rel=1e-9)
    assert seen and all(item[3] is initial_potential and item[4] == 1.0 for item in seen)


def test_final_distribution_rejects_invalid_inputs_and_propagates_mapping_failure():
    with pytest.raises(TypeError, match="callable"):
        make_final_distribution(None, lambda radius: radius, 1.0)
    with pytest.raises(ValueError, match="map_kwargs"):
        make_final_distribution(lambda energy: 1.0, lambda radius: radius, 1.0,
                                map_kwargs={"G": 2.0})
    final_df = make_final_distribution(
        lambda energy: 1.0, lambda radius: -1.0 / radius, 1.0, G=1.0
    )
    with pytest.raises(ValueError, match="L_prime"):
        final_df(-0.5, -0.1)
