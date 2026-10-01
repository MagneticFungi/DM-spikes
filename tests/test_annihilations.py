"""Harmonic saturation and power-law core-radius placement."""

import math

import numpy as np
import pytest

import dm_spikes as dm
import dm_spikes.annihilations as annihilations
from dm_spikes.power_law_profile import D, gamma_spike, rho_D, rho_R, spike_radius


def test_harmonic_saturation_scalar_array_and_large_density():
    assert dm.rho_spike(10.0, 10.0, 1.0, bh_age=2.0) == pytest.approx(10.0 / 3.0)
    actual = dm.rho_spike(np.array([0.0, 1.0, 10.0, math.inf]), 10.0, 1.0, bh_age=2.0)
    assert actual == pytest.approx([0.0, 5.0 / 6.0, 10.0 / 3.0, 5.0])
    assert dm.rho_spike(1e308, 0.1, 1.0, bh_age=1.0) == pytest.approx(0.1)


def test_saturation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="m"):
        dm.rho_spike(1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="rho_cusp"):
        dm.rho_spike(-1.0, 1.0, 1.0)
    with pytest.raises(ArithmeticError, match="saturation"):
        dm.rho_spike(1.0, 1.0, 1e-300, bh_age=1e-300)


def test_core_radius_uses_power_law_normalization_and_is_not_in_annihilations():
    mass_bh = 4e6
    gamma = 1.0
    particle_mass = 50.0
    cross_section = 2.0
    age = 5.0
    radius_spike = spike_radius(mass_bh, gamma, rho_D(gamma), D)
    density_spike = rho_R(gamma, rho_D(gamma), D, radius_spike)
    density_sat = particle_mass / (cross_section * age)
    expected = radius_spike * (density_spike / density_sat) ** (1.0 / gamma_spike(gamma))
    assert dm.core_radius(mass_bh, particle_mass, gamma, cross_section, age) == pytest.approx(expected)
    assert not hasattr(annihilations, "core_radius")
    assert annihilations.__all__ == ["rho_spike"]
