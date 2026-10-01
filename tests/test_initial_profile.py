"""Generic spherical potential against analytic density--potential pairs."""

import math

import pytest

from dm_spikes.constants import G
from dm_spikes.initial_profile import make_initial_potential


@pytest.mark.parametrize("gamma", [0.7, 1.0, 1.95])
def test_power_law_confining_potential_and_derivatives(gamma):
    r_d = 8.5e3
    rho_d = 0.0062 * (1.0 - gamma / 3.0)
    amplitude = 4.0 * math.pi * G * rho_d * r_d**2 / (
        (3.0 - gamma) * (2.0 - gamma)
    )
    density = lambda radius: rho_d * (radius / r_d) ** (-gamma)
    potential = make_initial_potential(density, "confining", r_ref=r_d)

    assert potential(r_d) == 0.0
    for ratio in (1e-6, 1e-3, 0.1, 1.0 - 1e-8, 1.0 + 1e-8, 2.0, 1e3, 1e6):
        radius = r_d * ratio
        expected_phi = amplitude * math.expm1((2.0 - gamma) * math.log(ratio))
        expected_prime = (
            amplitude * (2.0 - gamma) / r_d * ratio ** (1.0 - gamma)
        )
        expected_second = (
            amplitude * (2.0 - gamma) * (1.0 - gamma)
            / r_d**2 * ratio ** (-gamma)
        )
        assert potential(radius) == pytest.approx(expected_phi, rel=3e-7, abs=1e-11)
        assert potential.prime(radius) == pytest.approx(expected_prime, rel=3e-7)
        if gamma != 1.0:
            assert potential.second(radius) == pytest.approx(
                expected_second, rel=2e-6, abs=1e-12
            )


def test_near_reference_avoids_cancellation_for_steep_power_law():
    gamma = 1.95
    r_d = 8.5e3
    rho_d = 0.0062 * (1.0 - gamma / 3.0)
    amplitude = 4.0 * math.pi * G * rho_d * r_d**2 / (
        (3.0 - gamma) * (2.0 - gamma)
    )
    potential = make_initial_potential(
        lambda radius: rho_d * (radius / r_d) ** (-gamma),
        "confining", r_ref=r_d,
    )
    for fractional_offset in (-1e-12, -1e-10, 1e-10, 1e-12):
        radius = r_d * (1.0 + fractional_offset)
        local_ratio = (radius - r_d) / r_d
        expected = amplitude * math.expm1((2.0 - gamma) * math.log1p(local_ratio))
        assert potential(radius) == pytest.approx(expected, rel=3e-8)


def test_nfw_finite_escape_without_finite_total_mass():
    rho_s = 0.02
    r_s = 1500.0

    def density(radius):
        x = radius / r_s
        return rho_s / (x * (1.0 + x) ** 2)

    potential = make_initial_potential(density, "finite_escape")
    scale = 4.0 * math.pi * G * rho_s * r_s**2
    for ratio in (0.01, 0.1, 1.0, 10.0, 100.0):
        radius = r_s * ratio
        enclosed_mass = 4.0 * math.pi * rho_s * r_s**3 * (
            math.log1p(ratio) - ratio / (1.0 + ratio)
        )
        expected_phi = -scale * math.log1p(ratio) / ratio
        expected_prime = G * enclosed_mass / radius**2
        expected_second = 4.0 * math.pi * G * density(radius) - 2.0 * expected_prime / radius
        assert potential(radius) == pytest.approx(expected_phi, rel=3e-7)
        assert potential.prime(radius) == pytest.approx(expected_prime, rel=3e-7)
        assert potential.second(radius) == pytest.approx(expected_second, rel=2e-7)

    far = r_s * 1e4
    assert potential(far) < 0.0
    assert abs(potential(far)) < abs(potential(r_s * 100.0))
    mass_at_100 = potential.prime(r_s * 100.0) * (r_s * 100.0) ** 2 / G
    mass_at_10000 = potential.prime(far) * far**2 / G
    assert mass_at_10000 > mass_at_100


def test_hernquist_finite_escape_potential_and_derivatives():
    halo_mass = 1e11
    a = 2000.0

    def density(radius):
        return halo_mass * a / (2.0 * math.pi * radius * (radius + a) ** 3)

    potential = make_initial_potential(density, "finite_escape")
    for ratio in (0.001, 0.1, 1.0, 10.0, 100.0):
        radius = a * ratio
        assert potential(radius) == pytest.approx(-G * halo_mass / (radius + a), rel=3e-7)
        assert potential.prime(radius) == pytest.approx(
            G * halo_mass / (radius + a) ** 2, rel=3e-7
        )
        assert potential.second(radius) == pytest.approx(
            -2.0 * G * halo_mass / (radius + a) ** 3, rel=2e-6
        )
    assert abs(potential(1e5 * a)) < abs(potential(100.0 * a))


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_hernquist_mass_over_widely_separated_scales(scale):
    density = lambda radius: scale / (2.0 * math.pi * radius * (radius + scale) ** 3)
    potential = make_initial_potential(density, "finite_escape", G=1.0)
    for ratio in (0.01, 1.0, 100.0, 1e6, 1e10):
        radius = scale * ratio
        assert potential(radius) == pytest.approx(-1.0 / (radius + scale), rel=2e-8)
        assert potential.prime(radius) == pytest.approx(
            1.0 / (radius + scale) ** 2, rel=2e-8
        )


def test_nfw_second_derivative_resolved_range_and_precision_limit():
    density = lambda radius: 1.0 / (radius * (1.0 + radius) ** 2)
    potential = make_initial_potential(density, "finite_escape", G=1.0)
    for radius in (1e-4, 1e-6, 1e-8):
        # Taylor references avoid cancellation in log1p(r)-r/(1+r).
        expected_phi = -4.0 * math.pi * (
            1.0 - radius / 2.0 + radius * radius / 3.0
        )
        expected_prime = 2.0 * math.pi - 8.0 * math.pi * radius / 3.0
        expected_second = -8.0 * math.pi / 3.0 + 6.0 * math.pi * radius
        expected_second -= 48.0 * math.pi * radius * radius / 5.0
        assert potential(radius) == pytest.approx(expected_phi, rel=2e-8)
        assert potential.prime(radius) == pytest.approx(expected_prime, rel=2e-8)
        assert potential.second(radius) == pytest.approx(expected_second, rel=2e-6)
    for radius in (1e-12, 1e-15, 1e-16):
        with pytest.raises(ArithmeticError, match="second derivative.*resolv|precision"):
            potential.second(radius)


def test_exactly_zero_second_derivative_is_not_misreported_as_resolved():
    potential = make_initial_potential(
        lambda radius: 1.0 / (2.0 * math.pi * radius),
        "confining", G=1.0, r_ref=1.0,
    )
    with pytest.raises(ArithmeticError, match="second derivative.*resolv|precision"):
        potential.second(1.0)


def test_hernquist_numerical_potential_in_eddington_at_ordinary_energy():
    from dm_spikes.eddington import make_eddington_df

    density = lambda radius: 1.0 / (2.0 * math.pi * radius * (1.0 + radius) ** 3)
    density_prime = lambda radius: density(radius) * (
        -1.0 / radius - 3.0 / (1.0 + radius)
    )
    density_second = lambda radius: density(radius) * (
        2.0 / radius**2 + 6.0 / (radius * (1.0 + radius))
        + 12.0 / (1.0 + radius) ** 2
    )
    numerical = make_initial_potential(density, "finite_escape", G=1.0)
    analytic = lambda radius: -1.0 / (1.0 + radius)
    common = dict(
        density_prime=density_prime,
        density_second=density_second,
        central_potential=-1.0,
    )
    df_numerical = make_eddington_df(
        density, numerical, "finite_escape",
        potential_prime=numerical.prime, potential_second=numerical.second,
        **common,
    )
    df_analytic = make_eddington_df(
        density, analytic, "finite_escape",
        potential_prime=lambda radius: 1.0 / (1.0 + radius) ** 2,
        potential_second=lambda radius: -2.0 / (1.0 + radius) ** 3,
        **common,
    )
    assert df_numerical(-0.5) == pytest.approx(df_analytic(-0.5), rel=1e-7)


def test_input_validation_and_nonconvergent_escape_tail():
    density = lambda radius: 1.0 / radius
    with pytest.raises(ValueError, match="boundary"):
        make_initial_potential(density, "other")
    with pytest.raises(ValueError, match="r_ref"):
        make_initial_potential(density, "confining")
    with pytest.raises(ValueError, match="r_ref"):
        make_initial_potential(density, "finite_escape", r_ref=1.0)
    with pytest.raises(ValueError, match="G"):
        make_initial_potential(density, "confining", G=0.0, r_ref=1.0)
    with pytest.raises(ValueError, match="radius"):
        make_initial_potential(density, "confining", r_ref=1.0)(0.0)
    with pytest.raises(ArithmeticError, match="density"):
        make_initial_potential(lambda radius: -1.0, "finite_escape")

    divergent = make_initial_potential(lambda radius: 1.0 / radius / radius, "finite_escape")
    with pytest.raises(ArithmeticError, match="shell|tail|converge"):
        divergent(1.0)


def test_callable_and_derivatives_connect_to_existing_modules():
    from dm_spikes.adiabatic_map import find_turning_points
    from dm_spikes.eddington import make_eddington_df

    amplitude = 1.0 / (2.0 * math.pi)
    density = lambda radius: amplitude / radius
    potential = make_initial_potential(density, "confining", G=1.0, r_ref=1.0)
    distribution = make_eddington_df(
        density, potential, "confining",
        density_prime=lambda radius: -amplitude / radius**2,
        density_second=lambda radius: 2.0 * amplitude / radius**3,
        potential_prime=potential.prime,
        # For this exactly linear potential, phi''=0 cannot be certified
        # from generic float64 density samples at arbitrary radii.
        potential_second=lambda radius: 0.0,
        central_potential=-1.0,
    )
    expected_df = amplitude * math.gamma(2.5) / (2.0 * math.pi) ** 1.5
    assert distribution(0.0) == pytest.approx(expected_df, rel=2e-6)
    assert find_turning_points(0.2, 0.4, potential) == pytest.approx(
        find_turning_points(0.2, 0.4, lambda radius: radius - 1.0),
        rel=2e-7,
    )
