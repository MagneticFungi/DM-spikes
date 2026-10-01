"""Analytic and density-reconstruction checks of direct Eddington inversion."""

import inspect
import math

import pytest
from scipy.integrate import quad
from scipy.special import gamma as gamma_function

from dm_spikes import make_eddington_df, solve_initial_energy


POWER_LAW_AMPLITUDE = 1.0 / (2.0 * math.pi)


def power_law_density(radius):
    return POWER_LAW_AMPLITUDE / radius


def power_law_potential(radius):
    # Gamma=1, G=1, rho0=1/(2 pi), r0=1: Gondolo--Silk phi0=1.
    return radius


def hernquist_density(radius):
    return 1.0 / (2.0 * math.pi * radius * (1.0 + radius) ** 3)


def hernquist_potential(radius):
    return -1.0 / (1.0 + radius)


def nfw_density(radius):
    return 1.0 / (4.0 * math.pi * radius * (1.0 + radius) ** 2)


def nfw_potential(radius):
    return -math.log1p(radius) / radius


def reconstruct_density(distribution, potential, radius, boundary):
    phi = potential(radius)
    if boundary == "finite_escape":
        v_escape = math.sqrt(-2.0 * phi)
        integral, _ = quad(
            lambda u: distribution(phi * (1.0 - u * u)) * u * u,
            0.0, 1.0, epsrel=2e-4,
        )
        return 4.0 * math.pi * v_escape**3 * integral
    integral, _ = quad(
        lambda speed: distribution(phi + 0.5 * speed * speed) * speed * speed,
        0.0, math.inf, epsrel=2e-4,
    )
    return 4.0 * math.pi * integral


def test_public_interface_has_three_required_arguments():
    signature = inspect.signature(make_eddington_df)
    required = [
        name for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    assert required == ["density", "potential", "boundary"]


@pytest.mark.parametrize("gamma", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("energy", [0.5, 1.0, 4.0])
def test_power_law_matches_gondolo_silk_distribution(gamma, energy):
    phi0 = 4.0 * math.pi * POWER_LAW_AMPLITUDE / ((3.0 - gamma) * (2.0 - gamma))
    distribution = make_eddington_df(
        lambda radius: POWER_LAW_AMPLITUDE * radius ** (-gamma),
        lambda radius: phi0 * radius ** (2.0 - gamma),
        "confining",
    )
    beta = (6.0 - gamma) / (2.0 * (2.0 - gamma))
    expected = (
        POWER_LAW_AMPLITUDE
        * gamma_function(beta)
        / ((2.0 * math.pi * phi0) ** 1.5 * gamma_function(beta - 1.5))
        * (phi0 / energy) ** beta
    )
    assert distribution(energy) == pytest.approx(expected, rel=3e-5)


def test_confining_energy_uses_potential_zero_without_conversion():
    shifted = make_eddington_df(
        power_law_density, lambda radius: radius + 7.0, "confining",
        central_potential=7.0,
    )
    unshifted = make_eddington_df(
        power_law_density, power_law_potential, "confining"
    )
    assert shifted(8.0) == pytest.approx(unshifted(1.0), rel=3e-5)
    assert shifted(7.0) == 0.0


@pytest.mark.parametrize(
    "density,potential,boundary,radii,tolerance",
    [
        (power_law_density, power_law_potential, "confining", (0.5, 2.0), 1e-4),
        (hernquist_density, hernquist_potential, "finite_escape", (0.5, 2.0), 3e-4),
        (nfw_density, nfw_potential, "finite_escape", (0.5, 2.0), 5e-4),
    ],
)
def test_density_is_reconstructed_from_velocity_integral(
    density, potential, boundary, radii, tolerance
):
    distribution = make_eddington_df(density, potential, boundary)
    for radius in radii:
        reconstructed = reconstruct_density(distribution, potential, radius, boundary)
        assert reconstructed == pytest.approx(density(radius), rel=tolerance)


def test_domains_boundary_name_and_surface_term():
    with pytest.raises(ValueError, match="boundary"):
        make_eddington_df(hernquist_density, hernquist_potential, "other")
    with pytest.raises(ValueError, match="tail_survey_log_radius"):
        make_eddington_df(
            power_law_density, power_law_potential, "confining",
            tail_survey_log_radius=0.0,
        )
    finite = make_eddington_df(
        hernquist_density, hernquist_potential, "finite_escape",
        central_potential=-1.0,
    )
    confining = make_eddington_df(
        power_law_density, power_law_potential, "confining",
        central_potential=0.0,
    )
    assert finite(-1.1) == 0.0
    assert finite(-1.0) == 0.0
    assert finite(0.0) == 0.0
    assert finite(0.1) == 0.0
    assert confining(-0.1) == 0.0
    assert confining(0.0) == 0.0

    # rho=-phi and d rho/d phi at escape = -1: the surface term alone remains.
    surface = make_eddington_df(
        lambda radius: 1.0 / radius,
        lambda radius: -1.0 / radius,
        "finite_escape",
        density_prime=lambda radius: -1.0 / radius**2,
        density_second=lambda radius: 2.0 / radius**3,
        potential_prime=lambda radius: 1.0 / radius**2,
        potential_second=lambda radius: -2.0 / radius**3,
        escape_slope=-1.0,
    )
    assert surface(-0.5) == pytest.approx(1.0 / (math.sqrt(8.0) * math.pi**2 * math.sqrt(0.5)))


def test_initial_energy_is_accepted_without_sign_change():
    initial_energy = solve_initial_energy(
        -0.5, 0.4, 1.0, hernquist_potential, G=1.0
    )
    distribution = make_eddington_df(
        hernquist_density, hernquist_potential, "finite_escape"
    )
    assert -1.0 < initial_energy < 0.0
    assert distribution(initial_energy) > 0.0


def test_unreliable_numerical_derivatives_fail_clearly():
    # The potential is quantized despite a nominally smooth shape.
    distribution = make_eddington_df(
        power_law_density,
        lambda radius: round(radius, 3),
        "confining",
    )
    with pytest.raises(ArithmeticError, match="derivative|potential"):
        distribution(1.0)


def test_unbracketed_central_energy_is_not_silently_zero():
    unresolved = make_eddington_df(
        power_law_density, lambda radius: 7.0 + radius, "confining"
    )
    with pytest.raises(ArithmeticError, match="central|inner root"):
        unresolved(7.0)
    with pytest.raises(ArithmeticError, match="central|inner root"):
        unresolved(6.0)
    with pytest.raises(ArithmeticError, match="unresolved|central"):
        unresolved(7.0 + 1e-14)
    known_center = make_eddington_df(
        power_law_density, lambda radius: 7.0 + radius, "confining",
        central_potential=7.0,
    )
    assert known_center(7.0) == 0.0
    assert known_center(6.0) == 0.0
    with pytest.raises(ArithmeticError, match="unresolved|root"):
        known_center(7.0 + 1e-14)


@pytest.mark.parametrize("distance_from_center", [1e-6, 1e-7])
def test_near_central_hernquist_with_analytic_derivatives(distance_from_center):
    def density_prime(radius):
        return hernquist_density(radius) * (-1.0 / radius - 3.0 / (1.0 + radius))

    def density_second(radius):
        slope = -1.0 / radius - 3.0 / (1.0 + radius)
        return hernquist_density(radius) * (
            slope * slope + 1.0 / radius**2 + 3.0 / (1.0 + radius) ** 2
        )

    distribution = make_eddington_df(
        hernquist_density, hernquist_potential, "finite_escape",
        density_prime=density_prime, density_second=density_second,
        potential_prime=lambda radius: 1.0 / (1.0 + radius) ** 2,
        potential_second=lambda radius: -2.0 / (1.0 + radius) ** 3,
        central_potential=-1.0,
    )
    q = 1.0 - distance_from_center

    def reference_integrand(u):
        y = q - u * u
        denominator = distance_from_center + u * u
        return 2.0 / (2.0 * math.pi) * (
            12.0 * y**2 / denominator
            + 8.0 * y**3 / denominator**2
            + 2.0 * y**4 / denominator**3
        )

    split = math.sqrt(distance_from_center)
    reference = sum(
        quad(reference_integrand, left, right, epsrel=1e-10, limit=200)[0]
        for left, right in ((0.0, split), (split, math.sqrt(q)))
    ) / (math.sqrt(8.0) * math.pi**2)
    assert distribution(-q) == pytest.approx(reference, rel=2e-6)


def test_multiscale_tail_does_not_stop_before_remote_contribution():
    # rho'' is exactly zero for the added component until r=R. An early
    # small-segment test alone would miss its remote contribution to f(1).
    remote_radius = math.exp(40.0)
    amplitude = 30.0 * remote_radius**1.5

    def density(radius):
        t = radius / remote_radius - 1.0
        if t < 0.0:
            bump = amplitude * (1.0 / 60.0 - t / 30.0)
        elif t < 1.0:
            moment0 = 1.0 / 30.0 - t**3 / 3.0 + t**4 / 2.0 - t**5 / 5.0
            moment1 = 1.0 / 60.0 - t**4 / 4.0 + 2.0 * t**5 / 5.0 - t**6 / 6.0
            bump = amplitude * (moment1 - t * moment0)
        else:
            bump = 0.0
        return 1.0 / radius + bump

    def density_prime(radius):
        t = radius / remote_radius - 1.0
        if t < 0.0:
            bump_slope = -amplitude / (30.0 * remote_radius)
        elif t < 1.0:
            moment0 = 1.0 / 30.0 - t**3 / 3.0 + t**4 / 2.0 - t**5 / 5.0
            bump_slope = -amplitude / remote_radius * moment0
        else:
            bump_slope = 0.0
        return -1.0 / radius / radius + bump_slope

    def density_second(radius):
        t = radius / remote_radius - 1.0
        bump_curvature = (
            amplitude / remote_radius**2 * t**2 * (1.0 - t) ** 2
            if 0.0 < t < 1.0 else 0.0
        )
        return 2.0 / radius / radius / radius + bump_curvature

    distribution = make_eddington_df(
        density, lambda radius: radius, "confining",
        density_prime=density_prime, density_second=density_second,
        potential_prime=lambda radius: 1.0,
        potential_second=lambda radius: 0.0,
        central_potential=0.0,
    )
    remote_integral = 30.0 * quad(
        lambda t: t**2 * (1.0 - t) ** 2
        / math.sqrt(1.0 + t - 1.0 / remote_radius),
        0.0, 1.0, epsrel=1e-11,
    )[0]
    expected = (3.0 * math.pi / 4.0 + remote_integral) / (
        math.sqrt(8.0) * math.pi**2
    )
    assert distribution(1.0) == pytest.approx(expected, rel=2e-6)
