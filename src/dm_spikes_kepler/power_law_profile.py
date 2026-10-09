"""Exact analytical spike generated from an initial power-law profile."""

import numpy as np
from scipy import special


D = 8.5e3  # pc


def _as_float_array(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _return_scalar_if_scalar(value, scalar):
    if scalar:
        return float(np.asarray(value))
    return value


def _gamma_parameters(gamma):
    gamma = _as_float_array(gamma, "gamma")
    if np.any((gamma <= 0.0) | (gamma >= 2.0)):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")

    epsilon = 2.0 - gamma
    four_minus_gamma = 4.0 - gamma
    beta = (6.0 - gamma) / (2.0 * epsilon)
    p = (6.0 - gamma) / four_minus_gamma

    log_b = (
        np.log(np.pi)
        + np.log(epsilon)
        - special.betaln(1.0 / epsilon, 1.5)
    )
    log_lambda = (
        -np.log1p(epsilon / 2.0) / epsilon
        + 0.5 * (np.log(epsilon) - np.log(four_minus_gamma))
    )
    log_b_lambda = log_b + log_lambda
    kappa_factor = -np.expm1(-log_b_lambda)
    if np.any(kappa_factor <= 0.0):
        raise FloatingPointError(
            "kappa_gamma is not positive for the supplied gamma values."
        )
    log_kappa = np.log(2.0) + np.log(kappa_factor)

    return beta, p, log_b, log_kappa


def _j_gamma_series(
    eta,
    gamma,
    *,
    rtol=1e-12,
    max_terms=1000,
    include_tail=True,
):
    if rtol <= 0.0:
        raise ValueError("rtol must be positive.")
    if max_terms < 1:
        raise ValueError("max_terms must be at least 1.")

    eta = _as_float_array(eta, "eta")
    gamma = _as_float_array(gamma, "gamma")
    _, p, log_b, log_kappa = _gamma_parameters(gamma)
    eta, p, log_b, log_kappa = np.broadcast_arrays(
        eta, p, log_b, log_kappa
    )

    if include_tail:
        below = eta < 0.0
        above = eta > 1.0
        if np.any(below & (np.abs(eta) > rtol)) or np.any(
            above & ((eta - 1.0) > rtol)
        ):
            raise ValueError("eta must satisfy 0 <= eta <= 1, up to roundoff.")
        eta = np.clip(eta, 0.0, 1.0)

    total = np.zeros_like(p, dtype=float)
    tiny = np.finfo(float).tiny

    for n in range(max_terms):
        n_float = float(n)
        a_tail = n_float / 2.0 + 1.0
        c_tail = (p + n_float + 3.0) / 2.0
        log_term = (
            -p * log_b
            - np.log(2.0)
            + special.gammaln(p + n_float)
            - special.gammaln(p)
            - special.gammaln(n_float + 1.0)
            + n_float * log_kappa
            + special.betaln((p + n_float) / 2.0 + 1.0, 0.5)
            + special.betaln(a_tail, c_tail)
        )
        term = np.exp(log_term)
        if include_tail:
            term *= special.betaincc(a_tail, c_tail, eta)
        total += term

        scale = np.maximum(np.abs(total), tiny)
        if np.all(np.abs(term) <= rtol * scale):
            return total

    raise RuntimeError(
        "The analytic beta-function series did not converge before max_terms."
    )


def J_gamma(eta, gamma, *, rtol=1e-12, max_terms=1000):
    """Evaluate the exact beta-function series J_gamma(eta).

    eta and gamma may be scalars or broadcast-compatible arrays. The physical
    domain is 0 <= eta <= 1 and 0 < gamma < 2.
    """
    scalar = (
        (np.isscalar(eta) or np.asarray(eta).shape == ())
        and (np.isscalar(gamma) or np.asarray(gamma).shape == ())
    )
    result = _j_gamma_series(
        eta,
        gamma,
        rtol=rtol,
        max_terms=max_terms,
        include_tail=True,
    )
    return _return_scalar_if_scalar(result, scalar)


def alpha_gamma(gamma, *, rtol=1e-12, max_terms=1000):
    """Return the exact dimensionless spike coefficient alpha_gamma."""
    scalar = np.isscalar(gamma) or np.asarray(gamma).shape == ()
    gamma = _as_float_array(gamma, "gamma")
    beta, p, _, _ = _gamma_parameters(gamma)
    J0 = _j_gamma_series(
        np.zeros_like(gamma, dtype=float),
        gamma,
        rtol=rtol,
        max_terms=max_terms,
        include_tail=False,
    )
    if np.any(J0 <= 0.0):
        raise FloatingPointError("J_gamma(0) must be positive.")

    a_gamma = (3.0 - gamma) / (4.0 - gamma)
    s_gamma = (3.0 - gamma) ** 2 / (4.0 - gamma)
    log_alpha = (
        np.log(J0)
        + (p + 1.0) * np.log(2.0)
        - 0.5 * np.log(np.pi)
        + special.gammaln(beta)
        - special.gammaln(beta - 1.5)
        + a_gamma
        * np.log((3.0 - gamma) * (2.0 - gamma) / (4.0 * np.pi))
    ) / s_gamma

    return _return_scalar_if_scalar(np.exp(log_alpha), scalar)


def g_gamma(eta, gamma, *, rtol=1e-12, max_terms=1000):
    """Return the exact capture factor J_gamma(eta) / J_gamma(0)."""
    scalar = (
        (np.isscalar(eta) or np.asarray(eta).shape == ())
        and (np.isscalar(gamma) or np.asarray(gamma).shape == ())
    )
    eta = _as_float_array(eta, "eta")
    gamma = _as_float_array(gamma, "gamma")
    eta, gamma = np.broadcast_arrays(eta, gamma)

    J_eta = _j_gamma_series(
        eta,
        gamma,
        rtol=rtol,
        max_terms=max_terms,
        include_tail=True,
    )
    J0 = _j_gamma_series(
        np.zeros_like(eta, dtype=float),
        gamma,
        rtol=rtol,
        max_terms=max_terms,
        include_tail=False,
    )
    result = J_eta / J0

    below = result < 0.0
    above = result > 1.0
    if np.any(below & (np.abs(result) > rtol)) or np.any(
        above & ((result - 1.0) > rtol)
    ):
        raise FloatingPointError(
            "g_gamma fell outside [0, 1] beyond roundoff."
        )
    result = np.clip(result, 0.0, 1.0)

    return _return_scalar_if_scalar(result, scalar)


def spike_radius(M, gamma, rho0, r0, *, rtol=1e-12, max_terms=1000):
    """Return the exact spike radius for an initial power-law cusp."""
    scalar = all(
        np.asarray(value).shape == () for value in (M, gamma, rho0, r0)
    )
    M = _as_float_array(M, "M")
    gamma = _as_float_array(gamma, "gamma")
    rho0 = _as_float_array(rho0, "rho0")
    r0 = _as_float_array(r0, "r0")
    if np.any(M <= 0.0):
        raise ValueError("M must be positive.")
    if np.any(rho0 <= 0.0):
        raise ValueError("rho0 must be positive.")
    if np.any(r0 <= 0.0):
        raise ValueError("r0 must be positive.")

    alfa = alpha_gamma(gamma, rtol=rtol, max_terms=max_terms)
    result = alfa * r0 * np.power(
        M / (rho0 * np.power(r0, 3.0)), 1.0 / (3.0 - gamma)
    )
    return _return_scalar_if_scalar(result, scalar)


def rho_R(gamma, rho0, r0, R_sp):
    """Return the initial-cusp density evaluated at the spike radius."""
    scalar = all(
        np.asarray(value).shape == () for value in (gamma, rho0, r0, R_sp)
    )
    gamma = _as_float_array(gamma, "gamma")
    rho0 = _as_float_array(rho0, "rho0")
    r0 = _as_float_array(r0, "r0")
    R_sp = _as_float_array(R_sp, "R_sp")
    if np.any((gamma <= 0.0) | (gamma >= 2.0)):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    if np.any(rho0 <= 0.0):
        raise ValueError("rho0 must be positive.")
    if np.any(r0 <= 0.0):
        raise ValueError("r0 must be positive.")
    if np.any(R_sp <= 0.0):
        raise ValueError("R_sp must be positive.")

    result = rho0 * np.power(R_sp / r0, -gamma)
    return _return_scalar_if_scalar(result, scalar)


def gamma_spike(gamma):
    """Return the final Gondolo-Silk spike slope."""
    scalar = np.isscalar(gamma) or np.asarray(gamma).shape == ()
    gamma = _as_float_array(gamma, "gamma")
    if np.any((gamma <= 0.0) | (gamma >= 2.0)):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    result = (9.0 - 2.0 * gamma) / (4.0 - gamma)
    return _return_scalar_if_scalar(result, scalar)


def rho_D(gamma):
    """Return the initial-cusp density normalization at the solar radius."""
    scalar = np.isscalar(gamma) or np.asarray(gamma).shape == ()
    gamma = _as_float_array(gamma, "gamma")
    if np.any((gamma <= 0.0) | (gamma >= 2.0)):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    result = 0.0062 * (1.0 - gamma / 3.0)
    return _return_scalar_if_scalar(result, scalar)


def core_radius(M, m, gamma, observable_sigma_v, bh_age=1e10):
    """Power-law radius where the unsaturated spike equals rho_sat.

    Uses the project's default cusp normalization at r0=D and the
    unmodified power-law spike slope; the capture factor is not part of this
    conventional radius estimate. Saturation inputs must have compatible
    units so that m/(observable_sigma_v*bh_age) is in Msun pc^-3.
    """
    scalar = all(
        np.asarray(value).shape == ()
        for value in (M, m, gamma, observable_sigma_v, bh_age)
    )
    mass = _as_float_array(m, "m")
    cross_section = _as_float_array(observable_sigma_v, "observable_sigma_v")
    age = _as_float_array(bh_age, "bh_age")
    if np.any(mass <= 0.0):
        raise ValueError("m must be positive.")
    if np.any(cross_section <= 0.0):
        raise ValueError("observable_sigma_v must be positive.")
    if np.any(age <= 0.0):
        raise ValueError("bh_age must be positive.")
    saturation = mass / (cross_section * age)
    if np.any(~np.isfinite(saturation) | (saturation <= 0.0)):
        raise ArithmeticError("saturation density is outside the float64 range")

    density_at_d = rho_D(gamma)
    R_sp = spike_radius(M, gamma, density_at_d, D)
    density_at_spike = rho_R(gamma, density_at_d, D, R_sp)
    result = R_sp * np.power(
        density_at_spike / saturation, 1.0 / gamma_spike(gamma)
    )
    return _return_scalar_if_scalar(result, scalar)


def cusp_profile(
    r_values,
    M,
    gamma,
    R_S,
    *,
    r0=D,
    rho0=None,
    rtol=1e-12,
    max_terms=1000,
):
    """Return the exact analytical cusp profile in Msun pc^-3.

    By default the project normalization rho_D(gamma) at r0 = D is used. The
    profile vanishes at and inside the capture boundary.
    """
    scalar = (
        (np.isscalar(r_values) or np.asarray(r_values).shape == ())
        and (np.isscalar(gamma) or np.asarray(gamma).shape == ())
    )
    r_values = _as_float_array(r_values, "r_values")
    if np.any(r_values <= 0.0):
        raise ValueError("r_values must be positive.")
    if M <= 0.0:
        raise ValueError("M must be positive.")
    if R_S <= 0.0:
        raise ValueError("R_S must be positive.")
    if r0 <= 0.0:
        raise ValueError("r0 must be positive.")

    density_scale = rho_D(gamma) if rho0 is None else rho0
    R_sp = spike_radius(
        M,
        gamma,
        density_scale,
        r0,
        rtol=rtol,
        max_terms=max_terms,
    )
    slope = gamma_spike(gamma)
    density_at_spike = rho_R(gamma, density_scale, r0, R_sp)
    eta = np.minimum(4.0 * R_S / r_values, 1.0)
    capture_factor = g_gamma(
        eta,
        gamma,
        rtol=rtol,
        max_terms=max_terms,
    )
    result = (
        density_at_spike
        * np.power(R_sp / r_values, slope)
        * capture_factor
    )
    return _return_scalar_if_scalar(result, scalar)


def initial_cusp_profile(r_values, gamma):
    """Return the initial power-law cusp profile in Msun pc^-3."""
    scalar = (
        (np.isscalar(r_values) or np.asarray(r_values).shape == ())
        and (np.isscalar(gamma) or np.asarray(gamma).shape == ())
    )
    r_values = _as_float_array(r_values, "r_values")
    if np.any(r_values <= 0.0):
        raise ValueError("r_values must be positive.")

    result = rho_D(gamma) * np.power(r_values / D, -np.asarray(gamma))
    return _return_scalar_if_scalar(result, scalar)
