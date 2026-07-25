"""Gondolo-Silk cuspidal model and adiabatic radial-action mapping."""

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import beta as beta_function
from scipy.special import gammaln

from .density import G, rho_prime_profile


def gs_phi0(gamma, rho0, r0, G=G):
    """Return the initial cusp potential scale phi0 in (km/s)^2."""
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    if rho0 <= 0.0:
        raise ValueError("rho0 must be positive.")
    if r0 <= 0.0:
        raise ValueError("r0 must be positive.")
    return 4.0 * np.pi * G * r0**2 * rho0 / ((3.0 - gamma) * (2.0 - gamma))


def gs_beta_exponent(gamma):
    """Return the beta exponent for the GS cusp distribution."""
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return (6.0 - gamma) / (2.0 * (2.0 - gamma))


def gs_action_exponent(gamma):
    """Return the radial-action exponent for the GS cusp model."""
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return (4.0 - gamma) / (2.0 * (2.0 - gamma))


def gs_lambda(gamma):
    """Return the dimensionless lambda coefficient for the GS cusp model."""
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return (
        (2.0 / (4.0 - gamma)) ** (1.0 / (2.0 - gamma))
        * ((2.0 - gamma) / (4.0 - gamma)) ** 0.5
    )


def gs_b(gamma):
    """Return the dimensionless b coefficient for the GS cusp model."""
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return np.pi * (2.0 - gamma) / beta_function(1.0 / (2.0 - gamma), 1.5)


def initial_cusp_distribution(E, L, gamma, rho0, r0, G=G):
    """Return the initial cusp distribution in Msun pc^-3 (km/s)^-3."""
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    if E <= 0.0:
        raise ValueError("Initial GS energy E must be positive.")
    if rho0 <= 0.0:
        raise ValueError("rho0 must be positive.")
    if r0 <= 0.0:
        raise ValueError("r0 must be positive.")

    phi0 = gs_phi0(gamma, rho0, r0, G=G)
    beta = gs_beta_exponent(gamma)
    if phi0 <= 0.0 or beta <= 1.5:
        raise ValueError("Invalid GS cusp parameters.")

    log_f = (
        np.log(rho0)
        - 1.5 * np.log(2.0 * np.pi * phi0)
        + gammaln(beta)
        - gammaln(beta - 1.5)
        + beta * (np.log(phi0) - np.log(E))
    )
    return np.exp(log_f)


def initial_cusp_radial_action(E, L, gamma, rho0, r0, G=G):
    """Return initial radial action in pc km/s for the GS cusp model."""
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    if E <= 0.0:
        raise ValueError("Initial GS energy E must be positive.")
    if L < 0.0:
        raise ValueError("L must be non-negative.")

    phi0 = gs_phi0(gamma, rho0, r0, G=G)
    p = gs_action_exponent(gamma)
    lam = gs_lambda(gamma)
    b = gs_b(gamma)

    return (2.0 * np.pi / b) * (
        -L / lam + np.sqrt(2.0 * r0**2 * phi0) * (E / phi0) ** p
    )


def radial_action_final_kepler(Ep, Lp, M, G=G):
    """Return final Keplerian radial action in pc km/s for E' < 0."""
    if Ep >= 0.0:
        return np.nan
    return 2.0 * np.pi * (-Lp + G * M / np.sqrt(-2.0 * Ep))


def solve_initial_energy_from_action(
    Ep,
    Lp,
    M,
    radial_action_initial,
    E_bracket,
    action_args=(),
    final_action_func=radial_action_final_kepler,
    G=G,
    xtol=1e-10,
    rtol=1e-10,
    maxiter=200,
):
    """Solve E from radial-action conservation using root_scalar in log(E)."""
    I_target = final_action_func(Ep, Lp, M, G=G)

    if callable(E_bracket):
        E_min, E_max = E_bracket(Ep, Lp, I_target)
    else:
        E_min, E_max = E_bracket

    if E_min <= 0.0 or E_max <= 0.0 or E_max <= E_min:
        raise ValueError(f"Invalid positive energy bracket: {(E_min, E_max)}.")

    def action_residual(E):
        return radial_action_initial(E, Lp, *action_args) - I_target

    f_min = action_residual(E_min)
    f_max = action_residual(E_max)
    if not np.isfinite(f_min) or not np.isfinite(f_max) or f_min * f_max > 0.0:
        raise ValueError(
            "Initial-energy bracket does not change sign for action conservation. "
            f"Ep={Ep}, Lp={Lp}, I_target={I_target}, E_min={E_min}, E_max={E_max}, "
            f"residual(E_min)={f_min}, residual(E_max)={f_max}."
        )

    log_E_min = np.log(E_min)
    log_E_max = np.log(E_max)

    def log_action_residual(log_E):
        return action_residual(np.exp(log_E))

    solution = root_scalar(
        log_action_residual,
        bracket=(log_E_min, log_E_max),
        xtol=xtol,
        rtol=rtol,
        maxiter=maxiter,
    )
    if not solution.converged:
        raise RuntimeError(
            "root_scalar did not converge while solving action conservation in log(E). "
            f"Ep={Ep}, Lp={Lp}, E_bracket={(E_min, E_max)}, "
            f"log_E_bracket={(log_E_min, log_E_max)}."
        )
    return np.exp(solution.root)


def make_f_prime_gs_cusp(
    gamma,
    rho0,
    r0,
    M,
    E_bracket,
    G=G,
    final_action_func=radial_action_final_kepler,
    return_zero_on_fail=False,
):
    """Build f'(E', L') for the adiabatically mapped GS cusp."""
    gs_phi0(gamma, rho0, r0, G=G)
    action_args = (gamma, rho0, r0)

    def f_prime_adiabatic(Ep, Lp):
        if Ep >= 0.0:
            return 0.0

        try:
            E_initial = solve_initial_energy_from_action(
                Ep=Ep,
                Lp=Lp,
                M=M,
                radial_action_initial=initial_cusp_radial_action,
                E_bracket=E_bracket,
                action_args=action_args,
                final_action_func=final_action_func,
                G=G,
            )
            return initial_cusp_distribution(E_initial, Lp, gamma, rho0, r0, G=G)
        except Exception:
            if return_zero_on_fail:
                return 0.0
            raise

    return f_prime_adiabatic


def compute_cusp_profile(
    gamma,
    r_values_adiabatic,
    M,
    D_pc=8.5e3,
    epsrel=1e-5,
    limit=100,
):
    """Return final GS cusp rho'(r) in Msun pc^-3 for radii in pc."""
    rho_D = 0.0062 * (1.0 - gamma / 3.0)
    r0 = D_pc
    rho0 = rho_D
    phi0 = gs_phi0(gamma, rho0, r0, G=G)

    def E_bracket_gs_example(Ep, Lp, I_target):
        return (1e-12 * phi0, 1e12 * phi0)

    f_prime_adiabatic = make_f_prime_gs_cusp(
        gamma=gamma,
        rho0=rho0,
        r0=r0,
        M=M,
        E_bracket=E_bracket_gs_example,
        final_action_func=radial_action_final_kepler,
        return_zero_on_fail=False,
    )

    rho_values = rho_prime_profile(
        f_prime_adiabatic,
        r_values_adiabatic,
        M,
        epsrel=epsrel,
        limit=limit,
    )

    return rho_values
