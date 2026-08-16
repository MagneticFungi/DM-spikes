import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from scipy.special import beta as beta_function
from scipy.special import gammaln

G = 4.30091e-3  # pc (km/s)^2 / Msun
C_LIGHT = 299792.458  # km/s


def schwarzschild_radius(M, G=G, clight=C_LIGHT):
    return 2.0 * G * M / clight**2


def energy_lower_bound(r, M, G=G, clight=C_LIGHT):
    return -G * M / r * (1.0 - 4.0 * schwarzschild_radius(M, G=G, clight=clight) / r)


def angular_momentum_capture(M, G=G, clight=C_LIGHT):
    return 2.0 * clight * schwarzschild_radius(M, G=G, clight=clight)


def angular_momentum_max(Ep, r, M, G=G):
    return np.sqrt(np.maximum(2.0 * r**2 * (Ep + G * M / r), 0.0))


def radial_velocity(Ep, Lp, r, M, G=G):
    return np.sqrt(np.maximum(2.0 * (Ep + G * M / r - Lp**2 / (2.0 * r**2)), 0.0))


def rho_prime_at_r(f_prime, r, M, G=G, clight=C_LIGHT, epsabs=0.0, epsrel=1e-6, limit=100):
    if r <= 0.0:
        raise ValueError("r must be positive.")

    R_S = schwarzschild_radius(M, G=G, clight=clight)
    if r <= 4.0 * R_S:
        return 0.0

    E_m = energy_lower_bound(r, M, G=G, clight=clight)
    L_c = angular_momentum_capture(M, G=G, clight=clight)
    L_c2 = L_c**2

    def inner(Ep):
        L_m = angular_momentum_max(Ep, r, M, G=G)
        L_m2 = L_m**2
        if L_m2 <= L_c2:
            return 0.0

        u_max = np.sqrt(L_m2 - L_c2)

        def u_integrand(u):
            return f_prime(Ep, np.sqrt(np.maximum(L_m2 - u**2, 0.0)))

        value, _ = quad(u_integrand, 0.0, u_max, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return 4.0 * np.pi / r * value

    rho, _ = quad(inner, E_m, 0.0, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return rho


def rho_prime_profile(f_prime, r_array, M, **kwargs):
    return np.array([rho_prime_at_r(f_prime, r, M, **kwargs) for r in r_array])


def make_isothermal_distribution(rho0, sigma_v):
    f0 = rho0 * (2 * np.pi * sigma_v**2) ** (-1.5)

    def f_prime(Ep, Lp):
        return f0

    return f_prime


def compute_isothermal_profile(r_values, M, rho0, sigma_v, **integration_kwargs):
    f_prime = make_isothermal_distribution(rho0, sigma_v)
    return rho_prime_profile(f_prime, r_values, M, **integration_kwargs)


def gs_phi0(gamma, rho0, r0, G=G):
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    if rho0 <= 0.0:
        raise ValueError("rho0 must be positive.")
    if r0 <= 0.0:
        raise ValueError("r0 must be positive.")
    return 4.0 * np.pi * G * r0**2 * rho0 / ((3.0 - gamma) * (2.0 - gamma))


def gs_beta_exponent(gamma):
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return (6.0 - gamma) / (2.0 * (2.0 - gamma))


def gs_action_exponent(gamma):
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return (4.0 - gamma) / (2.0 * (2.0 - gamma))


def gs_lambda(gamma):
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return (
        (2.0 / (4.0 - gamma)) ** (1.0 / (2.0 - gamma))
        * ((2.0 - gamma) / (4.0 - gamma)) ** 0.5
    )


def gs_b(gamma):
    if not (0.0 < gamma < 2.0):
        raise ValueError("gamma must satisfy 0 < gamma < 2.")
    return np.pi * (2.0 - gamma) / beta_function(1.0 / (2.0 - gamma), 1.5)


def initial_cusp_distribution(E, L, gamma, rho0, r0, G=G):
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


def compute_cusp_density_worker(args):
    """Compute one cusp-density point from serializable worker arguments."""
    gamma, r, M, D_pc, epsrel, limit = args
    return compute_cusp_profile(
        gamma=gamma,
        r_values_adiabatic=(r,),
        M=M,
        D_pc=D_pc,
        epsrel=epsrel,
        limit=limit,
    )[0]


# Cubic interpolation of the tabulated Gondolo-Silk alpha_gamma values.
alfa_spline = CubicSpline(np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 2.0]), np.array([0.0733, 0.12, 0.140, 0.142, 0.135, 0.122, 0.103, 0.0818, 0.0177]))


def alpha_gamma(gamma):
    """Interpolated alpha_gamma value for a given gamma."""

    if not (0.0 < gamma < 2.0):
        raise ValueError("Se requiere 0 < gamma < 2.")

    if gamma < 0.05:
        return 0.293 * np.power(gamma, 4.0/9.0)
    else:
        return float(alfa_spline(gamma))


def spike_radius(M, gamma, rho0, r0):
    """Spike radius for a given gamma, M, rho0, and r0."""
    alfa = alpha_gamma(gamma)
    return alfa * r0 * np.power(M / (rho0 * r0**3), 1.0 / (3.0 - gamma))


def positive_worker_count(value):
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return workers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Gondolo-Silk density profiles."
    )
    parser.add_argument(
        "--workers",
        type=positive_worker_count,
        default=32,
        help="number of worker processes (default: 32)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Inputs
    M = 2.6e6
    D = 8.5e3
    #rho0_iso = 1.0
    rho0_iso = 0.0062
    sigma_v = 100.0

    R_S = schwarzschild_radius(M)
    #gamma_values = np.array([0.5,1.0,1.5])
    #gamma_values = np.array([1.0, 1.95])
    gamma_values = np.arange(0.01, 1.96, 0.01)
    #gamma_values = np.arange(0.01, 2.0, 0.01)
    rho_D_values = 0.0062 * (1.0 - gamma_values / 3.0)

    total_start = time.perf_counter()
    print(f"Using {args.workers} worker processes\n", flush=True)

    # Spike radii
    R_spike_values = np.array(
        [
            spike_radius(M, gamma_i, rho_D_i, D)
            for gamma_i, rho_D_i in zip(gamma_values, rho_D_values)
        ]
    )

    # Isothermal profile
    r_iso = np.logspace(np.log10(4.001 * R_S), np.log10(0.2), 300)
    rho_iso = compute_isothermal_profile(r_iso, M, rho0_iso, sigma_v)

    x_iso_values = np.log10(r_iso / R_S)
    y_iso_values = np.log10(rho_iso)

    # Cusp profiles. Each gamma has its own radial domain ending at R_sp(gamma).
    r_cusp_by_gamma = {}
    rho_cusp_by_gamma = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for gamma_i, R_sp_i in zip(gamma_values, R_spike_values):
            gamma_start = time.perf_counter()
            gamma_i = float(gamma_i)

            r_cusp_i = np.logspace(
                np.log10(4.001 * R_S), np.log10(R_sp_i), 300
            )
            worker_args = (
                (gamma_i, float(r), M, D, 1e-5, 100) for r in r_cusp_i
            )

            r_cusp_by_gamma[gamma_i] = r_cusp_i
            rho_cusp_by_gamma[gamma_i] = np.array(
                list(
                    executor.map(
                        compute_cusp_density_worker,
                        worker_args,
                        chunksize=1,
                    )
                )
            )

            gamma_elapsed = time.perf_counter() - gamma_start
            print(
                f"gamma = {gamma_i:.2f} completed in {gamma_elapsed:.3f} s",
                flush=True,
            )

    r_cusp_grid = np.array(
        [r_cusp_by_gamma[float(gamma_i)] for gamma_i in gamma_values]
    )
    rho_cusp_grid = np.array(
        [rho_cusp_by_gamma[float(gamma_i)] for gamma_i in gamma_values]
    )

    x_cusp_grid = np.log10(r_cusp_grid / R_S)
    y_cusp_grid = np.log10(rho_cusp_grid)

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal computation time: {total_elapsed:.3f} s\n", flush=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"perfiles_densidad_gs_gammas_{timestamp}.npz"

    np.savez(
        output_file,
        r_iso=r_iso,
        rho_iso=rho_iso,
        gamma_values=gamma_values,
        r_cusp=r_cusp_grid,
        rho_cusp_grid=rho_cusp_grid,
    )

    print(f"Perfiles guardados en: {output_file}")


if __name__ == "__main__":
    main()
