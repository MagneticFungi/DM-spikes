"""Fit a global posterior model for the numerical capture factor g_gamma(r)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval, chebvander
from scipy.optimize import least_squares

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import dm_spikes as dm


def _as_rows(values):
    values = np.asarray(values, dtype=object if np.asarray(values).dtype == object else float)
    if values.dtype == object:
        return [np.asarray(row, dtype=float) for row in values]
    if values.ndim == 1:
        return [values.astype(float)]
    return [np.asarray(row, dtype=float) for row in values]


def load_capture_data(path, M=2.6e6):
    data = np.load(path, allow_pickle=True)
    gamma_values = np.asarray(data["gamma_values"], dtype=float)
    r_key = "r_cusp_grid" if "r_cusp_grid" in data.files else "r_cusp"
    r_rows = _as_rows(data[r_key])
    if len(r_rows) == 1 and gamma_values.size > 1:
        r_rows = [r_rows[0] for _ in gamma_values]

    if "g_gamma_grid" in data.files:
        g_rows = _as_rows(data["g_gamma_grid"])
    else:
        rho_key = "rho_cusp_grid" if "rho_cusp_grid" in data.files else "rho_cusp"
        rho_rows = _as_rows(data[rho_key])
        if len(rho_rows) == 1 and gamma_values.size > 1:
            rho_rows = [rho_rows[0] for _ in gamma_values]
        g_rows = [
            dm.g_gamma(rho_i, r_i, gamma_i, M=M)
            for rho_i, r_i, gamma_i in zip(rho_rows, r_rows, gamma_values)
        ]

    if len(r_rows) != len(gamma_values) or len(g_rows) != len(gamma_values):
        raise ValueError("The number of radial/g rows must match gamma_values.")

    return {
        "data": data,
        "gamma_values": gamma_values,
        "r_rows": r_rows,
        "g_rows": [np.asarray(row, dtype=float) for row in g_rows],
        "R_S": dm.schwarzschild_radius(M),
        "source_r_key": r_key,
        "source_g_key": "g_gamma_grid" if "g_gamma_grid" in data.files else "reconstructed_from_density",
    }


def valid_row(r, g, R_S, g_floor):
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    x = 1.0 - 4.0 * R_S / r
    mask = np.isfinite(r) & np.isfinite(g) & (r > 4.0 * R_S) & (x > 0.0) & (g > g_floor)
    return x, mask


def row_diagnostics(gamma_values, r_rows, g_rows, R_S, g_floor, min_points=8):
    rows = []
    for i, (gamma_i, r_i, g_i) in enumerate(zip(gamma_values, r_rows, g_rows)):
        x_i, mask = valid_row(r_i, g_i, R_S, g_floor)
        n = int(mask.sum())
        record = {
            "index": i,
            "gamma": float(gamma_i),
            "valid_points": n,
            "ok": False,
            "reason": "",
            "A": np.nan,
            "B": np.nan,
            "log_rmse": np.nan,
            "r2": np.nan,
            "plateau": np.nan,
            "outer_slope": np.nan,
        }
        if n < min_points:
            record["reason"] = "few_valid_points"
            rows.append(record)
            continue

        lx = np.log(x_i[mask])
        lg = np.log(np.asarray(g_i)[mask])
        if np.nanstd(lx) <= 0.0 or not np.all(np.isfinite(lx + lg)):
            record["reason"] = "degenerate_log_data"
            rows.append(record)
            continue

        B, logA = np.polyfit(lx, lg, 1)
        pred = logA + B * lx
        resid = lg - pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((lg - lg.mean()) ** 2))
        record.update(
            A=float(np.exp(logA)),
            B=float(B),
            log_rmse=float(np.sqrt(np.mean(resid**2))),
            r2=float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan,
        )
        outer = mask & (r_i >= np.nanpercentile(r_i[mask], 80.0))
        if int(outer.sum()) >= 4:
            record["plateau"] = float(np.nanmedian(np.asarray(g_i)[outer]))
            lr = np.log(r_i[outer] / R_S)
            record["outer_slope"] = float(np.polyfit(lr, np.log(np.asarray(g_i)[outer]), 1)[0])
        if record["A"] <= 0.0 or record["B"] <= 0.0 or not np.isfinite(record["A"] + record["B"]):
            record["reason"] = "non_positive_or_nonfinite_fit"
        else:
            record["ok"] = True
        rows.append(record)
    return rows


def _model_values(gamma, x, coeffs_logA, coeffs_logB, gamma_domain, coeffs_C=None):
    gh = dm.gamma_hat(gamma, gamma_domain)
    A = np.exp(chebval(gh, coeffs_logA))
    B = np.exp(chebval(gh, coeffs_logB))
    y = A * np.power(x, B)
    if coeffs_C is not None:
        C = chebval(gh, coeffs_C)
        y = y * np.exp(C * (1.0 - x))
    return y


def cross_validate_degrees(gamma_values, r_rows, g_rows, R_S, g_floor, row_rows, degrees=range(4)):
    gamma_domain = (float(np.min(gamma_values)), float(np.max(gamma_values)))
    row_ok = np.array([row["ok"] for row in row_rows], dtype=bool)
    row_A = np.array([row["A"] for row in row_rows], dtype=float)
    row_B = np.array([row["B"] for row in row_rows], dtype=float)
    results = []
    for deg_A in degrees:
        for deg_B in degrees:
            fold_errors = []
            for holdout in range(len(gamma_values)):
                train = (np.arange(len(gamma_values)) != holdout) & row_ok
                ok = row_ok[train]
                if ok.sum() <= max(deg_A, deg_B):
                    continue
                gtrain = gamma_values[train]
                Atrain = np.log(row_A[train])
                Btrain = np.log(row_B[train])
                cA = chebfit(dm.gamma_hat(gtrain, gamma_domain), Atrain, deg_A)
                cB = chebfit(dm.gamma_hat(gtrain, gamma_domain), Btrain, deg_B)
                x_h, mask_h = valid_row(r_rows[holdout], g_rows[holdout], R_S, g_floor)
                if int(mask_h.sum()) < 8:
                    continue
                pred = _model_values(gamma_values[holdout], x_h[mask_h], cA, cB, gamma_domain)
                err = np.log(g_rows[holdout][mask_h]) - np.log(np.maximum(pred, g_floor))
                fold_errors.append(float(np.sqrt(np.mean(err**2))))
            results.append(
                {
                    "deg_A": int(deg_A),
                    "deg_B": int(deg_B),
                    "cv_log_rmse": float(np.mean(fold_errors)) if fold_errors else np.inf,
                    "folds": int(len(fold_errors)),
                }
            )
    finite = [row for row in results if np.isfinite(row["cv_log_rmse"])]
    best = min(finite, key=lambda row: row["cv_log_rmse"])
    threshold = best["cv_log_rmse"] * 1.02
    selected = min(
        [row for row in finite if row["cv_log_rmse"] <= threshold],
        key=lambda row: (row["deg_A"] + row["deg_B"], row["deg_A"], row["deg_B"]),
    )
    return results, selected


def initial_coefficients(row_rows, gamma_domain, deg_A, deg_B):
    ok_rows = [row for row in row_rows if row["ok"]]
    gamma_ok = np.array([row["gamma"] for row in ok_rows], dtype=float)
    logA = np.log([row["A"] for row in ok_rows])
    logB = np.log([row["B"] for row in ok_rows])
    cA = chebfit(dm.gamma_hat(gamma_ok, gamma_domain), logA, deg_A)
    cB = chebfit(dm.gamma_hat(gamma_ok, gamma_domain), logB, deg_B)
    return cA, cB


def refine_global(gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, deg_A, deg_B, mode, coeffs_C0=None):
    row_rows = row_diagnostics(gamma_values, r_rows, g_rows, R_S, g_floor)
    cA0, cB0 = initial_coefficients(row_rows, gamma_domain, deg_A, deg_B)
    parts = [cA0, cB0]
    if coeffs_C0 is not None:
        parts.append(np.asarray(coeffs_C0, dtype=float))
    p0 = np.concatenate(parts)

    prepared = []
    for gamma_i, r_i, g_i in zip(gamma_values, r_rows, g_rows):
        x_i, mask_i = valid_row(r_i, g_i, R_S, g_floor)
        if int(mask_i.sum()) < 8:
            continue
        g_valid = np.asarray(g_i)[mask_i]
        scale = float(np.nanmedian(g_valid))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        prepared.append((float(gamma_i), x_i[mask_i], g_valid, 1.0 / np.sqrt(mask_i.sum()), scale))

    nA = deg_A + 1
    nB = deg_B + 1

    def split(params):
        cA = params[:nA]
        cB = params[nA:nA + nB]
        cC = params[nA + nB:] if coeffs_C0 is not None else None
        return cA, cB, cC

    def residuals(params):
        cA, cB, cC = split(params)
        out = []
        for gamma_i, x_i, g_i, w_i, scale_i in prepared:
            pred = _model_values(gamma_i, x_i, cA, cB, gamma_domain, cC)
            if mode == "log":
                res = np.log(np.maximum(pred, g_floor)) - np.log(g_i)
            else:
                res = (pred - g_i) / scale_i
            out.append(w_i * res)
        return np.concatenate(out)

    result = least_squares(residuals, p0, max_nfev=5000)
    cA, cB, cC = split(result.x)
    return result, cA, cB, cC, prepared


def metrics_by_row(gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, cA, cB, cC=None):
    rows = []
    all_direct = []
    all_log = []
    for i, (gamma_i, r_i, g_i) in enumerate(zip(gamma_values, r_rows, g_rows)):
        x_i, mask_i = valid_row(r_i, g_i, R_S, g_floor)
        if int(mask_i.sum()) < 1:
            continue
        pred = _model_values(float(gamma_i), x_i[mask_i], cA, cB, gamma_domain, cC)
        err = pred - np.asarray(g_i)[mask_i]
        log_err = np.log(np.maximum(pred, g_floor)) - np.log(np.asarray(g_i)[mask_i])
        rows.append(
            {
                "index": i,
                "gamma": float(gamma_i),
                "valid_points": int(mask_i.sum()),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
                "max_abs": float(np.max(np.abs(err))),
                "log_rmse": float(np.sqrt(np.mean(log_err**2))),
            }
        )
        all_direct.append(err)
        all_log.append(log_err)
    all_direct = np.concatenate(all_direct)
    all_log = np.concatenate(all_log)
    return rows, {
        "rmse": float(np.sqrt(np.mean(all_direct**2))),
        "mae": float(np.mean(np.abs(all_direct))),
        "max_abs": float(np.max(np.abs(all_direct))),
        "log_rmse": float(np.sqrt(np.mean(all_log**2))),
    }


def save_figures(out_dir, gamma_values, r_rows, g_rows, R_S, row_rows, gamma_domain, cA, cB, cC, row_metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    gamma_dense = np.linspace(gamma_domain[0], gamma_domain[1], 400)
    A_dense = np.exp(chebval(dm.gamma_hat(gamma_dense, gamma_domain), cA))
    B_dense = np.exp(chebval(dm.gamma_hat(gamma_dense, gamma_domain), cB))
    ok = np.array([row["ok"] for row in row_rows])

    plt.figure(figsize=(7, 4))
    plt.plot([row["gamma"] for row in row_rows if row["ok"]], [row["A"] for row in row_rows if row["ok"]], ".", label="A_i por fila")
    plt.plot(gamma_dense, A_dense, "-", label="A(gamma)")
    plt.xlabel("gamma")
    plt.ylabel("A")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "A_vs_gamma.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot([row["gamma"] for row in row_rows if row["ok"]], [row["B"] for row in row_rows if row["ok"]], ".", label="B_i por fila")
    plt.plot(gamma_dense, B_dense, "-", label="B(gamma)")
    plt.axhline(3.0, color="k", ls="--", lw=1, label="GS: B=3")
    plt.xlabel("gamma")
    plt.ylabel("B")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "B_vs_gamma.png", dpi=160)
    plt.close()

    selected = np.linspace(0, len(gamma_values) - 1, 6, dtype=int)
    plt.figure(figsize=(7, 5))
    for idx in selected:
        x_i, mask_i = valid_row(r_rows[idx], g_rows[idx], R_S, 1e-300)
        row = row_rows[idx]
        plt.plot(np.log(x_i[mask_i]), np.log(g_rows[idx][mask_i]), ".", ms=2, alpha=0.45, label=f"gamma={gamma_values[idx]:.2f}")
        if row["ok"]:
            lx = np.linspace(np.log(x_i[mask_i]).min(), np.log(x_i[mask_i]).max(), 150)
            plt.plot(lx, np.log(row["A"]) + row["B"] * lx, "-", lw=1)
    plt.xlabel("ln x")
    plt.ylabel("ln g")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "logg_vs_logx_rows.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for idx in selected:
        r_i = r_rows[idx]
        g_i = g_rows[idx]
        x_i = 1.0 - 4.0 * R_S / r_i
        mask = np.isfinite(x_i) & (x_i > 0) & np.isfinite(g_i)
        pred = _model_values(float(gamma_values[idx]), x_i[mask], cA, cB, gamma_domain, cC)
        plt.loglog(r_i[mask] / R_S, g_i[mask], ".", ms=2, alpha=0.45)
        plt.loglog(r_i[mask] / R_S, pred, "-", lw=1.2, label=f"fit gamma={gamma_values[idx]:.2f}")
    r_ref = np.logspace(np.log10(4.001), np.log10(max(np.nanmax(r / R_S) for r in r_rows)), 400)
    plt.loglog(r_ref, np.maximum(1.0 - 4.0 / r_ref, 0.0) ** 3, "k--", lw=1, label="GS")
    plt.xlabel("r/R_S")
    plt.ylabel("g_gamma(r)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_profiles.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for idx in selected:
        r_i = r_rows[idx]
        g_i = g_rows[idx]
        x_i, mask_i = valid_row(r_i, g_i, R_S, 1e-300)
        pred = _model_values(float(gamma_values[idx]), x_i[mask_i], cA, cB, gamma_domain, cC)
        plt.plot(np.log10(r_i[mask_i] / R_S), pred - g_i[mask_i], label=f"gamma={gamma_values[idx]:.2f}")
    plt.axhline(0.0, color="k", lw=1)
    plt.xlabel("log10(r/R_S)")
    plt.ylabel("fit - numerico")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "selected_residuals.png", dpi=160)
    plt.close()

    residual_grid = []
    x_grid = []
    for gamma_i, r_i, g_i in zip(gamma_values, r_rows, g_rows):
        x_i, mask_i = valid_row(r_i, g_i, R_S, 1e-300)
        pred = _model_values(float(gamma_i), x_i[mask_i], cA, cB, gamma_domain, cC)
        residual_grid.append(np.log10(np.maximum(pred, 1e-300)) - np.log10(np.maximum(g_i[mask_i], 1e-300)))
        x_grid.append(np.log10(r_i[mask_i] / R_S))
    common_x = np.linspace(max(row.min() for row in x_grid), min(row.max() for row in x_grid), 240)
    heat = np.vstack([np.interp(common_x, xi, ri) for xi, ri in zip(x_grid, residual_grid)])
    plt.figure(figsize=(8, 5))
    plt.imshow(heat, aspect="auto", origin="lower", extent=[common_x.min(), common_x.max(), gamma_values.min(), gamma_values.max()], cmap="coolwarm")
    plt.colorbar(label="Delta log10 g")
    plt.xlabel("log10(r/R_S)")
    plt.ylabel("gamma")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot([row["gamma"] for row in row_metrics], [row["rmse"] for row in row_metrics], label="RMSE")
    plt.plot([row["gamma"] for row in row_metrics], [row["max_abs"] for row in row_metrics], label="max abs")
    plt.xlabel("gamma")
    plt.ylabel("error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "errors_by_gamma.png", dpi=160)
    plt.close()


def write_summary(path, selected, model_name, gamma_domain, r_domain, cA, cB, cC, metrics, direct_metrics, log_metrics, row_rows):
    mean_B = float(np.mean([row["B"] for row in row_rows if row["ok"]]))
    max_outer_slope = float(np.nanmax(np.abs([row["outer_slope"] for row in row_rows])))
    warnings = []
    if max_outer_slope > 0.05:
        warnings.append("La pendiente exterior no tiende claramente a cero en todas las filas; g_gamma podria absorber diferencias de pendiente.")
    if metrics["log_rmse"] > 0.1:
        warnings.append("Queda estructura sistematica apreciable en residuos logaritmicos; revisar la curvatura radial.")
    warnings.append("A(gamma) es una normalizacion efectiva y puede reflejar rho_R, R_sp o gamma_sp, no solo fisica de captura.")

    formula = "A(gamma) x**B(gamma)"
    if cC is not None:
        formula += " exp(C(gamma) (1 - x))"

    text = [
        "Ajuste global de g_gamma(r)",
        "",
        f"Formula final: g_fit(r,gamma) = {formula}, x = 1 - 4 R_S/r.",
        f"Modelo seleccionado: {model_name}",
        f"Grados seleccionados: A={selected['deg_A']}, B={selected['deg_B']}",
        f"Dominio gamma: [{gamma_domain[0]:.8g}, {gamma_domain[1]:.8g}]",
        f"Dominio r/R_S usado: [{r_domain[0]:.8g}, {r_domain[1]:.8g}]",
        "",
        f"Coeficientes Chebyshev log A: {np.array2string(cA, precision=12)}",
        f"Coeficientes Chebyshev log B: {np.array2string(cB, precision=12)}",
    ]
    if cC is not None:
        text.append(f"Coeficientes Chebyshev C: {np.array2string(cC, precision=12)}")
    text.extend(
        [
            "",
            f"Metricas finales: RMSE={metrics['rmse']:.8e}, MAE={metrics['mae']:.8e}, max_abs={metrics['max_abs']:.8e}, log_RMSE={metrics['log_rmse']:.8e}",
            f"Comparacion directa: RMSE={direct_metrics['rmse']:.8e}, log_RMSE={direct_metrics['log_rmse']:.8e}",
            f"Comparacion log: RMSE={log_metrics['rmse']:.8e}, log_RMSE={log_metrics['log_rmse']:.8e}",
            f"B_i medio por filas: {mean_B:.8g}; referencia Gondolo-Silk B=3.",
            "",
            "Advertencias:",
        ]
    )
    text.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return warnings


def run_analysis(input_file, output_dir, g_floor=1e-12, M=2.6e6):
    loaded = load_capture_data(input_file, M=M)
    gamma_values = loaded["gamma_values"]
    r_rows = loaded["r_rows"]
    g_rows = loaded["g_rows"]
    R_S = loaded["R_S"]
    gamma_domain = (float(np.min(gamma_values)), float(np.max(gamma_values)))
    r_domain = (
        float(min(np.nanmin(np.asarray(r) / R_S) for r in r_rows)),
        float(max(np.nanmax(np.asarray(r) / R_S) for r in r_rows)),
    )

    row_rows = row_diagnostics(gamma_values, r_rows, g_rows, R_S, g_floor)
    cv_results, selected = cross_validate_degrees(gamma_values, r_rows, g_rows, R_S, g_floor, row_rows)

    direct_result, cA_direct, cB_direct, cC_direct, _ = refine_global(
        gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, selected["deg_A"], selected["deg_B"], "direct"
    )
    log_result, cA_log, cB_log, cC_log, _ = refine_global(
        gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, selected["deg_A"], selected["deg_B"], "log"
    )
    direct_rows, direct_metrics = metrics_by_row(gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, cA_direct, cB_direct)
    log_rows, log_metrics = metrics_by_row(gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, cA_log, cB_log)

    simple_is_enough = log_metrics["log_rmse"] <= 0.1
    model_name = "simple_log_refined"
    cA_final, cB_final, cC_final = cA_log, cB_log, None
    final_rows, final_metrics = log_rows, log_metrics

    alt = None
    if not simple_is_enough:
        cC0 = np.zeros(2)
        alt_result, cA_alt, cB_alt, cC_alt, _ = refine_global(
            gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, selected["deg_A"], selected["deg_B"], "log", cC0
        )
        alt_rows, alt_metrics = metrics_by_row(gamma_values, r_rows, g_rows, R_S, g_floor, gamma_domain, cA_alt, cB_alt, cC_alt)
        alt = {"result": alt_result, "rows": alt_rows, "metrics": alt_metrics, "coeffs_C": cC_alt}
        if alt_metrics["log_rmse"] < 0.9 * log_metrics["log_rmse"]:
            model_name = "alternative_log_refined"
            cA_final, cB_final, cC_final = cA_alt, cB_alt, cC_alt
            final_rows, final_metrics = alt_rows, alt_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    save_figures(fig_dir, gamma_values, r_rows, g_rows, R_S, row_rows, gamma_domain, cA_final, cB_final, cC_final, final_rows)

    A_row = np.array([row["A"] for row in row_rows])
    B_row = np.array([row["B"] for row in row_rows])
    A_fit = np.exp(chebval(dm.gamma_hat(gamma_values, gamma_domain), cA_final))
    B_fit = np.exp(chebval(dm.gamma_hat(gamma_values, gamma_domain), cB_final))
    plateau = np.array([row["plateau"] for row in row_rows])
    outer_slope = np.array([row["outer_slope"] for row in row_rows])
    valid_counts = np.array([row["valid_points"] for row in row_rows], dtype=int)
    ok_rows = np.array([row["ok"] for row in row_rows], dtype=bool)

    npz_path = output_dir / "capture_factor_global_fit.npz"
    np.savez(
        npz_path,
        cheb_coeffs_logA=cA_final,
        cheb_coeffs_logB=cB_final,
        cheb_coeffs_C=np.array([]) if cC_final is None else cC_final,
        selected_deg_A=np.array(selected["deg_A"]),
        selected_deg_B=np.array(selected["deg_B"]),
        gamma_domain=np.array(gamma_domain),
        r_over_R_S_domain=np.array(r_domain),
        gamma_values=gamma_values,
        A_row=A_row,
        B_row=B_row,
        A_fit=A_fit,
        B_fit=B_fit,
        plateau_outer=plateau,
        outer_log_slope=outer_slope,
        valid_counts=valid_counts,
        ok_rows=ok_rows,
        cv_results=np.array(cv_results, dtype=object),
        row_metrics=np.array(final_rows, dtype=object),
        global_metrics=np.array(final_metrics, dtype=object),
        direct_global_metrics=np.array(direct_metrics, dtype=object),
        log_global_metrics=np.array(log_metrics, dtype=object),
        g_floor=np.array(g_floor),
        R_S=np.array(R_S),
        source_file=np.array(str(input_file)),
        source_r_key=np.array(loaded["source_r_key"]),
        source_g_key=np.array(loaded["source_g_key"]),
        model_name=np.array(model_name),
    )

    summary_path = output_dir / "capture_factor_global_fit_summary.txt"
    warnings = write_summary(
        summary_path, selected, model_name, gamma_domain, r_domain, cA_final, cB_final, cC_final,
        final_metrics, direct_metrics, log_metrics, row_rows
    )

    report = {
        "npz": str(npz_path),
        "summary": str(summary_path),
        "figures": str(fig_dir),
        "model_name": model_name,
        "formula": "A(gamma) x**B(gamma)" if cC_final is None else "A(gamma) x**B(gamma) exp(C(gamma)(1-x))",
        "selected": selected,
        "coeffs_logA": cA_final.tolist(),
        "coeffs_logB": cB_final.tolist(),
        "coeffs_C": None if cC_final is None else cC_final.tolist(),
        "metrics": final_metrics,
        "direct_metrics": direct_metrics,
        "log_metrics": log_metrics,
        "warnings": warnings,
    }
    (output_dir / "capture_factor_global_fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/gs_numerical_reconstruction.npz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--g-floor", type=float, default=1e-12)
    parser.add_argument("--M", type=float, default=2.6e6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"capture_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report = run_analysis(Path(args.input), output_dir, g_floor=args.g_floor, M=args.M)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
