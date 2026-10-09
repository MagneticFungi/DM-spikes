"""Harmonic saturation of a supplied dark-matter spike density."""

import math

import numpy as np


def rho_spike(rho_cusp, m, observable_sigma_v, bh_age=1e10):
    """Return rho_sp = (rho_cusp^-1 + rho_sat^-1)^-1.

    ``rho_sat = m / (observable_sigma_v * bh_age)`` must be in the same
    density units as ``rho_cusp``. The three saturation inputs are positive
    scalars; ``rho_cusp`` may be a nonnegative scalar or array. An infinite
    cusp density is allowed and gives the finite saturation limit.
    """
    values = {}
    for name, raw in (
        ("m", m),
        ("observable_sigma_v", observable_sigma_v),
        ("bh_age", bh_age),
    ):
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be positive and finite") from exc
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite")
        values[name] = value

    denominator = values["observable_sigma_v"] * values["bh_age"]
    if not (math.isfinite(denominator) and denominator > 0.0):
        raise ArithmeticError("saturation inputs are outside the float64 range")
    saturation = values["m"] / denominator
    if not (math.isfinite(saturation) and saturation > 0.0):
        raise ArithmeticError("saturation density is outside the float64 range")

    rho_cusp = np.asarray(rho_cusp, dtype=float)
    if np.any(np.isnan(rho_cusp) | (rho_cusp < 0.0)):
        raise ValueError("rho_cusp must be nonnegative and contain no NaN")
    small = np.minimum(rho_cusp, saturation)
    large = np.maximum(rho_cusp, saturation)
    result = small / (1.0 + small / large)
    return float(result) if result.ndim == 0 else result


__all__ = ["rho_spike"]
