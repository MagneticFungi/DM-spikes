"""Manual storage helpers for density profile arrays."""

from datetime import datetime
from pathlib import Path

import numpy as np


def save_density_profiles(
    r_iso,
    rho_iso,
    gamma_values,
    r_cusp,
    rho_cusp_grid,
    etiqueta,
    carpeta="results",
):
    """Save separated isothermal and cusp profiles to a timestamped .npz file."""
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = carpeta / f"perfiles_densidad_{etiqueta}_{timestamp}.npz"

    if archivo.exists():
        raise FileExistsError(f"El archivo ya existe: {archivo.resolve()}")

    np.savez(
        archivo,
        r_iso=r_iso,
        rho_iso=rho_iso,
        gamma_values=gamma_values,
        r_cusp=r_cusp,
        rho_cusp_grid=rho_cusp_grid,
    )

    print(f"Perfiles guardados en:\n{archivo.resolve()}")

    return archivo


def load_density_profiles(archivo):
    """Load saved profile arrays from a .npz file."""
    data = np.load(archivo)
    return {
        "r_iso": data["r_iso"],
        "rho_iso": data["rho_iso"],
        "gamma_values": data["gamma_values"],
        "r_cusp": data["r_cusp"],
        "rho_cusp_grid": data["rho_cusp_grid"],
    }


def list_density_profiles(carpeta="results"):
    """Return saved density-profile .npz files in carpeta."""
    return sorted(
        archivo
        for archivo in Path(carpeta).glob("perfiles_densidad_*.npz")
        if "g_reconstruction" not in archivo.name
    )
