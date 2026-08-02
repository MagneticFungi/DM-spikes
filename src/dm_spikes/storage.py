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


def save_g_reconstruction(
    r_values,
    g_gondolo_silk,
    gamma_values,
    g_reconstruction_grid,
    etiqueta="g_reconstruction",
    carpeta="results",
):
    """Save reconstructed g_gamma arrays for radii in pc."""
    carpeta = Path(carpeta)
    carpeta.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = carpeta / f"{etiqueta}_{timestamp}.npz"

    if archivo.exists():
        raise FileExistsError(f"El archivo ya existe: {archivo}")

    np.savez(
        archivo,
        r_values=np.asarray(r_values),
        g_gondolo_silk=np.asarray(g_gondolo_silk),
        gamma_values=np.asarray(gamma_values),
        g_reconstruction_grid=np.asarray(g_reconstruction_grid),
    )

    print(f"Reconstrucciones de g guardadas en:\n{archivo.resolve()}")
    return archivo


def load_g_reconstruction(archivo):
    """Load reconstructed g_gamma arrays saved in an .npz file."""
    data = np.load(archivo)
    keys = set(data.files)
    expected_keys = {
        "r_values",
        "g_gondolo_silk",
        "gamma_values",
        "g_reconstruction_grid",
    }

    if expected_keys.issubset(keys):
        return {key: data[key] for key in expected_keys}

    legacy_keys = {"r_cusp", "rho_iso", "gamma_values", "rho_cusp_grid"}
    if legacy_keys.issubset(keys):
        return {
            "r_values": data["r_cusp"],
            "g_gondolo_silk": data["rho_iso"],
            "gamma_values": data["gamma_values"],
            "g_reconstruction_grid": data["rho_cusp_grid"],
        }

    raise KeyError(
        "El archivo no contiene una reconstruccion de g valida. "
        f"Claves disponibles: {data.files}"
    )


def list_g_reconstructions(carpeta="results"):
    """List saved reconstructed g_gamma .npz files."""
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []

    archivos = set(carpeta.glob("g_reconstruction_*.npz"))
    archivos.update(carpeta.glob("perfiles_densidad_g_reconstruction_*.npz"))
    return sorted(archivos)
