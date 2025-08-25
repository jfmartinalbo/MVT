# mvt/io_espre.py
"""
I/O helpers for ESPRESSO S1D spectra.

Compat:
    read_s1d(path) -> (wave, flux, hdr, bjd, berv, rv_star)   # <- para tu run_mvt_espre.py

Nueva API (opcional):
    read_s1d_detailed(path) -> (wave, flux, err, meta)

- wave: 1D numpy array [Angstrom]
- flux: 1D numpy array [erg s^-1 cm^-2 Angstrom^-1]
- err : 1D numpy array o None
- hdr : astropy.io.fits.Header (primario)
- bjd, berv, rv_star: floats (o None si faltan)
- meta: dict con 'primary_header' y extras
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from astropy.io import fits

# ---------- internal utilities ----------

def _colmap(table_hdu) -> Dict[str, str]:
    names = list(table_hdu.columns.names or [])
    return {name.lower(): name for name in names}

def _read_from_bintable(hdu) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    cmap = _colmap(hdu)

    def pick(*cands: str):
        for c in cands:
            key = c.lower()
            if key in cmap:
                return np.asarray(hdu.data[cmap[key]])
        return None

    wave = pick("wavelength", "lambda", "wave")
    flux = pick("flux_cal", "flux", "fluxcor", "fluxcorr")
    err  = pick("error_cal", "err_cal", "error", "err")

    if wave is None or flux is None:
        raise KeyError(
            "BINTABLE found but required columns are missing. "
            f"Available columns: {list(cmap.values())}"
        )

    wave = np.ascontiguousarray(wave.ravel())
    flux = np.ascontiguousarray(flux.ravel())
    if err is not None:
        err = np.ascontiguousarray(err.ravel())

    return wave, flux, err

def _approx_linear_wcs(wave: np.ndarray, rtol: float = 1e-6):
    n = int(wave.size)
    if n < 2:
        return None, None, n
    dw = np.diff(wave)
    if np.allclose(dw, dw[0], rtol=rtol, atol=0.0):
        return float(wave[0]), float(dw[0]), n
    return None, None, n

def _read_from_image(hdu0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    hdr = hdu0.header
    for k in ("CRVAL1", "CDELT1", "NAXIS1"):
        if k not in hdr:
            raise KeyError(f"Missing header key {k} for image-style S1D.")
    if hdu0.data is None:
        raise ValueError("Primary HDU has no data.")

    flux = np.asarray(hdu0.data).ravel()
    n = int(hdr["NAXIS1"])
    if flux.size != n:
        flux = flux[:n]

    wave0 = float(hdr["CRVAL1"])
    dw = float(hdr["CDELT1"])
    wave = wave0 + dw * np.arange(n, dtype=float)
    return np.ascontiguousarray(wave), np.ascontiguousarray(flux), None

def _extract_header_numbers(hdr):
    def _g(*keys):
        for k in keys:
            if k in hdr and hdr[k] is not None:
                try:
                    return float(hdr[k])
                except Exception:
                    pass
        return None

    # BJD (acepta BJD_TDB y variantes QC/DRS) y BERV
    bjd  = _g("BJD_TDB", "BJD", "HIERARCH ESO QC BJD", "HIERARCH ESO DRS BJD")
    berv = _g("BERV", "HIERARCH ESO QC BERV", "HIERARCH ESO DRS BERV")

    # RV estrella: acepta tu RV_STAR y variantes CCF/OBJ
    rv_star = _g("RV_STAR",
                 "HIERARCH ESO OCS OBJ RV", "OCS OBJ RV",
                 "HIERARCH ESO DRS CCF RV", "HIERARCH ESO QC CCF RV",
                 "HIERARCH ESO QC RV", "RV")

    return bjd, berv, rv_star

# ---------- nueva API detallada ----------

def read_s1d_detailed(path: str):
    """
    Lee un producto ESPRESSO S1D.

    Estrategia:
      1) Si HDU 1 es BINTABLE con columnas wavelength/flux_cal -> úsalo.
      2) Si no, intenta WCS lineal desde el primario (CRVAL1/CDELT1/NAXIS1).

    Devuelve:
        wave, flux, err, meta
    """
    with fits.open(path, memmap=True) as hdul:
        hdr0 = hdul[0].header
        meta = {
            "primary_header": hdr0,
            "path": path,
            "BJD": hdr0.get("HIERARCH ESO QC BJD"),
            "BERV": hdr0.get("HIERARCH ESO QC BERV"),
            "OBJECT": hdr0.get("OBJECT"),
            "DATE-OBS": hdr0.get("DATE-OBS"),
        }

        bintable_error = None
        if len(hdul) > 1 and getattr(hdul[1], "is_image", True) is False:
            try:
                wave, flux, err = _read_from_bintable(hdul[1])
                crval1, cdelt1, n = _approx_linear_wcs(wave)
                meta["linear_wcs"] = (
                    None if crval1 is None or cdelt1 is None
                    else {"CRVAL1": crval1, "CDELT1": cdelt1, "NAXIS1": n}
                )
                return wave, flux, err, meta
            except Exception as e:
                bintable_error = e

        # Fallback imagen
        try:
            wave, flux, err = _read_from_image(hdul[0])
            meta["linear_wcs"] = {
                "CRVAL1": wave[0],
                "CDELT1": (wave[1]-wave[0]) if wave.size > 1 else None,
                "NAXIS1": wave.size
            }
            return wave, flux, err, meta
        except Exception as image_error:
            msg = [
                f"Could not read S1D spectrum from {path}.",
                f"- BINTABLE read error: {bintable_error!r}" if bintable_error else "- No usable BINTABLE in HDU 1.",
                f"- Image-style read error: {image_error!r}",
            ]
            raise RuntimeError("\n".join(msg)) from None

# ---------- compatibilidad con tu script ----------

def read_s1d(path: str):
    """
    Compat layer para run_mvt_espre.py:
    Devuelve (wave, flux, hdr, bjd, berv, rv_star)
    """
    wave, flux, err, meta = read_s1d_detailed(path)
    hdr = meta["primary_header"]
    bjd, berv, rv_star = _extract_header_numbers(hdr)
    # valores por defecto seguros para el pipeline
    if bjd is None:
        bjd = np.nan
    if berv is None:
        berv = 0.0
    if rv_star is None:
        rv_star = 0.0
    return wave, flux, hdr, bjd, berv, rv_star

# Alias historical
load_spectrum = read_s1d