# mvt/frameqa.py
from __future__ import annotations
import os
import numpy as np
from typing import Tuple
from .matrix import make_matrix_lambda
from .fitlines import fit_gaussian_velocity

C_KMS = 299_792.458

def _vgrid_from_lambda(lam_grid_A: np.ndarray, center_A: float) -> np.ndarray:
    lam_grid_A = np.asarray(lam_grid_A, float)
    return C_KMS * (lam_grid_A - float(center_A)) / float(center_A)

def _fit_per_row(lam_grid_A: np.ndarray, R: np.ndarray,
                 center_A: float, window_kms: float, init_sigma_kms: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    vgrid = _vgrid_from_lambda(lam_grid_A, center_A)
    mwin = (vgrid >= -window_kms) & (vgrid <= +window_kms)
    v0_kms = np.full(R.shape[0], np.nan, float)
    depth  = np.full(R.shape[0], np.nan, float)
    for i in range(R.shape[0]):
        y = R[i, mwin]; x = vgrid[mwin]
        if np.isfinite(y).sum() >= 8:
            fit = fit_gaussian_velocity(x, y, init_sigma_kms, center_A=center_A)
            v0_kms[i] = float(getattr(fit, "v0_kms", np.nan))
            depth[i]  = float(getattr(fit, "depth",   np.nan))
    return v0_kms, depth

def _baseline_band_stats(vgrid: np.ndarray, R: np.ndarray, band_kms=(-120.0, -80.0)):
    j = (vgrid >= band_kms[0]) & (vgrid <= band_kms[1])
    med = np.nanmedian(R[:, j], axis=1)
    mad = 1.4826 * np.nanmedian(np.abs(R[:, j] - med[:, None]), axis=1)
    return med, mad

def run_frame_qa(
    wavelength_A, flux,
    lam_sr, R_sr,
    lam_pr, R_pr,
    bjd_tdb_arr, phases, in_transit,
    berv_kms, rv_star_kms,
    center_A, half_A,
    cfg, fig_dir, tab_dir,
    input_frame: str = "native",        # ← NEW
) -> None:
    # Build barycentric residuals on the SAME Npix as the others
    Npix = int(cfg.get("figures", {}).get("colormap_Npix", 401))
    lam_by, R_by, _ = make_matrix_lambda(
        wavelength_A, flux,
        frame="barycentric", quantity="residuals",
        center_A=center_A, half_A=half_A,
        bervs=berv_kms, rv_stars=rv_star_kms,
        in_transit=in_transit,
        Npix=Npix,
        sideband_flatten=True,
        subtract_additive_trend=cfg.get("residuals", {}).get("subtract_additive_trend", True),
        trend_gap_kms=cfg.get("residuals", {}).get("sideband_gap_kms", 40.0),
        input_frame=input_frame,        # ← respect how ESPRESSO files are encoded
    )

    init_sigma_A   = float(cfg["fit"]["init_sigma_A"])
    init_sigma_kms = C_KMS * init_sigma_A / float(center_A)
    fit_window_kms = float(cfg["fit"].get("window_kms", 15.0))

    # Per-row centroids & depths
    v0_stellar, d_stellar = _fit_per_row(lam_sr, R_sr, center_A, fit_window_kms, init_sigma_kms)
    v0_bary,   d_bary    = _fit_per_row(lam_by, R_by, center_A, fit_window_kms, init_sigma_kms)
    v0_planet, d_planet  = _fit_per_row(lam_pr, R_pr, center_A, fit_window_kms, init_sigma_kms)

    # Baseline QC on planet-rest matrix
    vgrid_pr = _vgrid_from_lambda(lam_pr, center_A)
    band_kms = tuple(cfg.get("debug", {}).get("off_line_kms", [-120.0, -80.0]))
    base_med, base_mad = _baseline_band_stats(vgrid_pr, R_pr, band_kms=band_kms)

    finite_frac = np.isfinite(R_pr).mean(axis=1)

    # CSV
    os.makedirs(tab_dir, exist_ok=True)
    qa_csv = os.path.join(tab_dir, "framecheck.csv")
    with open(qa_csv, "w") as f:
        f.write("idx,bjd,phase,in_transit,berv_kms,rv_star_kms,"
                "v0_stellar_kms,v0_bary_kms,v0_planet_kms,"
                "depth_stellar_pct,depth_bary_pct,depth_planet_pct,"
                "baseline_offline,baseline_mad_offline,finite_frac\n")
        for i in range(len(bjd_tdb_arr)):
            f.write(
                f"{i},{bjd_tdb_arr[i]:.6f},{phases[i]:+.6f},{int(in_transit[i])},"
                f"{(berv_kms[i] if berv_kms is not None else np.nan):.3f},"
                f"{(rv_star_kms[i] if rv_star_kms is not None else np.nan):.3f},"
                f"{v0_stellar[i]:.3f},{v0_bary[i]:.3f},{v0_planet[i]:.3f},"
                f"{(100.0*d_stellar[i] if np.isfinite(d_stellar[i]) else np.nan):.3f},"
                f"{(100.0*d_bary[i]    if np.isfinite(d_bary[i])    else np.nan):.3f},"
                f"{(100.0*d_planet[i]  if np.isfinite(d_planet[i])  else np.nan):.3f},"
                f"{base_med[i]:.4e},{base_mad[i]:.4e},{finite_frac[i]:.3f}\n"
            )

    it = np.asarray(in_transit, bool)
    msg = (f"Frame QA: median|v0_stellar|={np.nanmedian(np.abs(v0_stellar)):.2f} km/s; "
           f"median|v0_bary|={np.nanmedian(np.abs(v0_bary)):.2f} km/s; "
           f"median_IT|v0_planet|={np.nanmedian(np.abs(v0_planet[it])):.2f} km/s; "
           f"median_IT depth={np.nanmedian(100.0*d_planet[it]):.3f}%  → wrote {qa_csv}")
    print(msg)