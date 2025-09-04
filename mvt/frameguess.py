# mvt/frameguess.py
from __future__ import annotations
import sys
import argparse
import numpy as np

C_KMS = 299_792.458

def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation → robust sigma (≈ std) via 1.4826 * MAD."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

def _nan_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Robust correlation with NaN guarding; returns np.nan if undefined."""
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 5:
        return np.nan
    x = x[m]; y = y[m]
    sx = np.nanstd(x); sy = np.nanstd(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def _read_framecheck(csv_path: str):
    """Read the CSV produced by run_frame_qa()."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    # Required columns for verdict:
    try:
        v0 = np.asarray(data["v0_stellar_kms"], float)
        berv = np.asarray(data["berv_kms"], float)
    except Exception as e:
        raise KeyError(
            f"framecheck.csv missing required columns: 'v0_stellar_kms' and/or 'berv_kms' "
            f"(error: {e}). Did run_frame_qa() run and write the expected header?"
        )
    # Optional columns:
    in_tr = None
    if "in_transit" in data.dtype.names:
        in_tr = np.asarray(data["in_transit"], int) == 1
    return v0, berv, in_tr

def _verdict(v0: np.ndarray, berv: np.ndarray, *, thr_ratio=0.30, corr_thr=0.50):
    """
    One-line verdict: 'barycentric' vs 'native' based on scatter & correlation.

    Heuristic:
      - If robust scatter of v0* is comparable to BERV (ratio > thr_ratio)
        AND |corr(v0*, BERV)| >= corr_thr → looks **native** (BERV not removed).
      - else → looks **barycentric** (typical ESPRESSO DRS wavelength grid).
    """
    m = np.isfinite(v0) & np.isfinite(berv)
    v0 = v0[m]; berv = berv[m]
    if v0.size < 5:
        return ("unknown", "not enough finite rows for a verdict", np.nan, np.nan, np.nan, np.nan)

    s_v0   = _mad(v0)     # km/s
    s_berv = _mad(berv)   # km/s
    corr   = _nan_corr(v0, berv)
    ratio  = (s_v0 / s_berv) if s_berv > 0 else np.inf
    medabs = float(np.nanmedian(np.abs(v0)))

    if np.isfinite(corr) and (ratio > thr_ratio) and (abs(corr) >= corr_thr):
        guess  = "native"
        reason = f"v0_stellar scatter {s_v0:.2f} ~ BERV scatter {s_berv:.2f} (corr={corr:.2f})"
    else:
        guess  = "barycentric"
        reason = f"v0_stellar scatter {s_v0:.2f} << BERV scatter {s_berv:.2f} (corr={corr:.2f})"

    return (guess, reason, s_v0, s_berv, corr, medabs)

def _impact(v0: np.ndarray,
            berv: np.ndarray,
            *,
            sigma_kms: float,
            center_A: float,
            use_mask: np.ndarray | None = None):
    """
    Impact metrics (smear + wavelength displacement):
      - R_frame = |rho| * MAD(BERV) / sigma_line
      - Predicted depth attenuation = 1/sqrt(1 + R^2)
      - Predicted width increase = (sqrt(1 + R^2) - 1)
      - Δλ_mismatch_A ≈ λ0 * (|rho|*MAD(BERV))/c   (smear scale in Å)
      - Δλ_bias_med_A ≈ λ0 * (|median(v0)|)/c      (constant bias in Å)
    """
    if use_mask is None:
        m = np.isfinite(v0) & np.isfinite(berv)
    else:
        m = np.isfinite(v0) & np.isfinite(berv) & np.asarray(use_mask, bool)

    v0 = v0[m]; berv = berv[m]
    if v0.size < 5:
        return None

    s_berv = _mad(berv)
    rho    = _nan_corr(v0, berv)
    v0_med = float(np.nanmedian(v0))

    sigma_line = float(sigma_kms)
    fwhm_kms   = 2.355 * sigma_line
    fwhm_A     = center_A * (fwhm_kms / C_KMS)

    sigma_mismatch_kms = (abs(rho) * s_berv) if np.isfinite(rho) else np.nan
    R = sigma_mismatch_kms / sigma_line if (np.isfinite(sigma_mismatch_kms) and sigma_line > 0) else np.nan
    dlam_mismatch_A = center_A * (sigma_mismatch_kms / C_KMS) if np.isfinite(sigma_mismatch_kms) else np.nan

    dlam_bias_med_A = center_A * (abs(v0_med) / C_KMS) if np.isfinite(v0_med) else np.nan

    depth_factor = 1.0 / np.sqrt(1.0 + R*R) if np.isfinite(R) else np.nan
    loss_pct = (1.0 - depth_factor) * 100.0 if np.isfinite(depth_factor) else np.nan
    width_increase_pct = (np.sqrt(1.0 + R*R) - 1.0) * 100.0 if np.isfinite(R) else np.nan

    return dict(
        rho=rho,
        mad_berv=s_berv,
        v0_med=v0_med,
        R_frame=R,
        depth_factor=depth_factor,
        depth_loss_pct=loss_pct,
        width_increase_pct=width_increase_pct,
        smear_kms=sigma_mismatch_kms,
        smear_A=dlam_mismatch_A,
        bias_A=dlam_bias_med_A,
        fwhm_kms=fwhm_kms,
        fwhm_A=fwhm_A,
        bias_vs_fwhm=(abs(v0_med)/fwhm_kms) if fwhm_kms > 0 else np.nan,
        smear_vs_fwhm=(sigma_mismatch_kms/fwhm_kms) if fwhm_kms > 0 else np.nan
    )

def verdict_from_framecheck(csv_path: str,
                            thr_ratio: float = 0.30,
                            corr_thr: float = 0.50,
                            sigma_kms: float | None = None,
                            center_A: float | None = None,
                            use_all_rows: bool = False) -> None:
    """
    Print consolidated assessment.
      - Always prints one-line verdict.
      - If sigma_kms & center_A are provided → prints impact metrics too.
    """
    v0, berv, in_tr = _read_framecheck(csv_path)

    guess, reason, s_v0, s_berv, corr, medabs = _verdict(
        v0, berv, thr_ratio=thr_ratio, corr_thr=corr_thr
    )

    # One-line verdict (keeps backward compatibility)
    print(
        f"Input wavelength frame looks **{guess}** — {reason}; "
        f"median|v0_stellar|={medabs:.2f} km/s — file: {csv_path}"
    )

    # Impact block (optional)
    if sigma_kms is not None and center_A is not None:
        mask = None if (use_all_rows or in_tr is None) else (in_tr == 1)
        impact = _impact(v0, berv, sigma_kms=float(sigma_kms), center_A=float(center_A), use_mask=mask)
        if impact is None:
            print("Impact: not enough finite rows to estimate metrics.")
            return

        scope = "ALL exposures" if (use_all_rows or in_tr is None) else "in-transit only"
        print(
            "\nImpact assessment "
            f"(σ_line={float(sigma_kms):.3f} km/s, λ0={float(center_A):.3f} Å, {scope}):"
        )
        print(
            "  Smear driver: R_frame = |ρ|·MAD(BERV)/σ_line = "
            f"{impact['R_frame']:.3f}  (ρ={impact['rho']:.3f}, MAD(BERV)={impact['mad_berv']:.3f} km/s)"
        )
        print(
            f"  Predicted depth attenuation = {impact['depth_factor']:.3f}  "
            f"(loss ≈ {impact['depth_loss_pct']:.1f}%)"
        )
        print(
            f"  Predicted width increase    ≈ {impact['width_increase_pct']:.1f}%"
        )
        print(
            f"  Velocity smear scale ≈ {impact['smear_kms']:.3f} km/s "
            f"({impact['smear_vs_fwhm']:.3f} × FWHM={impact['fwhm_kms']:.3f} km/s)"
        )
        print(
            f"  Wavelength smear scale ≈ {impact['smear_A']:.5f} Å;   "
            f"bias (median) ≈ {impact['bias_A']:.5f} Å "
            f"({impact['bias_vs_fwhm']:.3f} × FWHM={impact['fwhm_A']:.4f} Å)"
        )
    else:
        # gentle hint to enable impact block
        print("(Pass --sigma-kms and --center-A to also get smearing and wavelength-displacement impact.)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Consolidated frame verdict + impact from framecheck.csv"
    )
    ap.add_argument("csv", help="outputs/.../framecheck.csv")
    ap.add_argument("--thr-ratio", type=float, default=0.30,
                    help="threshold on s(v0*)/s(BERV) for 'native' (default 0.30)")
    ap.add_argument("--corr-thr", type=float, default=0.50,
                    help="|corr(v0*, BERV)| threshold for 'native' (default 0.50)")
    ap.add_argument("--sigma-kms", type=float, default=None,
                    help="intrinsic line sigma in km/s (enables impact block)")
    ap.add_argument("--center-A", type=float, default=None,
                    help="line center wavelength in Å (enables impact block)")
    ap.add_argument("--all", action="store_true",
                    help="use ALL rows for impact (default: in-transit only when available)")
    args = ap.parse_args()

    verdict_from_framecheck(
        args.csv,
        thr_ratio=args.thr_ratio,
        corr_thr=args.corr_thr,
        sigma_kms=args.sigma_kms,
        center_A=args.center_A,
        use_all_rows=args.all,
    )