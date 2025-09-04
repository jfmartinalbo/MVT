# run_mvt_espre.py
# -*- coding: utf-8 -*-
"""
ESPRESSO Na I D₂ Transmission Pipeline — single-night reduction
"""

from __future__ import annotations
import argparse
from pathlib import Path
import os
from typing import List, Tuple

import yaml
import numpy as np
import numpy.typing as npt

from mvt.discover import list_s1d_files
from mvt.io_espre import read_s1d
from mvt.timephase import Ephemeris, contacts_from_yaml, phases_from_bjd, in_transit_mask
from mvt.fitlines import fit_gaussian_velocity
from mvt.plots import plot_stack, plot_nulls_grid
from mvt.saveio import write_table71
from mvt.nulls import (
    null_oot_only, null_wrong_rest_frame, null_phase_shuffle, null_control_window
)
from mvt.stack import robust_nanmedian

# colour-map helpers (single unified API, wavelength x-axis only)
from mvt.matrix import make_matrix_lambda
from mvt.plots  import plot_colormap_lambda, plot_window_examples, plot_residuals_matrix
from mvt.rv     import to_stellar_rest

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
C_KMS: float = 299_792.458  # speed of light [km/s]


# ----------------------------------------------------------------------------- #
# Small helpers kept near the pipeline for readability
# ----------------------------------------------------------------------------- #
def _default_cfg_path() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "configs" / "hd189733_naid.yaml")


def _fit_amp_from_profile(
    v_grid_kms: npt.NDArray[np.floating],
    profile:    npt.NDArray[np.floating],
    window_kms: float,
    init_sigma_kms: float,
    center_kms: float = 0.0,
) -> float:
    mwin = (v_grid_kms >= center_kms - window_kms) & (v_grid_kms <= center_kms + window_kms)
    if np.count_nonzero(mwin) < 5 or not np.any(np.isfinite(profile[mwin])):
        return np.nan
    fit = fit_gaussian_velocity(v_grid_kms[mwin], profile[mwin], init_sigma_kms, center_A=5895.924)
    return float(abs(getattr(fit, "depth", np.nan)))


def _bootstrap_amp(
    v_grid_kms: npt.NDArray[np.floating],
    profiles_exp_by_v: npt.NDArray[np.floating],
    window_kms: float,
    init_sigma_kms: float,
    nboot: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(rng)
    profiles = np.asarray(profiles_exp_by_v, float)
    n_exp = profiles.shape[0]
    amps: List[float] = []
    for _ in range(int(nboot)):
        idx = rng.integers(0, n_exp, size=n_exp)
        stack = robust_nanmedian(profiles[idx], axis=0, min_valid=3)
        amps.append(_fit_amp_from_profile(v_grid_kms, stack, window_kms, init_sigma_kms))
    amps_arr = np.asarray(amps, float)
    a_med = float(np.nanmedian(amps_arr))
    a_lo  = float(np.nanpercentile(amps_arr, 16))
    a_hi  = float(np.nanpercentile(amps_arr, 84))
    a_lo, a_med, a_hi = np.sort([a_lo, a_med, a_hi])
    return a_med, a_lo, a_hi


def _sideband_metrics(w, f, center_A, half_A, gap_kms=30.0):
    gap_A = center_A * (gap_kms / C_KMS)
    mL = (w >= center_A - half_A) & (w <= center_A - gap_A)
    mR = (w >= center_A + gap_A) & (w <= center_A + half_A)
    L = np.nanmedian(f[mL]) if np.any(mL) else np.nan
    R = np.nanmedian(f[mR]) if np.any(mR) else np.nan
    x = np.concatenate([w[mL], w[mR]])
    y = np.concatenate([f[mL], f[mR]])
    slope = np.nan
    if np.sum(np.isfinite(x) & np.isfinite(y)) >= 3:
        coeff = np.polyfit(x - center_A, y, 1)
        slope = coeff[0]
    return L, R, slope


# ----------------------------------------------------------------------------- #
# Main pipeline
# ----------------------------------------------------------------------------- #
def main() -> None:
    # ── STEP 1 — Config & folders ─────────────────────────────────────────
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default=None, help="Path to YAML config")
    args = p.parse_args()

    cfg_path = args.cfg or os.environ.get("MVT_CFG") or _default_cfg_path()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    night_dir   : str = cfg["night_dir"]
    outputs_dir : str = cfg["outputs_dir"]
    fig_dir     : str = os.path.join(outputs_dir, "figures")
    tab_dir     : str = os.path.join(outputs_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # define debug dict (fixes NameError on dbg.get(...))
    dbg = cfg.get("debug", {})

    # ── STEP 2 — Ephemeris & contacts ─────────────────────────────────────
    eph = Ephemeris(
        T0_bjdtdb   = cfg["ephemeris"]["T0_bjdtdb"],
        period_days = cfg["ephemeris"]["period_days"],
        impact_b    = cfg["ephemeris"]["impact_b"],
        k_rprs      = cfg["ephemeris"]["k_rprs"],
        a_over_rs   = cfg["ephemeris"]["a_over_rs"],
        inc_deg     = cfg["ephemeris"]["inc_deg"],
        Kp_kms      = cfg["ephemeris"]["Kp_kms"],
    )
    contacts = contacts_from_yaml(eph, cfg.get("contacts"))

    # line choice from YAML
    lines_dict = cfg["lines"]
    line_key   = cfg.get("target_line", next(iter(lines_dict)))
    line_cfg   = lines_dict[line_key]
    center_A   = float(line_cfg["center_A"])
    half_A     = float(line_cfg["half_width_A"])
    line_label = line_cfg.get("label", f"{line_key} ({center_A:.3f} Å)")
    INPUT_FRAME = cfg.get("input_frame", "native")
    
    # ── STEP 3 — Discover & read S1D exposures ────────────────────────────
    files = list_s1d_files(night_dir)
    wavelength_A: List[npt.NDArray[np.floating]] = []
    flux:         List[npt.NDArray[np.floating]] = []
    bjd_tdb:      List[float] = []
    berv_kms:     List[float] = []
    rv_star_kms:  List[float] = []
    for path in files:
        w_A, f, _hdr, bjd, berv, rv_star = read_s1d(path)
        wavelength_A.append(w_A); flux.append(f)
        bjd_tdb.append(float(bjd)); berv_kms.append(float(berv)); rv_star_kms.append(float(rv_star))

    bjd_tdb_arr = np.asarray(bjd_tdb, float)
    phases      = phases_from_bjd(bjd_tdb_arr, eph.T0_bjdtdb, eph.period_days)
    in_transit  = in_transit_mask(bjd_tdb_arr, contacts, eph)

    if dbg.get("enable", True):
        from mvt.plots import plot_phase_coverage
        out_pc = os.path.join(
            fig_dir,
            cfg.get("figures", {}).get("phase_coverage_png", "phase_coverage.png")
        )
        plot_phase_coverage(phases, in_transit, contacts, out_pc)

        # Optional: exposures QC table (same as before)
        qc_path = os.path.join(tab_dir, "exposures_qc.csv")
        with open(qc_path, "w") as f:
            f.write("idx,filename,bjd,phase,in_transit,berv_kms,rv_star_kms\n")
            for i, p in enumerate(files):
                f.write(f"{i},{Path(p).name},{bjd_tdb_arr[i]:.6f},{phases[i]:+.6f},"
                        f"{int(in_transit[i])},{berv_kms[i]:.3f},{rv_star_kms[i]:.3f}\n")

    # --- COLOUR MAPS in λ (stellar rest) ---
    fig_opts = cfg.get("figures", {})
    vlim_resid = fig_opts.get("colormap_vlim_resid", "auto")
    vlim_flux  = fig_opts.get("colormap_vlim_flux", "auto")
    pct        = float(fig_opts.get("colormap_percentile", 99.5))
    sym        = bool(fig_opts.get("colormap_symmetric", True))
    cmap_resid = fig_opts.get("colormap_cmap_resid", "RdBu_r")


    # Flux in stellar rest
    lam_sr, F_sr, _ = make_matrix_lambda(
        wavelength_A, flux,
        frame="stellar", quantity="flux",
        center_A=center_A, half_A=half_A,
        bervs=berv_kms, rv_stars=rv_star_kms,
        in_transit=in_transit,
        Npix=cfg.get("figures", {}).get("colormap_Npix", 401),
        input_frame=INPUT_FRAME,
#        subtract_additive_trend=cfg.get("residuals", {}).get("subtract_additive_trend", True),
        trend_gap_kms=cfg.get("residuals", {}).get("sideband_gap_kms", 40.0),
    )
    plot_colormap_lambda(
        lam_sr, F_sr,
        os.path.join(fig_dir, cfg["figures"].get("colormap_stellar_flux_png", "colormap_stellar_flux.png")),
        title=f"Flux colormap (stellar rest) — {line_label}",
        frame="stellar", quantity="flux", center_A=center_A,
        vlim=vlim_flux, pct=pct, symmetric=False, cmap="viridis",
    )

    # Residuals in stellar rest
    lam_sr, R_sr, _ = make_matrix_lambda(
        wavelength_A, flux,
        frame="stellar", quantity="residuals",
        center_A=center_A, half_A=half_A,
        bervs=berv_kms, rv_stars=rv_star_kms,
        in_transit=in_transit,
        Npix=cfg.get("figures", {}).get("colormap_Npix", 401),
        sideband_flatten=True,
        input_frame=INPUT_FRAME,
        subtract_additive_trend=cfg.get("residuals", {}).get("subtract_additive_trend", True),
        trend_gap_kms=cfg.get("residuals", {}).get("sideband_gap_kms", 40.0),
    )
    plot_colormap_lambda(
        lam_sr, R_sr,
        os.path.join(fig_dir, cfg["figures"].get("colormap_stellar_resid_png", "colormap_stellar_resid.png")),
        title=f"Residuals colormap (stellar rest) — {line_label}",
        frame="stellar", quantity="residuals", center_A=center_A,
        show_velocity_axis_top=False,
        vlim=vlim_resid, pct=pct, symmetric=sym, cmap=cmap_resid,
    )

    # ── STEP 4 — Residuals in λ (planet rest) + λ colormap ────────────────
    # (and velocity grid for downstream steps)
    # Audit B: raw windows (use stellar-rest wavelengths for clearer center)
    if dbg.get("enable", True):
        w_stellar = [
            to_stellar_rest(wavelength_A[i], rv_star_kms=rv_star_kms[i], berv_kms=berv_kms[i])
            if (berv_kms is not None and rv_star_kms is not None) else np.asarray(wavelength_A[i], float)
            for i in range(len(wavelength_A))
        ]
        # choose a few exposures
        idx_oot = [i for i, it in enumerate(in_transit) if not it]
        idx_it  = [i for i, it in enumerate(in_transit) if it]
        idxs = (idx_oot[:1] + idx_it[:2] + idx_oot[-1:]) or list(range(min(4, len(files))))
        plot_window_examples(w_stellar, flux, center_A, half_A, idxs,
                             os.path.join(fig_dir, "window_examples.png"))

        # sideband metrics CSV on native flux
        sb_csv = os.path.join(tab_dir, "sideband_metrics.csv")
        with open(sb_csv, "w") as f:
            f.write("idx,bjd,phase,in_transit,med_left,med_right,slope_perA\n")
            for i in range(len(files)):
                L, R, s = _sideband_metrics(wavelength_A[i], flux[i], center_A, half_A)
                f.write(f"{i},{bjd_tdb_arr[i]:.6f},{phases[i]:+.6f},{int(in_transit[i])},"
                        f"{L:.6g},{R:.6g},{s:.6g}\n")

    # Planet-rest residuals on common λ-grid
    lam_pr, R_pr, v_plan = make_matrix_lambda(
        wavelength_A, flux,
        frame="planet", quantity="residuals",
        center_A=center_A, half_A=half_A,
        bervs=berv_kms, rv_stars=rv_star_kms,
        phases=phases, Kp_kms=eph.Kp_kms,
        in_transit=in_transit,
        Npix=cfg.get("figures", {}).get("colormap_Npix", 401),
        sideband_flatten=True,
        input_frame=INPUT_FRAME,
        subtract_additive_trend=cfg.get("residuals", {}).get("subtract_additive_trend", True),
        trend_gap_kms=cfg.get("residuals", {}).get("sideband_gap_kms", 40.0),
    )

    # Planet-rest λ colormap (with velocity tick marks on top)
    plot_colormap_lambda(
        lam_pr, R_pr,
        os.path.join(fig_dir, cfg.get("figures", {}).get("colormap_planet_lambda_png", "colormap_planet_lambda.png")),
        title=f"Residuals colormap (planet rest) — {line_label}",
        frame="planet", quantity="residuals", center_A=center_A,
        show_velocity_axis_top=True,
        vlim=vlim_resid, pct=pct, symmetric=sym, cmap=cmap_resid,
    )

    # Make velocity x-axis for later steps
    velocity_grid_kms = C_KMS * (lam_pr - center_A) / center_A   # 1D
    residual_profiles = [row for row in R_pr]                    # list of 1D

    # ---- Frame QA (guarded) -------------------------------------------------
    if cfg.get("frameqa", {}).get("enable", True):
        try:
            from mvt.frameqa import run_frame_qa
        except Exception as exc:  # import failed
            if dbg.get("enable", True):
                print(f"[frame QA import skipped] {exc!r}")
        else:
            try:
                run_frame_qa(
                    wavelength_A, flux,
                    lam_sr, R_sr,                 # stellar-rest residuals (STEP 3)
                    lam_pr, R_pr,                 # planet-rest residuals (STEP 4)
                    bjd_tdb_arr, phases, in_transit,
                    berv_kms, rv_star_kms,
                    center_A, half_A,
                    cfg, fig_dir, tab_dir,
                    input_frame=INPUT_FRAME       # ← pass your YAML input_frame
                )
            except Exception as exc:  # function raised
                if dbg.get("enable", True):
                    print(f"[frame QA run skipped] {exc!r}")


    # ── AUDIT C — Residuals matrix & NaN map ──────────────────────────────
    if dbg.get("enable", True):
        plot_residuals_matrix(velocity_grid_kms, residual_profiles, in_transit,
                              os.path.join(fig_dir, "residuals_matrix.png"))
        try:
            import matplotlib.pyplot as plt
            isfin = np.asarray([np.isfinite(r) for r in residual_profiles], bool)
            fig = plt.figure(figsize=(7, 4)); ax = fig.add_subplot(111)
            ax.imshow(isfin, aspect="auto", origin="lower",
                      extent=[velocity_grid_kms.min(), velocity_grid_kms.max(), 0, isfin.shape[0]])
            ax.set_xlabel("Velocity [km/s]"); ax.set_ylabel("Exposure index")
            ax.set_title("Finite coverage mask"); fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "nanmask.png"), dpi=160); plt.close(fig)
        except Exception:
            pass
        v_lo, v_hi = (dbg.get("off_line_kms") or [-120.0, -80.0])
        j = (velocity_grid_kms >= v_lo) & (velocity_grid_kms <= v_hi)
        base = [np.nanmedian(r[j]) for r in residual_profiles]
        with open(os.path.join(tab_dir, "residual_baseline.csv"), "w") as f:
            f.write("idx,bjd,phase,in_transit,baseline_offline\n")
            for i, b in enumerate(base):
                f.write(f"{i},{bjd_tdb_arr[i]:.6f},{phases[i]:+.6f},{int(in_transit[i])},{b:.6g}\n")

    # ── STEP 5 — Optional detrend ─────────────────────────────────────────
    residual_profiles_for_stack = residual_profiles
    if cfg.get("detrend", {}).get("enable", False) and cfg.get("actin_csv"):
        from mvt.detrend import (
            read_actin_csv, match_by_time, rlm_detrend,
            depth_proxy_per_exposure, apply_detrend_to_profiles
        )
        act_times_bjd, act_cols = read_actin_csv(cfg["actin_csv"])
        idx = match_by_time(bjd_tdb_arr, act_times_bjd)
        keep = idx >= 0
        predictors = cfg["detrend"].get("predictors", [])
        X = np.column_stack([act_cols[p][idx[keep]] for p in predictors])

        init_sigma_A   = cfg["fit"]["init_sigma_A"]
        init_sigma_kms = C_KMS * init_sigma_A / center_A
        half_window_kms = C_KMS * half_A / center_A

        depth_proxy = depth_proxy_per_exposure(velocity_grid_kms, residual_profiles, half_window_kms)[keep]
        oot = (~in_transit)[keep]

        y_hat, params = rlm_detrend(depth_proxy[oot], X[oot])
        import statsmodels.api as sm
        yhat_all = sm.add_constant(X, has_constant="add") @ params

        yhat_full = np.zeros_like(bjd_tdb_arr); yhat_full[keep] = yhat_all
        residual_profiles_for_stack = apply_detrend_to_profiles(
            velocity_grid_kms, residual_profiles, yhat_full, init_sigma_kms
        )

    # ── STEP 6 — Stack residuals ──────────────────────────────────────────
    stack_matrix = np.vstack(residual_profiles_for_stack)
    valid_cols = np.isfinite(stack_matrix).any(axis=0)

    velocity_grid_kms = velocity_grid_kms[valid_cols]
    residual_profiles_for_stack = [row[valid_cols] for row in residual_profiles_for_stack]
    stack_matrix = np.vstack(residual_profiles_for_stack)

    stack_median = np.nanmedian(stack_matrix, axis=0)
    stack_p16    = np.nanpercentile(stack_matrix, 16, axis=0)
    stack_p84    = np.nanpercentile(stack_matrix, 84, axis=0)

    # OOT vs IT stacks
    if dbg.get("enable", True):
        A = np.vstack(residual_profiles_for_stack)
        A_oot = A[~in_transit]; A_it = A[in_transit]
        def _stack(arr):
            return (np.nanmedian(arr, 0),
                    np.nanpercentile(arr, 16, 0),
                    np.nanpercentile(arr, 84, 0))
        if A_oot.size:
            m, p16, p84 = _stack(A_oot)
            plot_stack(velocity_grid_kms, m, p16, p84,
                       os.path.join(fig_dir, "stack_OOT.png"),
                       title=f"Stacked residual — OOT only ({line_label})")
        if A_it.size:
            m, p16, p84 = _stack(A_it)
            plot_stack(velocity_grid_kms, m, p16, p84,
                       os.path.join(fig_dir, "stack_IT.png"),
                       title=f"Stacked residual — In-transit only ({line_label})")

    # ── STEP 7 — Fit Gaussian to the stacked line ─────────────────────────
    init_sigma_A   = cfg["fit"]["init_sigma_A"]
    init_sigma_kms = C_KMS * init_sigma_A / center_A
    line_fit = fit_gaussian_velocity(velocity_grid_kms, stack_median, init_sigma_kms, center_A=center_A)

    plot_stack(velocity_grid_kms, stack_median, stack_p16, stack_p84,
               os.path.join(fig_dir, cfg.get("figures", {}).get("stack_png", "stack.png")),
               title=f"Stacked residual (planet rest) — {line_label}")

    write_table71(os.path.join(tab_dir, cfg["tables"]["table71_csv"]),
                  {"line": line_label,
                   "depth_percent": line_fit.depth * 100.0,
                   "ew_mA": line_fit.ew_A * 1e3,
                   "v0_kms": line_fit.v0_kms,
                   "sigma_kms": line_fit.sigma_kms,
                   "N_in": int(np.sum(in_transit))})

    # ── STEP 8 — Null tests ───────────────────────────────────────────────
    rng = np.random.default_rng(42)
    null_oot     = null_oot_only(velocity_grid_kms, residual_profiles_for_stack, in_transit)
    null_wrong   = null_wrong_rest_frame(velocity_grid_kms, residual_profiles_for_stack, phases, eph.Kp_kms)
    ctrl_off_A   = cfg.get("null_tests", {}).get("control_offset_A", 2.0)
    ctrl_off_kms = C_KMS * ctrl_off_A / center_A
    null_ctrl    = null_control_window(velocity_grid_kms, residual_profiles_for_stack, +ctrl_off_kms)
    null_shuffle = null_phase_shuffle(velocity_grid_kms, residual_profiles_for_stack, phases, eph.Kp_kms, rng)

    plot_nulls_grid(
        velocity_grid_kms,
        [("OOT-only",        null_oot),
         ("Wrong rest frame", null_wrong),
         ("Control +Δλ",      null_ctrl),
         ("Phase shuffle",    null_shuffle)],
        os.path.join(fig_dir, cfg.get("figures", {}).get("nulls_png", "nulls.png")),
    )


if __name__ == "__main__":
    main()