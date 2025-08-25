# run_mvt_espre.py
# -*- coding: utf-8 -*-
"""
ESPRESSO Na I D₂ Transmission Pipeline — single-night reduction

This script is intentionally written as a *readable pipeline*: each step has a
clear *aim*, states the *problem it solves*, and documents *inputs/outputs* with
units and shapes. Variable names include unit suffixes for fast scanning:

    _A    : Angstroms
    _kms  : Kilometers per second
    _bjd  : BJD_TDB timestamps

High-level flow:
    1) Load configuration & make output folders
    2) Load orbital ephemeris & compute contact times
    3) Discover & read S1D exposures
    4) Build residuals in velocity space around Na I D2
    5) (Optional) Detrend exposure-level systematics
    6) Stack residuals (median) and spread (p16/p84)
    7) Fit Gaussian line to stacked profile
    8) Run null tests (QC)
    9) Injection–recovery (sensitivity curve)
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
from mvt.residuals import make_residuals_and_vgrid
from mvt.fitlines import fit_gaussian_velocity  # canonical fitter (init_sigma_kms positional)
from mvt.plots import plot_stack, plot_nulls_grid, plot_injection_curves
from mvt.saveio import write_table71, write_injection_csv
from mvt.nulls import (
    null_oot_only, null_wrong_rest_frame, null_phase_shuffle, null_control_window
)
from mvt.injection import inject_into_residuals, make_nonneg_yerr
from mvt.stack import robust_nanmedian

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #
C_KMS: float = 299_792.458  # speed of light [km/s]


# ----------------------------------------------------------------------------- #
# Small helpers kept near the pipeline for readability
# ----------------------------------------------------------------------------- #
def _default_cfg_path() -> str:
    """Return default YAML config path within the repo."""
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "configs" / "hd189733_naid.yaml")


def _fit_amp_from_profile(
    v_grid_kms: npt.NDArray[np.floating],
    profile:    npt.NDArray[np.floating],
    window_kms: float,
    init_sigma_kms: float,
    center_kms: float = 0.0,
) -> float:
    """
    Step-local utility: fit a Gaussian within ±window_kms around center_kms and
    return the absolute amplitude |A| (≈ line depth in residual units).

    Inputs
    ------
    v_grid_kms    : [N_v] velocity grid [km/s]
    profile       : [N_v] residual profile (dimensionless; flux ratio – 1)
    window_kms    : half-window around center [km/s]
    init_sigma_kms: initial σ for the fitter [km/s]
    center_kms    : fit center in velocity space [km/s], default 0 in planet RF

    Output
    ------
    |A| (float): absolute amplitude (dimensionless). If fit is ill-posed,
    returns NaN.
    """
    mwin = (v_grid_kms >= center_kms - window_kms) & (v_grid_kms <= center_kms + window_kms)
    if np.count_nonzero(mwin) < 5 or not np.any(np.isfinite(profile[mwin])):
        return np.nan
    # We pass center_A for completeness; EW is not used here, only amplitude.
    fit = fit_gaussian_velocity(v_grid_kms[mwin], profile[mwin], init_sigma_kms, center_A=5895.924)
    return float(abs(getattr(fit, "depth", np.nan)))  # 'depth' is positive by convention


def _bootstrap_amp(
    v_grid_kms: npt.NDArray[np.floating],
    profiles_exp_by_v: npt.NDArray[np.floating],
    window_kms: float,
    init_sigma_kms: float,
    nboot: int = 1000,
    rng: np.random.Generator | int | None = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap the *stacked* amplitude: resample exposures with replacement,
    stack by robust median, fit |A| each time, then return (median, p16, p84).

    Inputs
    ------
    v_grid_kms         : [N_v]
    profiles_exp_by_v  : [N_exp, N_v] residual profiles (dimensionless)
    window_kms         : half-window for Gaussian fit [km/s]
    init_sigma_kms     : initial σ [km/s]
    nboot              : # bootstrap iterations
    rng                : seed or Generator

    Outputs
    -------
    (A_med, A_lo, A_hi): amplitude distribution summary, dimensionless
    """
    rng = np.random.default_rng(rng)
    profiles = np.asarray(profiles_exp_by_v, float)
    n_exp = profiles.shape[0]
    amps: List[float] = []
    for _ in range(int(nboot)):
        idx = rng.integers(0, n_exp, size=n_exp)
        stack = robust_nanmedian(profiles[idx], axis=0, min_valid=3)  # [N_v]
        amps.append(_fit_amp_from_profile(v_grid_kms, stack, window_kms, init_sigma_kms))
    amps_arr = np.asarray(amps, float)
    a_med = float(np.nanmedian(amps_arr))
    a_lo  = float(np.nanpercentile(amps_arr, 16))
    a_hi  = float(np.nanpercentile(amps_arr, 84))
    a_lo, a_med, a_hi = np.sort([a_lo, a_med, a_hi])
    return a_med, a_lo, a_hi


# ----------------------------------------------------------------------------- #
# Main pipeline
# ----------------------------------------------------------------------------- #
def main() -> None:

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 — Load configuration & prepare output folders
    #
    # Aim: Centralize all tunables (paths, line definitions, fitting params).
    # Problem: Reproducibility & portability; we avoid hard-coded values.
    #
    # Inputs:
    #   cfg_path : str (CLI --cfg | $MVT_CFG | default YAML)
    #
    # Outputs:
    #   cfg       : dict with all parameters
    #   fig_dir   : output/figures path
    #   tab_dir   : output/tables  path
    # Notes:
    #   - Folders are created if missing to keep later code simple.
    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 — Load ephemeris & contact times (BJD_TDB)
    #
    # Aim: Place exposures in orbital phase and define in-transit window.
    # Problem: We need precise timing to build planetary rest-frame residuals.
    #
    # Inputs (cfg.ephemeris):
    #   T0_bjdtdb [BJD_TDB], period_days [day], b, Rp/Rs, a/Rs, inc_deg, Kp_kms
    # Outputs:
    #   eph      : Ephemeris object
    #   contacts : (t1,t2,t3,t4) inferred or from YAML
    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 — Discover & read S1D exposures
    #
    # Aim: Load per-exposure spectra & metadata.
    # Problem: Different files, different wavelength sampling per exposure.
    #
    # Outputs (lists of length N_exp):
    #   wavelength_A[i] : [N_λ_i] Angstrom grid
    #   flux[i]         : [N_λ_i] flux (instrumental units)
    #   bjd_tdb[i]      : scalar mid-exposure [BJD_TDB]
    #   berv_kms[i]     : scalar BERV [km/s]
    #   rv_star_kms[i]  : scalar stellar RV [km/s]
    # Pitfalls:
    #   - NaNs/zeros in flux: handled later in residual construction/stack.
    # ──────────────────────────────────────────────────────────────────────
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

    bjd_tdb_arr = np.asarray(bjd_tdb, float)  # [N_exp]
    phases      = phases_from_bjd(bjd_tdb_arr, eph.T0_bjdtdb, eph.period_days)
    in_transit  = in_transit_mask(bjd_tdb_arr, contacts)  # [N_exp] boolean

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 — Build residuals & velocity grid around Na I D2
    #
    # Aim: Express each exposure in *velocity space* around the target line and
    #      in the *planetary rest frame*; then form residuals (flux/oot - 1).
    # Problem: Wavelength grids differ across exposures and the signal is in
    #          velocity; we need a common v-grid and consistent normalization.
    #
    # Inputs:
    #   center_A, half_A : line center and half-width [Å] from cfg.lines.NaID2
    #   wavelength_A, flux, bjd_tdb_arr, eph, berv_kms, rv_star_kms
    # Outputs:
    #   velocity_grid_kms : [N_v] common velocity grid [km/s]
    #   residual_profiles : [N_exp, N_v] dimensionless residuals
    # Pitfalls:
    #   - OOT definition errors → NaNs; handled by nan-aware stats later.
    # ──────────────────────────────────────────────────────────────────────
    center_A : float = cfg["lines"]["NaID2"]["center_A"]
    half_A   : float = cfg["lines"]["NaID2"]["half_width_A"]
    velocity_grid_kms, residual_profiles = make_residuals_and_vgrid(
        wavelength_A, flux, bjd_tdb_arr, eph,
        bervs=berv_kms, rv_stars=rv_star_kms,
        center_A=center_A, half_width_A=half_A,
        contacts=contacts,
        it_mask=in_transit
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 — (Optional) Detrend exposure-level systematics
    #
    # Aim: Remove exposure-to-exposure trends correlated with activity proxies.
    # Problem: Stellar/instrumental systematics can bias the stacked depth.
    #
    # Method:
    #   - Build a depth proxy per exposure by integrating residuals near line.
    #   - Robust linear model (RLM) on OOT exposures vs. chosen predictors.
    #   - Apply model to all exposures as a scaling correction on profiles.
    #
    # Inputs:
    #   cfg.detrend.enable, cfg.actin_csv, predictors list
    # Outputs:
    #   residual_profiles_for_stack : [N_exp, N_v] corrected residuals
    # Pitfalls:
    #   - Mismatched timestamps → we align via nearest-match indices.
    # ──────────────────────────────────────────────────────────────────────
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

        y_hat, params = rlm_detrend(depth_proxy[oot], X[oot])  # fit on OOT only
        import statsmodels.api as sm
        yhat_all = sm.add_constant(X, has_constant="add") @ params

        yhat_full = np.zeros_like(bjd_tdb_arr)
        yhat_full[keep] = yhat_all
        residual_profiles_for_stack = apply_detrend_to_profiles(
            velocity_grid_kms, residual_profiles, yhat_full, init_sigma_kms
        )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6 — Stack residuals (median) and spread (p16/p84)
    #
    # Aim: Increase S/N while remaining robust to outliers and NaNs.
    # Problem: Individual residuals are noisy; median stack stabilizes depth.
    #
    # Inputs:
    #   residual_profiles_for_stack : [N_exp, N_v]
    # Outputs:
    #   stack_median : [N_v], stack_p16 : [N_v], stack_p84 : [N_v]
    # ──────────────────────────────────────────────────────────────────────

    # --- Trim to columns that are covered by at least one exposure ---
    stack_matrix = np.vstack(residual_profiles_for_stack)           # [N_exp, N_v]
    valid_cols = np.isfinite(stack_matrix).any(axis=0)               # [N_v] bool

    velocity_grid_kms = velocity_grid_kms[valid_cols]
    residual_profiles_for_stack = [row[valid_cols] for row in residual_profiles_for_stack]
    stack_matrix = np.vstack(residual_profiles_for_stack)            # recompute on trimmed grid

    stack_matrix = np.vstack(residual_profiles_for_stack)  # [N_exp, N_v]
    stack_median = np.nanmedian(stack_matrix, axis=0)
    stack_p16    = np.nanpercentile(stack_matrix, 16, axis=0)
    stack_p84    = np.nanpercentile(stack_matrix, 84, axis=0)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7 — Fit Gaussian to the stacked line
    #
    # Aim: Quantify the detection: depth, centroid velocity, width, and EW.
    # Problem: We need a compact, comparable summary of the line profile.
    #
    # Inputs:
    #   velocity_grid_kms : [N_v]
    #   stack_median      : [N_v]
    #   init_sigma_kms    : from cfg.fit.init_sigma_A
    # Outputs:
    #   line_fit.depth (dimensionless), line_fit.v0_kms, line_fit.sigma_kms,
    #   line_fit.ew_A [Å]
    # Notes:
    #   - `fit_gaussian_velocity` expects init_sigma_kms as 3rd positional arg.
    # ──────────────────────────────────────────────────────────────────────
    init_sigma_A   = cfg["fit"]["init_sigma_A"]
    init_sigma_kms = C_KMS * init_sigma_A / center_A
    line_fit = fit_gaussian_velocity(velocity_grid_kms, stack_median, init_sigma_kms, center_A=center_A)

    plot_stack(velocity_grid_kms, stack_median, stack_p16, stack_p84,
               os.path.join(fig_dir, cfg["figures"]["stack_png"]))
    write_table71(
        os.path.join(tab_dir, cfg["tables"]["table71_csv"]),
        {
            "line": "Na I D2",
            "depth_percent": line_fit.depth * 100.0,
            "ew_mA": line_fit.ew_A * 1e3,
            "v0_kms": line_fit.v0_kms,
            "sigma_kms": line_fit.sigma_kms,
            "N_in": int(np.sum(in_transit)),
        },
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 8 — Null tests (quality control)
    #
    # Aim: Stress-test the detection against common failure modes.
    # Problem: Spurious signals can arise from OOT normalization, rest-frame
    #          mistakes, or accidental windows; we need to *falsify* them.
    #
    # Tests:
    #   - OOT-only: stack only OOT → should show no in-transit signal.
    #   - Wrong rest frame: scramble the planet RF → line should wash out.
    #   - Control window: shift window by +Δλ → should be consistent with 0.
    #   - Phase shuffle: break phase coherence → signal should vanish.
    #
    # Outputs:
    #   QC figure with the four null profiles.
    # ──────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    null_oot     = null_oot_only(velocity_grid_kms, residual_profiles_for_stack, in_transit)
    null_wrong   = null_wrong_rest_frame(velocity_grid_kms, residual_profiles_for_stack, phases, eph.Kp_kms)
    ctrl_off_A   = cfg.get("null_tests", {}).get("control_offset_A", 2.0)  # +Δλ control window
    ctrl_off_kms = C_KMS * ctrl_off_A / center_A
    null_ctrl    = null_control_window(velocity_grid_kms, residual_profiles_for_stack, +ctrl_off_kms)
    null_shuffle = null_phase_shuffle(velocity_grid_kms, residual_profiles_for_stack, phases, eph.Kp_kms, rng)

    plot_nulls_grid(
        velocity_grid_kms,
        [("OOT-only", null_oot), ("Wrong rest frame", null_wrong),
         ("Control +Δλ", null_ctrl), ("Phase shuffle", null_shuffle)],
        os.path.join(fig_dir, cfg["figures"]["nulls_png"]),
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 9 — Injection–recovery (sensitivity curve)
    #
    # Aim: Calibrate how injected line depths translate to recovered amplitudes
    #      after the *entire* pipeline (stack + fit). This reveals sensitivity
    #      and potential bias (under/over-recovery).
    # Problem: Without this, quoted depths can be misleading due to processing.
    #
    # Procedure:
    #   For each injected depth d:
    #       a) Subtract a small Gaussian kernel from in-transit exposures.
    #       b) Bootstrap-resample exposures → median stack each time.
    #       c) Fit Gaussian amplitude within a fixed window.
    #       d) Normalize recovered amplitude by d → recovery factor.
    #
    # Inputs:
    #   depths_injected : [N_d] dimensionless fractions
    #   residual_profiles_for_stack : [N_exp, N_v]
    #   init_sigma_kms, fit_window_kms
    # Outputs:
    #   CSV + figure of recovery vs injected depth with (p16, p84) bands.
    # Notes:
    #   - yerr must be non-negative for matplotlib; we enforce it via
    #     make_nonneg_yerr before plotting.
    # ──────────────────────────────────────────────────────────────────────
    depths_injected = np.array(
        cfg.get("injection", {}).get("depths", [5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]),
        dtype=float
    )
    rng_inj = np.random.default_rng(cfg.get("injection", {}).get("rng_seed", 42))
    fit_window_kms = cfg["fit"].get("window_kms", C_KMS * (2.0 / center_A))  # default ≈ ±2 Å

    # --- Baseline amplitude (no extra injection) ---
    profiles_arr = np.asarray(residual_profiles_for_stack, float)  # [N_exp, N_v]
    fit_window_kms = cfg["fit"].get("window_kms", C_KMS * (2.0 / center_A))

    A0_med, A0_lo, A0_hi = _bootstrap_amp(
        velocity_grid_kms, profiles_arr, fit_window_kms, init_sigma_kms,
        nboot=int(cfg.get("injection", {}).get("bootstrap_n", 1000)),
        rng=rng_inj
    )

    # --- Build injected profiles per depth on the TRIMMED grid ---
    injected_profile_sets = []
    for d in depths_injected:
        inj = inject_into_residuals(
            residual_profiles_for_stack, velocity_grid_kms, d, init_sigma_kms, in_transit
        )
        injected_profile_sets.append(np.asarray(inj, float))

    # --- Recover delta depth (A_inj - A_base) ---
    rec, rec_lo, rec_hi = [], [], []
    for d, inj_set in zip(depths_injected, injected_profile_sets):
        A_med, A_lo, A_hi = _bootstrap_amp(
            velocity_grid_kms, inj_set, fit_window_kms, init_sigma_kms,
            nboot=int(cfg.get("injection", {}).get("bootstrap_n", 1000)),
            rng=rng_inj
        )
        # conservative bounds: subtract opposite sides of the intervals
        R_med = A_med - A0_med
        R_lo  = A_lo  - A0_hi
        R_hi  = A_hi  - A0_lo

        # ensure ordering
        rlo, r, rhi = np.sort([R_lo, R_med, R_hi])
        rec.append(r); rec_lo.append(rlo); rec_hi.append(rhi)

    rec    = np.asarray(rec, float)       # fraction
    rec_lo = np.asarray(rec_lo, float)    # fraction
    rec_hi = np.asarray(rec_hi, float)    # fraction

    # Plot & save (plots.py already formats in %)
    _ = make_nonneg_yerr(rec, rec_lo, rec_hi)
    plot_injection_curves(
        depths_injected, rec, rec_lo, rec_hi,
        os.path.join(fig_dir, cfg["figures"]["inject_png"])
    )
    write_injection_csv(
        os.path.join(tab_dir, "injection_recovery.csv"),
        depths_injected, rec, rec_lo, rec_hi
    )

if __name__ == "__main__":
    main()