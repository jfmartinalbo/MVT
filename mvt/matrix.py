from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .rv import to_stellar_rest
from .timephase import phases_from_bjd
from .residuals import _normalize_by_sidebands_local
from mvt.rv import planet_rv_kms, to_stellar_rest


C_KMS = 299_792.458


def _shift_to_frame(
    lam_native: np.ndarray,
    frame: str,
    *,
    rv_star_kms: float | None = None,
    berv_kms: float | None = None,
    v_planet_kms: float | None = None,
    center_A: float | None = None,
    input_frame: str = "native",   # NEW: tells us what lam_native already is
    relativistic: bool = False,    # optional: exact Doppler if you want
) -> np.ndarray:
    """
    Return wavelengths shifted into the requested frame.

    Frames
    ------
    - "native"      : return input as-is
    - "barycentric" : remove BERV only
    - "stellar"     : remove BERV + stellar RV*
    - "planet"      : start from STELLAR, then subtract planet RV (bring the
                      planetary line to ~v=0 when plotted in velocity space)

    Notes
    -----
    - `input_frame` declares what the *input* wavelengths already are. If you
      pass input_frame="barycentric", BERV will not be removed again.
    - For ESPRESSO S1D this is usually "native".
    - For |v|≲150 km/s the classical formula is fine; set `relativistic=True`
      if you want the exact factor.
    """
    lam = np.asarray(lam_native, float)

    if frame == "native":
        return lam

    # choose Doppler factor
    def _lam_shift(l, v_kms):
        if v_kms is None or v_kms == 0.0:
            return l
        beta = (v_kms / C_KMS)
        if not relativistic:
            # classical: lambda' = lambda / (1 + v/c)
            return l / (1.0 + beta)
        # relativistic (radial): D = sqrt((1+beta)/(1-beta)); lambda' = lambda / D
        D = np.sqrt((1.0 + beta) / (1.0 - beta))
        return l / D

    # Work out how much BERV is still present in the input
    berv_eff = 0.0 if input_frame.lower() == "barycentric" else (berv_kms or 0.0)

    if frame == "barycentric":
        return _lam_shift(lam, berv_eff)

    # Stellar rest = remove (BERV that remains in the input) + stellar RV
    lam_sr = _lam_shift(lam, berv_eff + (rv_star_kms or 0.0))
    if frame == "stellar":
        return lam_sr

    if frame == "planet":
        # planet rest = stellar rest minus planet RV
        return _lam_shift(lam_sr, (v_planet_kms or 0.0))

    # Fallback: nothing
    return lam

def _regular_lambda_grid(center_A: float, half_A: float, Npix: int) -> np.ndarray:
    return np.linspace(center_A - half_A, center_A + half_A, int(Npix), dtype=float)


def _interp_to_grid(x, y, xgrid):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2 or np.all(~np.isfinite(y)):
        return np.full_like(xgrid, np.nan, dtype=float)
    return np.interp(xgrid, x, y, left=np.nan, right=np.nan)

def make_matrix_lambda(
    waves, fluxes, *,
    frame: str,                   # "stellar" | "planet" | "barycentric" | "native"
    quantity: str,                # "flux" | "residuals"
    center_A: float, half_A: float,
    bervs=None, rv_stars=None,
    bjds=None, phases=None, Kp_kms=None,
    in_transit=None,
    Npix: int = 401,
    sideband_flatten: bool = False,
    subtract_additive_trend: bool = True,
    trend_gap_kms: float = 70.0,
    input_frame: str = "native",
):
    """
    Build a [N_exp, Npix] matrix on a *regular wavelength grid* in the requested
    OUTPUT frame; return (lam_grid, M, v_plan) where:

        lam_grid [Npix]      : wavelength grid (Å) in the chosen OUTPUT frame
        M        [N_exp,Npix]: matrix of FLUX or RESIDUALS (dimensionless)
        v_plan   [N_exp] | None: planet RV per exposure (km/s) if frame="planet"

    ─────────────────────────────────────────────────────────────────────────────
    HIGH-LEVEL FLOW (what happens, in which frame)
    ─────────────────────────────────────────────────────────────────────────────
    0) Convert each exposure to the *stellar rest* (this is the base working frame):
         lam_stellar = _to_stellar_from_input(lam_native, input_frame, berv, rv_star)
       Notes on `input_frame` semantics:
         - "native":  lam_native still carries BOTH BERV and RV_* → remove both.
         - "barycentric": lam_native already corrected for BERV → remove RV_* only.
       (ESPRESSO S1D is commonly *barycentric*; use frameguess to confirm.)

    1) Define a *stellar-rest* output grid lam_sr_grid covering [center_A±half_A]
       with Npix points, and interpolate **flux** for each exposure onto this grid.
       At this point we have F_sr [N_exp, Npix] — FLUX in stellar rest.

    2) If `quantity == "flux"`, we can already return (lam_sr_grid, F_sr, v_plan).

    3) For `quantity == "residuals"`:
       3a) Identify OOT rows from `in_transit`. We *require* enough OOT coverage
           per wavelength column; otherwise we TRIM columns with poor OOT support.
           If very little survives, we AUTO-SHRINK the grid to the intersection
           of OOT coverage and re-interpolate (this avoids division by near-zero).
       3b) Build the OOT master spectrum in *stellar rest*:
               oot_sr = median(F_sr[OOT], axis=0)
       3c) Form per-row *ratios* and residuals:
               ratio_sr = F_sr / oot_sr      (≈1 in continuum)
               R_sr     = ratio_sr - 1.0

       Optional cleanups (still in *stellar rest*):
         • sideband_flatten=True  → multiplicative flattening:
             normalize ratio_sr by sideband medians/slopes, then back to residuals
         • subtract_additive_trend=True → additive (linear) detrend of residuals
             fit resid ≈ a*v + b using |v| ≥ trend_gap_kms and subtract (a*v+b)

       IMPORTANT: The OOT master and all cleaning are done **in stellar rest**.
                  This avoids mixing frames while defining the continuum.

    4) Map residuals to the requested OUTPUT frame:
         - frame="stellar"     → return (lam_sr_grid, R_sr)
         - frame="planet"      → return residuals sampled on a *planet-rest*
                                 grid of the same λ coordinates:
               For exposure i, with planet RV v_p[i], evaluate the STELLAR
               residual at λ_src = λ_pr * (1 + v_p/c). This is equivalent to
               shifting the *model grid* rather than resampling the data loop.
         - frame="barycentric" → callers typically re-label; we return the same
                                 stellar-grid residuals (λ grid is stellar).

    Robustness choices:
      • Columns must have at least ~30% of the OOT rows finite (min ≥2). If that
        leaves too few columns, we shrink the grid to the OOT intersection and
        retry with a ≥20% threshold. If still too few columns, we raise.
      • All interpolations are NaN-safe; divisions use the trimmed, finite OOT
        master to avoid blowing up on poorly covered columns.

    Parameters not detailed above:
      phases, Kp_kms  : needed only for frame="planet" (v_plan[i] = Kp*sin(2πφ_i))
      trend_gap_kms   : velocity gap that defines sidebands for additive detrend
    """

    import numpy as np
    C_KMS = 299_792.458

    frame        = (frame or "stellar").lower()
    quantity     = (quantity or "residuals").lower()
    input_frame  = (input_frame or "native").lower()

    waves  = list(waves);   fluxes = list(fluxes)
    N = len(waves)
    if rv_stars is None: rv_stars = [None] * N
    if bervs    is None: bervs    = [None] * N

    # Planet RV curve if we will output in the planet frame
    v_plan = None
    if frame == "planet":
        if phases is None or Kp_kms is None:
            raise ValueError("frame='planet' requires phases and Kp_kms.")
        from .rv import planet_rv_kms
        v_plan = planet_rv_kms(np.asarray(phases, float), float(Kp_kms))

    # Simple wrapper: interpolate one exposure onto a target stellar grid
    def _interp_row(lam_native, f_native, grid_stellar):
        lam_native = np.asarray(lam_native, float)
        f_native   = np.asarray(f_native, float)
        return _interp_to_grid(lam_native, f_native, grid_stellar)

    # ── [STEP 0] Convert each exposure to STELLAR rest & track coverage window
    lam_stellar_rows, f_rows = [], []
    win_min, win_max = [], []
    for i in range(N):
        lam_native = np.asarray(waves[i], float)
        f_i        = np.asarray(fluxes[i], float)

        # from input_frame → stellar rest:
        #  - native → remove (BERV + RV_*)
        #  - barycentric → remove RV_* only

        lam_stellar = _to_stellar_from_input(
            lam_native, input_frame,
            berv_kms=(None if bervs    is None else bervs[i]),
            rv_star_kms=(None if rv_stars is None else rv_stars[i]),
        )
        lam_stellar_rows.append(lam_stellar); f_rows.append(f_i)

        # exposure-specific coverage within the requested window
        m0 = (lam_stellar >= center_A - half_A) & (lam_stellar <= center_A + half_A)
        if np.any(m0):
            win_min.append(float(np.nanmin(lam_stellar[m0])))
            win_max.append(float(np.nanmax(lam_stellar[m0])))
        else:
            win_min.append(np.nan); win_max.append(np.nan)

    # ── [STEP 1] Build base stellar grid over the requested window
    lam_sr_grid = _regular_lambda_grid(center_A, half_A, Npix)

    # Interpolate flux to this grid (stellar rest)
    F_sr = np.vstack([
        _interp_row(
            lam_stellar_rows[i][(lam_stellar_rows[i] >= lam_sr_grid.min()) &
                                (lam_stellar_rows[i] <= lam_sr_grid.max())],
            f_rows[i][(lam_stellar_rows[i] >= lam_sr_grid.min()) &
                      (lam_stellar_rows[i] <= lam_sr_grid.max())],
            lam_sr_grid
        )
        for i in range(N)
    ])  # [N, Npix]

    # If callers only want FLUX, we are done.
    if quantity == "flux":
        return lam_sr_grid, F_sr, v_plan

    # ── [STEP 2] Residuals need OOT master (still in stellar rest)
    if in_transit is None:
        raise ValueError("Residuals require in_transit mask to define OOT.")
    oot = ~np.asarray(in_transit, bool)
    if not np.any(oot):
        raise ValueError("No OOT exposures available.")

    # A) Keep only columns with sufficient OOT coverage
    oot_count_per_col = np.isfinite(F_sr[oot]).sum(axis=0)
    min_oot = max(2, int(0.3 * oot.sum()))  # at least 30% of OOT rows (but ≥2)
    good_col = oot_count_per_col >= min_oot

    # If that leaves too few columns, shrink to the OOT intersection & retry
    if good_col.sum() < max(10, int(0.2 * Npix)):
        oot_idx = np.where(oot)[0]
        mins = [win_min[i] for i in oot_idx if np.isfinite(win_min[i])]
        maxs = [win_max[i] for i in oot_idx if np.isfinite(win_max[i])]
        if len(mins) and len(maxs):
            lamL = float(np.nanmax(mins))
            lamR = float(np.nanmin(maxs))
        else:
            lamL, lamR = np.nan, np.nan

        if not (np.isfinite(lamL) and np.isfinite(lamR) and lamR > lamL):
            raise ValueError(
                "No OOT overlap with requested window. "
                "Check input_frame/BERV/RV* and line center/half_width."
            )

        lam_sr_grid = np.linspace(lamL, lamR, Npix, dtype=float)
        F_sr = np.vstack([
            _interp_row(
                lam_stellar_rows[i][(lam_stellar_rows[i] >= lam_sr_grid.min()) &
                                    (lam_stellar_rows[i] <= lam_sr_grid.max())],
                f_rows[i][(lam_stellar_rows[i] >= lam_sr_grid.min()) &
                          (lam_stellar_rows[i] <= lam_sr_grid.max())],
                lam_sr_grid
            ) for i in range(N)
        ])
        oot_count_per_col = np.isfinite(F_sr[oot]).sum(axis=0)
        min_oot = max(1, int(0.2 * oot.sum()))
        good_col = oot_count_per_col >= min_oot

    if good_col.sum() < 5:
        raise ValueError("Too few λ-columns with OOT coverage after trimming.")

    # Apply the column mask (guarantees finite OOT master)
    lam_sr_grid = lam_sr_grid[good_col]
    F_sr        = F_sr[:, good_col]

    # OOT master and residuals (stellar rest)
    with np.errstate(invalid="ignore"):
        oot_sr = np.nanmedian(F_sr[oot], axis=0)        # finite by construction on kept cols
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_sr = F_sr / oot_sr[None, :]
    R_sr = ratio_sr - 1.0

    # Optional multiplicative flatten → operate on ratio then back to residuals
    if sideband_flatten:
        for i in range(N):
            rpos = ratio_sr[i]
            if np.isfinite(rpos).sum() >= 5:
                rpos2 = _normalize_by_sidebands_local(lam_sr_grid, rpos, center_A, half_A)
                R_sr[i] = rpos2 - 1.0

    # Optional additive linear detrend in sidebands |v| ≥ trend_gap_kms
    if subtract_additive_trend:
        v_sr = C_KMS * (lam_sr_grid - center_A) / center_A
        sb = np.isfinite(v_sr) & (np.abs(v_sr) >= float(trend_gap_kms))
        x  = v_sr[sb]
        if x.size >= 2:
            for i in range(N):
                y = R_sr[i, sb]
                if np.isfinite(y).sum() >= 5:
                    a, b = np.polyfit(x, y, 1)
                    R_sr[i] = R_sr[i] - (a * v_sr + b)

    # ── [STEP 3] Map to the requested OUTPUT frame
    if frame == "stellar":
        return lam_sr_grid, R_sr, v_plan

    if frame == "planet":
        # Keep the same λ grid coordinates but *sample* the stellar residual at
        # a Doppler-shifted λ for each exposure (equivalent to de-shifting rows).
        lam_pr_grid = lam_sr_grid.copy()
        R_pr = np.empty_like(R_sr)
        for i in range(N):
            vp = 0.0 if v_plan is None else float(v_plan[i])
#            lam_src = lam_pr_grid * (1.0 + vp / C_KMS)  # sample the stellar residual at shifted λ
            lam_src = lam_pr_grid * (1.0 - vp / C_KMS)  # sample the stellar residual at shifted λ
            R_pr[i] = np.interp(lam_src, lam_sr_grid, R_sr[i], left=np.nan, right=np.nan)
        return lam_pr_grid, R_pr, v_plan

    if frame == "barycentric":
        # Residuals were constructed in stellar rest; most callers just re-label.
        return lam_sr_grid, R_sr, v_plan
    
# --- Additive trend removal on sidebands (AFTER residuals) ---
def _subtract_additive_trend_in_sidebands(lam_grid, resid_row, center_A, gap_kms=40.0):
    """
    Fit resid ≈ a*v + b in sidebands (|v| >= gap_kms) and subtract it.
    Inputs
      lam_grid : [Npix] λ-grid (already in the chosen frame)
      resid_row: [Npix] residuals (F/F_OOT - 1) for ONE exposure
      center_A : line center [Å]
      gap_kms  : half-gap defining sidebands [km/s]
    """
    C_KMS = 299_792.458
    lam_grid = np.asarray(lam_grid, float)
    r = np.asarray(resid_row, float)

    v_loc = C_KMS * (lam_grid - float(center_A)) / float(center_A)
    sb = np.isfinite(r) & (np.abs(v_loc) >= float(gap_kms))
    if np.count_nonzero(sb) >= 5:
        a, b = np.polyfit(v_loc[sb], r[sb], 1)
        return r - (a * v_loc + b)
    return r


def _to_stellar_from_input(lam_native, input_frame, *, berv_kms=None, rv_star_kms=None):
    lam = np.asarray(lam_native, float)
    f = (input_frame or "native").lower()
    if f in ("stellar", "sr"):
        return lam
    if f in ("barycentric", "bary", "bc"):
        return to_stellar_rest(lam, rv_star_kms=rv_star_kms or 0.0, berv_kms=0.0)
    if f in ("native", "obs", "topocentric"):
        return to_stellar_rest(lam, rv_star_kms=rv_star_kms or 0.0, berv_kms=berv_kms or 0.0)
    raise ValueError(f"Unknown input_frame='{input_frame}'")