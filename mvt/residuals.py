from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .timephase import Ephemeris, phases_from_bjd, compute_contacts_auto
from .rv import to_stellar_rest, planet_rv_kms

C_KMS = 299792.458

def frac_residual(flux, oot):
    return np.asarray(flux, float) / np.asarray(oot, float) - 1.0

def cut_window(wave, arr, center_A, half_A):
    m = (wave >= center_A - half_A) & (wave <= center_A + half_A)
    return wave[m], arr[m]

def residual_on_vgrid_with_shift(wave, resid, center_A, v_grid_kms, v_shift_kms):
    v = C_KMS * (np.asarray(wave, float) - float(center_A)) / float(center_A)
    x = v - float(v_shift_kms)
    return np.interp(v_grid_kms, x, resid, left=np.nan, right=np.nan)

def make_residuals_and_vgrid(waves: List[np.ndarray],
                             fluxes: List[np.ndarray],
                             bjds: np.ndarray,
                             ephem: Ephemeris,
                             bervs=None,
                             rv_stars=None,
                             center_A: float = 5889.951,
                             half_width_A: float = 0.30,
                             vmin: float = -150.0,
                             vmax: float = +150.0,
                             dv: float = 0.5,
                             *,
                             contacts=None,
                             it_mask: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Build residuals r=F/F_OOT−1, align to planet rest frame at a velocity grid.

    Returns
    -------
    v_grid : ndarray [km/s]
    resid_v : list of ndarray, one per exposure on v_grid
    """
    bjds = np.asarray(bjds, float)
    # Derive (or accept) in-transit mask in PHASE space (contacts are phases)
    if it_mask is None:
        if contacts is None:
            contacts = compute_contacts_auto(ephem)  # T1..T4 in phase units
        phases = phases_from_bjd(bjds, ephem.T0_bjdtdb, ephem.period_days)
        it_mask = (phases >= contacts.T1) & (phases <= contacts.T4)
    else:
        it_mask = np.asarray(it_mask, bool)

    # Shift each to stellar rest if RV info supplied
    w_rest = []
    for i, w in enumerate(waves):
        if bervs is not None and rv_stars is not None:
            w0 = to_stellar_rest(w, rv_star_kms=rv_stars[i], berv_kms=bervs[i])
        else:
            w0 = np.asarray(w, float)
        w_rest.append(w0)

    # Build OOT master on native grid
    oot_mask = ~it_mask

    F_oot = [f for f, m in zip(fluxes, oot_mask) if m]
    if len(F_oot) == 0:
        n_it = int(np.sum(it_mask)); n_oot = int(np.sum(oot_mask))
        raise ValueError(f"No OOT exposures available (IT={n_it}, OOT={n_oot}). "
                         "Check contact phases / mask input.")

    _valid = [f for f in F_oot if f is not None and np.size(f) > 0 and np.isfinite(f).any()]
    if not _valid:
        raise ValueError("No valid out-of-transit flux arrays (all empty or NaN).")
    oot_master = np.nanmedian(np.vstack(F_oot), axis=0)

    # Residuals per exposure
    resid = [frac_residual(f, oot_master) for f in fluxes]

    # Velocity grid
    v_grid = np.arange(vmin, vmax + dv, dv, dtype=float)

    # Planet velocities per exposure (requires phases)
    phases = phases_from_bjd(bjds, ephem.T0_bjdtdb, ephem.period_days)
    v_p = planet_rv_kms(phases, ephem.Kp_kms)

    # Window + rebin each residual onto v_grid, shifted by v_p
    resid_v = []
    for i in range(len(resid)):
        w_i, r_i = cut_window(w_rest[i], resid[i], center_A, half_width_A)
        rv_i = residual_on_vgrid_with_shift(w_i, r_i, center_A, v_grid, v_p[i])
        resid_v.append(rv_i)
    return v_grid, resid_v