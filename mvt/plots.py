from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

from mvt.rv import to_stellar_rest

from matplotlib.colors import TwoSlopeNorm


C_KMS = 299_792.458



def _robust_limits(arr2d, lo=1, hi=99):
    """Percentile-based vmin/vmax for imshow."""
    a = np.asarray(arr2d, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1, +1
    return np.percentile(a, lo), np.percentile(a, hi)


def plot_phase_coverage(phases, in_transit, contacts, out_png, title="Phase coverage"):
    phases = np.asarray(phases, float)
    it = np.asarray(in_transit, bool)

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)

    # Shade transit window
    ax.axvspan(contacts.T1, contacts.T4, color="tab:blue", alpha=0.08, label="Transit")
    # Vertical lines at contacts
    for x, ls in [(contacts.T1, "--"), (contacts.T2, ":"), (contacts.T3, ":"), (contacts.T4, "--")]:
        ax.axvline(x, color="tab:blue", ls=ls, lw=1)

    ax.scatter(phases[~it], np.zeros(np.sum(~it)), s=18, color="0.4", label="OOT", zorder=3)
    ax.scatter(phases[it],   np.zeros(np.sum(it)),   s=30, color="tab:red", label="IT",  zorder=4)

    ax.set_xlabel("Orbital phase")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_window_examples(waves, fluxes, center_A, half_A, idxs, out_png: str, title="Raw window & sidebands"):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs = axs.ravel()
    for ax, i in zip(axs, idxs):
        w = np.asarray(waves[i]); f = np.asarray(fluxes[i])
        m = (w >= center_A - half_A) & (w <= center_A + half_A)
        ax.plot(w[m], f[m], lw=1.0)
        ax.axvline(center_A, ls="--", color="k", alpha=0.4)
        ax.set_title(f"exp #{i}")
    for ax in axs[-2:]: ax.set_xlabel("Wavelength [Å]")
    for ax in axs[::2]: ax.set_ylabel("Flux")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)


def plot_residuals_matrix(v_grid, resid_v, in_transit, out_png: str):
    import matplotlib.pyplot as plt
    v = np.asarray(v_grid, float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for r, it in zip(resid_v, in_transit):
        (ax2 if it else ax1).plot(v, r, color="tab:blue" if not it else "tab:orange",
                                  alpha=0.25, lw=0.6)
    ax1.axvline(0, ls="--", color="k", alpha=0.4); ax2.axvline(0, ls="--", color="k", alpha=0.4)
    ax1.set_title("OOT residuals"); ax2.set_title("In-transit residuals")
    for ax in (ax1, ax2): ax.set_xlabel("Velocity [km/s]"); ax.set_xlim(v.min(), v.max())
    ax1.set_ylabel("Residual F/F_OOT - 1")
    fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)


def _robust_symmetric_ylim(*ys, q=99.0, min_span=5e-4):
    """
    Devuelve un (ymin, ymax) simétrico alrededor de 0 a partir del percentil |y|.
    q: percentil (99 → ignora el 1% más extremo). min_span: límite inferior.
    """
    import numpy as np
    vecs = [np.ravel(y) for y in ys if y is not None]
    if not vecs:
        return (-1.0, 1.0)
    v = np.concatenate(vecs)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    a = float(np.nanpercentile(np.abs(v), q))
    a = max(a, float(min_span))
    return (-a, +a)

def plot_stack(v_grid, median, p16, p84, out_png, title="Stacked residual (planet rest)"):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(v_grid, median, lw=1.5)
    ax.fill_between(v_grid, p16, p84, alpha=0.3, linewidth=0)
    ax.axvline(0.0, ls='--')
    ax.set_xlabel('Velocity [km/s]')
    ax.set_ylabel('Residual F/F_OOT - 1')
    ax.set_title(title)
    # Escala Y robusta y simétrica
    ymin, ymax = _robust_symmetric_ylim(median, p16, p84, q=99.0, min_span=5e-4)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    
def plot_nulls_grid(v_grid, stacks: List[tuple], out_png: str):
    fig = plt.figure(figsize=(9, 7))
    titles = [s[0] for s in stacks]
    # Y-lim común robusto para los 4 paneles
    all_med = [sr.median for _, sr in stacks if hasattr(sr, "median")]
    all_p16 = [sr.p16    for _, sr in stacks if hasattr(sr, "p16")]
    all_p84 = [sr.p84    for _, sr in stacks if hasattr(sr, "p84")]
    ymin, ymax = _robust_symmetric_ylim(*all_med, *all_p16, *all_p84, q=99.0, min_span=5e-4)

    for i, (_, sr) in enumerate(stacks, start=1):
        ax = fig.add_subplot(2, 2, i)
        ax.plot(sr.v_grid, sr.median, lw=1.2)
        ax.fill_between(sr.v_grid, sr.p16, sr.p84, alpha=0.3, linewidth=0)
        ax.axvline(0.0, ls='--')
        ax.set_xlim(v_grid.min(), v_grid.max())
        ax.set_ylim(ymin, ymax)  # misma escala en todos
        ax.set_title(titles[i-1])
        if i in (3,4): ax.set_xlabel('Velocity [km/s]')
        if i in (1,3): ax.set_ylabel('Residual')
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    
def plot_injection_curves(inj_depths, rec_depths, rec_lo, rec_hi, out_png: str,
                          ylim_pct: tuple | None = None, q: float = 95.0):
    """
    Plot recovered vs injected depth (in %), with robust y-limits.

    ylim_pct : (ymin, ymax) in percent to force a fixed scale, e.g. (-0.2, 0.35)
    q        : percentile used for robust auto-limits (95 → ignores top/bottom 5%)
    """
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    inj = np.asarray(inj_depths, float) * 100.0
    rec = np.asarray(rec_depths, float)
    lo  = np.asarray(rec_lo, float)
    hi  = np.asarray(rec_hi, float)

    # Ensure correct ordering of bounds
    lo2 = np.minimum(lo, hi)
    hi2 = np.maximum(lo, hi)

    y       = rec * 100.0
    yerr_lo = np.clip((rec - lo2) * 100.0, 0.0, np.inf)
    yerr_hi = np.clip((hi2 - rec) * 100.0, 0.0, np.inf)

    ax.errorbar(inj, y, yerr=[yerr_lo, yerr_hi], fmt='o', capsize=3)
    ax.plot(inj, inj, ls='--')  # 1:1 line

    ax.set_xlabel('Injected depth [%]')
    ax.set_ylabel('Recovered depth [%]')
    ax.set_title('Injection–Recovery (Na I)')

    # ----- sensible y-axis -----
    if ylim_pct is not None:
        ax.set_ylim(*ylim_pct)
    else:
        vals = np.concatenate([y, lo2 * 100.0, hi2 * 100.0])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            low  = float(np.nanpercentile(vals, 100.0 - q))
            high = float(np.nanpercentile(vals, q))
            # ensure the 1:1 line is fully visible
            high = max(high, float(np.nanmax(inj) * 1.05))
            low  = min(low,  -0.10 * float(np.nanmax(inj)))
            ax.set_ylim(low, high)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    
# Backward-compat wrapper (keeps your previous function name)
def plot_stack_velocity(v_kms, r_stack, lo=None, hi=None, centers_v=None, out_png="fig_7_1_stack.png"):
    plot_stack(v_kms, r_stack, lo if lo is not None else r_stack, hi if hi is not None else r_stack, out_png)
    

def plot_colormap_velocity(v_grid_kms, residuals_by_exp, phases, Kp_kms,
                           out_png, title="Residuals (planet rest)",
                           cmap="viridis", vline0=True):
    """
    Show residual matrix in velocity space (rows=exposures, cols=velocity bins).
    Overlays planet trail v_p(phase)=Kp*sin(2π*phase).
    """
    V = np.asarray(v_grid_kms, float)
    M = np.vstack(residuals_by_exp)  # shape [N_exp, N_v]
    vmin, vmax = _robust_limits(M, 2, 98)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(M, aspect="auto", origin="lower",
                   extent=[V.min(), V.max(), 0, M.shape[0]],
                   vmin=vmin, vmax=vmax, cmap=cmap)

    # Planet trail: one point per exposure (x=v_p, y=index)
    v_p = Kp_kms * np.sin(2*np.pi*np.asarray(phases, float))
#    ax.plot(v_p, np.arange(len(phases)) + 0.5, color="r", lw=1.2, alpha=0.9) --> To be confirmed with professor

    if vline0:
        ax.axvline(0.0, ls="--", lw=1.2, color="#1f77b4", alpha=0.8)

    ax.set_xlabel("Velocity [km/s]")
    ax.set_ylabel("Exposure index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Residual  F/F$_{\\rm OOT}$ − 1")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_colormap_wavelength(waves_rest, fluxes, center_A, half_A,
                             out_png, title="Flux (stellar rest)",
                             Npix=401, continuum_norm=True, cmap="viridis"):
    """
    Build a common λ-grid around the line and image all exposures stacked.
    Optionally normalizes each exposure by sideband median (continuum_norm).
    """
    # Common λ-grid in stellar rest frame
    lam_lo, lam_hi = center_A - half_A, center_A + half_A
    lam_grid = np.linspace(lam_lo, lam_hi, int(Npix))

    # Interpolate each exposure to lam_grid
    imgs = []
    for w, f in zip(waves_rest, fluxes):
        w = np.asarray(w, float); f = np.asarray(f, float)
        m = (w >= lam_lo) & (w <= lam_hi) & np.isfinite(f)
        if np.count_nonzero(m) < 5:
            imgs.append(np.full_like(lam_grid, np.nan))
            continue
        fi = np.interp(lam_grid, w[m], f[m], left=np.nan, right=np.nan)
        if continuum_norm:
            # sidebands: exclude ±(center±30 km/s) around line
            C_KMS = 299792.458
            v_loc = C_KMS*(lam_grid-center_A)/center_A
            sb = np.abs(v_loc) >= 30.0
            cont = np.nanmedian(fi[sb]) if np.any(sb & np.isfinite(fi)) else 1.0
            if not np.isfinite(cont) or cont == 0:
                cont = 1.0
            fi = fi/cont
        imgs.append(fi)

    M = np.vstack(imgs)  # [N_exp, Npix]
    vmin, vmax = _robust_limits(M, 1, 99)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(M, aspect="auto", origin="lower",
                   extent=[lam_grid.min(), lam_grid.max(), 0, M.shape[0]],
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.axvline(center_A, ls="--", color="0.6")
    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Exposure index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Flux (normalized)" if continuum_norm else "Flux")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    
    
def _sideband_mask_lambda(w, center_A, half_A, gap_kms=30.0):
    """Sidebands: exclude ±gap_kms around the line center."""
    gap_A = center_A * (gap_kms / C_KMS)
    mL = (w >= center_A - half_A) & (w <= center_A - gap_A)
    mR = (w >= center_A + gap_A) & (w <= center_A + half_A)
    return mL | mR


# --------------------------- utilities --------------------------------
def _autoscale_symmetric(M, pct_lo=5, pct_hi=95):
    """Robust symmetric color limits from percentiles of finite values."""
    finite = np.isfinite(M)
    if not np.any(finite):
        return -1.0, +1.0
    lo, hi = np.nanpercentile(M[finite], [pct_lo, pct_hi])
    span = max(abs(lo), abs(hi), 1e-12)
    return -span, +span


# -------------------- STELLAR-REST: FLUX vs λ -------------------------
def make_colormap_stellar_rest_matrix(
    waves, fluxes, *, center_A, half_A,
    bervs=None, rv_stars=None, Npix=401, continuum_norm=True
):
    """
    Build a wavelength grid and a 2D matrix of flux in the *stellar rest* frame.

    Inputs
    ------
    waves, fluxes    : lists of 1D arrays per exposure (same lengths)
    center_A, half_A : line window [Å]
    bervs, rv_stars  : per-exposure BERV and stellar RV [km/s] (optional but recommended)
    Npix             : number of wavelength pixels in the common grid
    continuum_norm   : if True, normalize each row by sidebands

    Returns
    -------
    lam_grid : [Npix] common wavelength grid (stellar rest) [Å]
    M        : [N_exp, Npix] flux matrix (normalized if continuum_norm)
    """
    waves = list(waves); fluxes = list(fluxes)

    # 1) Shift each exposure to stellar rest (remove observer BERV & systemic RV)
    from mvt.rv import to_stellar_rest
    waves_rest = []
    for i, w in enumerate(waves):
        if (bervs is not None) and (rv_stars is not None):
            w0 = to_stellar_rest(w, rv_star_kms=float(rv_stars[i]), berv_kms=float(bervs[i]))
        else:
            w0 = np.asarray(w, float)
        waves_rest.append(w0)

    # 2) Common wavelength grid around the line
    lam_lo = center_A - half_A
    lam_hi = center_A + half_A
    lam_grid = np.linspace(lam_lo, lam_hi, int(Npix), dtype=float)

    # 3) Interpolate each exposure onto lam_grid
    M = []
    for w, f in zip(waves_rest, fluxes):
        # window, then interp
        m = (w >= lam_lo) & (w <= lam_hi)
        if not np.any(m):
            M.append(np.full_like(lam_grid, np.nan))
            continue
        fi = np.interp(lam_grid, w[m], np.asarray(f, float)[m], left=np.nan, right=np.nan)

        if continuum_norm:
            # sidebands: |v| >= 30 km/s relative to center_A
            v_loc = C_KMS * (lam_grid - center_A) / center_A
            sb = np.abs(v_loc) >= 30.0
            med = np.nanmedian(fi[sb]) if np.any(sb & np.isfinite(fi)) else np.nanmedian(fi)
            if np.isfinite(med) and med != 0.0:
                fi = fi / med
        M.append(fi)

    return lam_grid, np.vstack(M)


def plot_colormap_stellar_rest(
    lam_grid, M, out_png, *, title="Flux colormap (stellar rest)",
    center_A=None, vmin=None, vmax=None
):
    """Plot stellar-rest 2D *flux* image vs wavelength (Å)."""
    if vmin is None or vmax is None:
        vmin, vmax = np.nanmin(M), np.nanmax(M)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    extent = [float(lam_grid.min()), float(lam_grid.max()), 0, M.shape[0]]
    im = ax.imshow(M, aspect="auto", origin="lower", extent=extent, cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel("Wavelength [Å]  (stellar rest)")
    ax.set_ylabel("Exposure index")
    if center_A is not None:
        ax.axvline(center_A, color="w", ls="--", lw=1.8, alpha=0.7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Flux" + (" (normalized)" if 0.7 < np.nanmedian(M) < 1.3 else ""))
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# --------------- PLANET-REST: RESIDUALS vs λ --------------------------
def plot_colormap_planet_rest_lambda(
    v_grid_kms, residuals_exp_by_v, *, center_A, out_png,
    title="Residuals colormap (planet rest, λ)",
    vmin=None, vmax=None, pct=(5, 95), show_velocity_axis_top=False
):
    """
    Plot planet-rest *residuals* vs wavelength. The input residuals are already
    on a common velocity grid in the planet rest frame (output of STEP 4).

    x-axis transform: λ = λ0 * (1 + v/c)
    """
    M = np.vstack(residuals_exp_by_v).astype(float)
    lam_grid = center_A * (1.0 + np.asarray(v_grid_kms, float)/C_KMS)

    # Robust symmetric colour range if not provided
    if vmin is None or vmax is None:
        vmin, vmax = _autoscale_symmetric(M, pct_lo=pct[0], pct_hi=pct[1])

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    extent = [float(lam_grid.min()), float(lam_grid.max()), 0, M.shape[0]]
    im = ax.imshow(M, aspect="auto", origin="lower", extent=extent, cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel("Wavelength [Å]  (planet rest)")
    ax.set_ylabel("Exposure index")
    ax.axvline(center_A, color="w", ls="--", lw=1.8, alpha=0.7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Residual  F/F$_{\\rm OOT}$ − 1")

    if show_velocity_axis_top:
        # optional twin top axis to show velocity ticks for reference
        C = C_KMS; lam0 = center_A
        lam_to_v = lambda lam: C * (lam/lam0 - 1.0)
        v_to_lam = lambda v: lam0 * (1.0 + v/C)
        secax = ax.secondary_xaxis('top', functions=(lam_to_v, v_to_lam))
        secax.set_xlabel("Velocity [km/s]")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
 
    
 # --------------- STELLAR-REST: RESIDUALS vs λ --------------------------

def make_colormap_stellar_rest_residuals_matrix(
    waves, fluxes, *, in_transit, center_A, half_A,
    bervs=None, rv_stars=None, Npix=401, sideband_flatten=True
):
    """
    Build a 2D matrix of *residuals* in the STELLAR rest frame, on a common λ-grid.

    Steps
    -----
    1) Shift each exposure to stellar rest using (BERV, RV_*).
    2) Interpolate onto λ-grid around [center_A ± half_A].
    3) Build OOT master from OOT rows only.
    4) Residuals = F / F_OOT - 1.
    5) (Optional) subtract per-row sideband median (|v|>=30 km/s) to flatten offsets.

    Returns
    -------
    lam_grid : [Npix] wavelength grid (Å, stellar rest)
    R        : [N_exp, Npix] residual matrix (dimensionless)
    """
    from mvt.rv import to_stellar_rest

    lam_lo = float(center_A - half_A)
    lam_hi = float(center_A + half_A)
    lam_grid = np.linspace(lam_lo, lam_hi, int(Npix), dtype=float)

    rows = []
    for i, (w, f) in enumerate(zip(waves, fluxes)):
        w = np.asarray(w, float); f = np.asarray(f, float)

        # Shift to stellar rest (remove observer's motion + systemic RV)
        if bervs is not None and rv_stars is not None:
            w_rest = to_stellar_rest(w, rv_star_kms=float(rv_stars[i]), berv_kms=float(bervs[i]))
        else:
            w_rest = w

        # Interpolate to common λ-grid within the window
        m = (w_rest >= lam_lo) & (w_rest <= lam_hi)
        if not np.any(m):
            rows.append(np.full_like(lam_grid, np.nan))
            continue
        fi = np.interp(lam_grid, w_rest[m], f[m], left=np.nan, right=np.nan)
        rows.append(fi)

    F = np.vstack(rows)  # [N_exp, Npix]

    oot = ~np.asarray(in_transit, bool)
    if not np.any(oot):
        raise ValueError("make_colormap_stellar_rest_residuals_matrix: no OOT rows available.")

    # OOT master on the same λ-grid
    F_oot = F[oot]
    if not np.isfinite(F_oot).any():
        raise ValueError("All OOT flux rows are NaN in the stellar-rest grid.")
    oot_master = np.nanmedian(F_oot, axis=0)  # [Npix]
    # Guard against zeros
    good = np.isfinite(oot_master) & (oot_master != 0)
    if not np.any(good):
        raise ValueError("OOT master is invalid on the stellar-rest grid.")
    # Safe divide (broadcast)
    R = F / oot_master - 1.0

    if sideband_flatten:
        # Subtract per-row median over sidebands (|v|>=30 km/s) to zero the continuum
        v_loc = C_KMS * (lam_grid - center_A) / center_A
        sb = np.abs(v_loc) >= 30.0
        if np.any(sb):
            med = np.nanmedian(R[:, sb], axis=1)  # [N_exp]
            R = R - med[:, None]

    return lam_grid, R


def plot_colormap_stellar_rest_residuals(
    lam_grid, R, out_png, *,
    title="Residuals colormap (stellar rest)",
    center_A=None, vmin=None, vmax=None, pct=(5, 95)
):
    """Plot STELLAR-rest 2D *residuals* image vs wavelength (Å)."""
    if vmin is None or vmax is None:
        vmin, vmax = _autoscale_symmetric(R, pct_lo=pct[0], pct_hi=pct[1])

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    extent = [float(lam_grid.min()), float(lam_grid.max()), 0, R.shape[0]]
    im = ax.imshow(R, aspect="auto", origin="lower", extent=extent, cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel("Wavelength [Å]  (stellar rest)")
    ax.set_ylabel("Exposure index")
    if center_A is not None:
        ax.axvline(center_A, color="w", ls="--", lw=1.8, alpha=0.7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Residual  F/F$_{\\rm OOT}$ − 1")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    
# --- unified λ-space colormap plotter ---

def _robust_vlim(M: np.ndarray, pct: float = 99.5, symmetric: bool = True) -> tuple[float, float]:
    """
    Robust color limits for residual matrices.
    - pct: percentile of |values| if symmetric, else upper/lower percentiles.
    """
    vals = np.asarray(M, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-0.01, 0.01)  # safe default
    if symmetric:
        a = float(np.nanpercentile(np.abs(vals), pct))
        a = max(a, 1e-4)  # avoid zero range
        return (-a, +a)
    else:
        lo = float(np.nanpercentile(vals, (100.0 - pct) / 2.0))
        hi = float(np.nanpercentile(vals, 100.0 - (100.0 - pct) / 2.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return (-0.01, 0.01)
        return (lo, hi)

def plot_colormap_lambda(
    lam_grid_A: np.ndarray,              # [Npix]
    M: np.ndarray,                       # [N_exp, Npix]
    out_png: str,
    *,
    title: str,
    frame: str,                          # "stellar" | "planet"
    quantity: str,                       # "flux" | "residuals"
    center_A: float,
    show_velocity_axis_top: bool = False,
    # NEW knobs
    vlim: str | tuple[float, float] = "auto",   # "auto" or (vmin, vmax)
    pct: float = 99.5,                          # robust percentile when auto
    symmetric: bool = True,                     # center the scale on 0 for residuals
    cmap: str | None = None,                    # default diverging for residuals
):
    lam = np.asarray(lam_grid_A, float)
    A   = np.asarray(M, float)

    # Decide color limits
    if isinstance(vlim, (list, tuple)) and len(vlim) == 2:
        vmin, vmax = float(vlim[0]), float(vlim[1])
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if (symmetric and quantity == "residuals") else None
    elif vlim == "auto":
        vmin, vmax = _robust_vlim(A, pct=pct, symmetric=(symmetric and quantity == "residuals"))
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if (symmetric and quantity == "residuals") else None
    else:
        vmin = vmax = None
        norm = TwoSlopeNorm(vcenter=0.0) if (symmetric and quantity == "residuals") else None

    # Pick a sensible colormap
    if cmap is None:
        cmap = "RdBu_r" if quantity == "residuals" else "viridis"

    # Plot
    fig = plt.figure(figsize=(12, 3.6))
    ax = fig.add_subplot(111)

    # extent maps [x_min, x_max, y_min, y_max] to imshow coordinates
    extent = [lam.min(), lam.max(), 0, A.shape[0]]
    im = ax.imshow(A, aspect="auto", origin="lower",
                   extent=extent, cmap=cmap,
                   vmin=None if norm else vmin, vmax=None if norm else vmax,
                   norm=norm)

    ax.set_xlabel("Wavelength [Å]")
    ax.set_ylabel("Exposure index")
    ax.set_title(f"{title} — {frame} rest, {quantity}")

    # Vertical dashed line at the line center
    ax.axvline(center_A, ls="--", color="w", alpha=0.6)

    # Optional velocity ticks on top
    if show_velocity_axis_top:
        c = 299_792.458
        v = c * (lam - float(center_A)) / float(center_A)
        ax2 = ax.twiny()
        ax2.set_xlim(v.min(), v.max())
        ax2.set_xlabel("Velocity [km/s]")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    label = "Residuals (F/F_OOT − 1)" if quantity == "residuals" else "Flux"
    cbar.set_label(label)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)