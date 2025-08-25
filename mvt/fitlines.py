from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from scipy.optimize import curve_fit

C_KMS = 299792.458

@dataclass
class LineFit:
    depth: float
    v0_kms: float
    sigma_kms: float
    ew_A: float

def _gauss(v, A, v0, sig):
    return -abs(A) * np.exp(-0.5 * ((v - v0) / sig) ** 2)

def fit_gaussian_velocity(v_grid: np.ndarray, prof: np.ndarray, init_sigma_kms: float, center_A: float = 5895.924) -> LineFit:
    m = np.isfinite(v_grid) & np.isfinite(prof)
    v = np.asarray(v_grid, float)[m]
    y = np.asarray(prof, float)[m]
    if v.size < 10:
        return LineFit(depth=0.0, v0_kms=0.0, sigma_kms=init_sigma_kms, ew_A=0.0)
    A0 = abs(np.nanmin(y))
    p0 = (-A0, 0.0, init_sigma_kms)
    bounds = ([-10*A0, -20.0, 0.1], [0.0, 20.0, 30.0])
    try:
        popt, _ = curve_fit(_gauss, v, y, p0=p0, bounds=bounds, maxfev=20000)
        A, v0, sig = popt
    except Exception:
        A, v0, sig = p0
    depth = -A
    ew_A = -np.trapz(y, v) * (center_A / C_KMS)  # velocity-integral → Å
    return LineFit(depth=depth, v0_kms=v0, sigma_kms=sig, ew_A=ew_A)

# Backward-compat thin wrapper (keeps your original API available)
def gauss_v(v, a, mu, sig):
    return _gauss(v, -a, mu, sig)

def fit_velocity_profile(v_kms, r, init_sigma_kms=8.0, center_A=5895.924) -> Dict[str, Any]:
    fit = fit_gaussian_velocity(np.asarray(v_kms), np.asarray(r), init_sigma_kms=float(init_sigma_kms), center_A=center_A)
    return dict(depth=fit.depth, v0=fit.v0_kms, ew_A=fit.ew_A, mu=fit.v0_kms, sigma_kms=fit.sigma_kms)