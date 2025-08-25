from __future__ import annotations
import csv
from typing import Dict, List, Tuple
import numpy as np
import statsmodels.api as sm

def read_actin_csv(path: str):
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        cols = r.fieldnames or []
    data = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            try:
                data[c].append(float(row[c]))
            except Exception:
                data[c].append(np.nan)
    out = {c: np.asarray(v, float) for c, v in data.items()}
    for key in ("BJD_TDB","BJD","bjd_tdb","bjd"):
        if key in out:
            t = out[key]
            break
    else:
        raise ValueError("ACTIN CSV must include a BJD column")
    return t, out

def match_by_time(expo_bjd: np.ndarray, actin_time: np.ndarray, tol_days: float = 15.0/86400.0) -> np.ndarray:
    idx = np.full(expo_bjd.size, -1, dtype=int)
    for i, t in enumerate(expo_bjd):
        j = int(np.argmin(np.abs(actin_time - t)))
        if abs(actin_time[j] - t) <= tol_days:
            idx[i] = j
    return idx

def rlm_detrend(y: np.ndarray, X: np.ndarray):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.RLM(y, Xc, M=sm.robust.norms.HuberT())
    res = model.fit()
    y_hat = res.predict(Xc)
    return y_hat, res.params

def depth_proxy_per_exposure(v_grid: np.ndarray, resid_v: List[np.ndarray], half_width_kms: float) -> np.ndarray:
    win = np.abs(v_grid) <= half_width_kms
    y = np.array([np.nanmean(r[win]) for r in resid_v], dtype=float)
    return y

def apply_detrend_to_profiles(v_grid: np.ndarray, resid_v: List[np.ndarray], y_hat: np.ndarray, sigma_kms: float) -> List[np.ndarray]:
    g = np.exp(-0.5 * (v_grid / sigma_kms) ** 2)
    out = []
    for r, ypred in zip(resid_v, y_hat):
        out.append(r - float(ypred) * g)
    return out