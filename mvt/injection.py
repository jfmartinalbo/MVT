# injection.py
from typing import List
import numpy as np

def _gauss(v: np.ndarray, sigma_kms: float) -> np.ndarray:
    return np.exp(-0.5 * (v / sigma_kms) ** 2)

def inject_into_residuals(resid_v: List[np.ndarray],
                          v_grid: np.ndarray,
                          depth: float,
                          sigma_kms: float,
                          in_transit: np.ndarray) -> np.ndarray:          # ← devuelve ndarray
    kernel = _gauss(v_grid, sigma_kms)
    out = []
    for i, r in enumerate(resid_v):
        r = np.asarray(r, float)                                         # ← asegura float/ndarray
        if in_transit[i]:
            out.append(r - depth * kernel)
        else:
            out.append(r.copy())
    return np.asarray(out, float)                                        # ← matriz [Nexp, Nv]

def bootstrap_stack(v_grid: np.ndarray,
                    profiles: List[np.ndarray],
                    n_boot: int,
                    rng: np.random.Generator):
    n = len(profiles)
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        arr = np.vstack([profiles[j] for j in idx])
        boots.append(np.nanmedian(arr, axis=0))
    boots = np.stack(boots, axis=0)
    lo = np.nanpercentile(boots, 16, axis=0)
    hi = np.nanpercentile(boots, 84, axis=0)
    return lo, hi

def make_nonneg_yerr(y, y_lo, y_hi, tiny=0.0):
    y    = np.asarray(y,    float)
    y_lo = np.asarray(y_lo, float)
    y_hi = np.asarray(y_hi, float)

    # Ordena por si llegan invertidos
    lo = np.minimum(y_lo, y_hi)
    hi = np.maximum(y_lo, y_hi)

    err_lo = y - lo
    err_hi = hi - y

    # Evita negativos (matplotlib lanza ValueError)
    err_lo = np.where(err_lo >= 0, err_lo, tiny)
    err_hi = np.where(err_hi >= 0, err_hi, tiny)

    return np.vstack([err_lo, err_hi])