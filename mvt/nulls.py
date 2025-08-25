from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class StackResult:
    v_grid: np.ndarray
    median: np.ndarray
    p16: np.ndarray
    p84: np.ndarray

def _percentile_stack(v_grid: np.ndarray, profiles: List[np.ndarray]) -> StackResult:
    arr = np.vstack(profiles)
    med = np.nanmedian(arr, axis=0)
    p16 = np.nanpercentile(arr, 16, axis=0)
    p84 = np.nanpercentile(arr, 84, axis=0)
    return StackResult(v_grid=v_grid, median=med, p16=p16, p84=p84)

def _interp_shift(v_grid: np.ndarray, prof: np.ndarray, dv: float) -> np.ndarray:
    return np.interp(v_grid, v_grid - dv, prof, left=np.nan, right=np.nan)

def null_oot_only(v_grid: np.ndarray, resid_v: List[np.ndarray], in_transit: np.ndarray) -> StackResult:
    oot_profiles = [resid_v[i] for i, it in enumerate(in_transit) if not it]
    return _percentile_stack(v_grid, oot_profiles)

def null_wrong_rest_frame(v_grid: np.ndarray, resid_v: List[np.ndarray], phases: np.ndarray, Kp_kms: float) -> StackResult:
    v_p = Kp_kms * np.sin(2.0 * np.pi * phases)
    shifted = [_interp_shift(v_grid, r, +2.0 * vp) for r, vp in zip(resid_v, v_p)]
    return _percentile_stack(v_grid, shifted)

def null_phase_shuffle(v_grid: np.ndarray, resid_v: List[np.ndarray], phases: np.ndarray, Kp_kms: float, rng: np.random.Generator) -> StackResult:
    v_p = Kp_kms * np.sin(2.0 * np.pi * phases)
    shuf = v_p.copy()
    rng.shuffle(shuf)
    shifted = [_interp_shift(v_grid, r, dv_i) for r, dv_i in zip(resid_v, shuf)]
    return _percentile_stack(v_grid, shifted)

def null_control_window(v_grid: np.ndarray, resid_v: List[np.ndarray], offset_kms: float) -> StackResult:
    shifted = [_interp_shift(v_grid, r, +offset_kms) for r in resid_v]
    return _percentile_stack(v_grid, shifted)