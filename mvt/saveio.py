from __future__ import annotations
import csv, os
import numpy as np

def save_table_4x(per_exposure_rows, path_csv):
    if not per_exposure_rows:
        return
    fieldnames = list(per_exposure_rows[0].keys())
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_exposure_rows:
            w.writerow(r)

def save_table_71(rows, path_csv):
    save_table_4x(rows, path_csv)

def save_stack_npz(path_npz, x_grid, r_stack, lo_band=None, hi_band=None, meta=None):
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)
    if lo_band is None:
        lo_band = r_stack
    if hi_band is None:
        hi_band = r_stack
    if meta is None:
        meta = {}
    np.savez(path_npz, x=x_grid, r=r_stack, lo=lo_band, hi=hi_band, **meta)

# New writers for Session-1 outputs
def write_table71(path_csv: str, row: dict) -> None:
    cols = ["line", "depth_percent", "ew_mA", "v0_kms", "sigma_kms", "N_in"]
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    hdr_exists = os.path.exists(path_csv)
    with open(path_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not hdr_exists:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})

def write_injection_csv(path_csv: str, inj_depths, rec_depths, rec_lo, rec_hi) -> None:
    cols = ["inj_depth_percent", "rec_depth_percent", "rec_lo_percent", "rec_hi_percent"]
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a, b, lo, hi in zip(inj_depths, rec_depths, rec_lo, rec_hi):
            w.writerow({
                "inj_depth_percent": float(a) * 100.0,
                "rec_depth_percent": float(b) * 100.0,
                "rec_lo_percent": float(lo) * 100.0,
                "rec_hi_percent": float(hi) * 100.0,
            })