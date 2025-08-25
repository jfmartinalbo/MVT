#!/usr/bin/env python3
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse, os
import numpy as np

from mvt.config import load_cfg
from mvt.discover import list_s1d_files
from mvt.io_espre import read_s1d
from mvt.rv import to_stellar_rest, planet_rv_kms
from mvt.timephase import tag_phase_and_flag
from mvt.oot import make_oot_master
from mvt.residuals import frac_residual, residual_on_vgrid_with_shift
from mvt.stack import nanmedian_stack, nanpercentile_band
from mvt.fitlines import fit_velocity_profile
from mvt.saveio import save_table_71
from mvt.plots import plot_stack_velocity

def ensure_dirs(base_out):
    for d in ("figures","tables","stacks","logs"):
        os.makedirs(os.path.join(base_out, d), exist_ok=True)

def main():
    p = argparse.ArgumentParser(description="Run MVT on an ESPRESSO S1D night")
    p.add_argument("--cfg", required=True, help="Path to YAML config")
    args = p.parse_args()
    cfg = load_cfg(args.cfg)
    out_base = cfg["outputs_dir"]
    ensure_dirs(out_base)

    files = list_s1d_files(cfg["night_dir"])
    if len(files) == 0:
        print("No S1D files found")
        return

    expos = [read_s1d(str(f)) for f in files]
    expos = sorted(expos, key=lambda e: (np.isfinite(e.bjd_tdb), e.bjd_tdb))

    for e in expos:
        e.wave = to_stellar_rest(e.wave, e.rv_star_kms)
        phi, in_tr, (phi1, phi4) = tag_phase_and_flag(e.bjd_tdb, cfg["ephemeris"])
        e.phase = phi
        e.in_transit = in_tr

    oot_expos = [e for e in expos if not e.in_transit]
    if len(oot_expos) == 0:
        print("No OOT exposures; cannot proceed")
        return

    master = make_oot_master([e.wave for e in oot_expos], [e.flux for e in oot_expos])

    v_grid = np.arange(-150.0, 150.0 + 0.5, 0.5)

    results = []
    for line_key in ("NaID1","NaID2"):
        line = cfg["lines"].get(line_key, None)
        if line is None:
            continue
        center = float(line["center_A"])
        init_sig = float(cfg["fit"].get("init_sigma_A", 0.06))
        Kp = float(cfg["ephemeris"].get("Kp_kms", 0.0))

        resid_arrays = []
        for e in expos:
            if not e.in_transit:
                continue
            r = frac_residual(e.flux, master)
            vshift = planet_rv_kms(e.phase, Kp)
            rv_on_grid = residual_on_vgrid_with_shift(e.wave, r, center, v_grid, vshift)
            resid_arrays.append(rv_on_grid)

        if len(resid_arrays) == 0:
            print("No in-transit residuals for", line_key)
            continue

        stack_r = nanmedian_stack(resid_arrays)
        lo_band, hi_band = nanpercentile_band(resid_arrays, 16, 84)

        met = fit_velocity_profile(v_grid, stack_r, init_sigma_kms=8.0, center_A=center)
        depth_pct = 100.0 * met["depth"]
        ew_mA = 1e3 * met["ew_A"]

        results.append(dict(
            Target=cfg["target"],
            Instr="ESPRESSO",
            Line=line_key,
            Window_A=f"{center:.3f}±{float(line['half_width_A']):.2f}",
            N_in=sum(int(e.in_transit) for e in expos),
            Stack="median",
            Depth_post_pct=f"{depth_pct:.4f}",
            EW_post_mA=f"{ew_mA:.3f}",
            V0_post_kms=f"{met['v0']:.3f}",
            Nulls_passed="NA",
            Detrender_vars="NA"
        ))

        fig_path = os.path.join(out_base, "figures", cfg["figures"]["stack_png"])
        if line_key == "NaID2":
            plot_stack_velocity(v_grid, stack_r, lo=lo_band, hi=hi_band, centers_v=[0.0], out_png=fig_path)

    tbl_path = os.path.join(out_base, "tables", cfg["tables"]["table71_csv"])
    save_table_71(results, tbl_path)
    print("MVT complete")
    print("Table:", tbl_path)
    print("Figure:", os.path.join(out_base, "figures", cfg["figures"]["stack_png"]))

if __name__ == "__main__":
    main()
