#!/usr/bin/env python3
"""
Synthetic night generator for MVT dry-run validation.

Creates N S1D-like FITS spectra with:
- Flat continuum F=1.0 with white noise.
- In-transit exposures include a Gaussian absorption line at Na I D2
  Doppler-shifted by the planet radial velocity.
- Minimal headers used by the pipeline: BJD, BERV, RV_STAR.

Outputs to: data/synth_night/*.fits
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from astropy.io import fits
from datetime import datetime, timedelta

# ---- configurable knobs (keep in sync with your YAML) -----------------------
N_EXP              = 40
N_IN_TRANSIT       = 20                 # first 10 OOT, 20 in-transit, last 10 OOT
CENTER_A           = 5889.951           # Na I D2 (air) — override in cfg if needed
WINDOW_A           = 3.0                # +/- Å window written to file
DLAM_A             = 0.01               # Å step
Kp_KMS             = 154.0              # planet Kp for the RV curve
SIGMA_KMS          = 8.0                # intrinsic line σ in km/s
DEPTH_FRAC         = 1.0e-3             # true depth = 0.10%
WHITE_NOISE_FRAC   = 5.0e-4             # per-pixel white noise
T0                 = 2457000.0          # arbitrary epoch
P_DAYS             = 2.218575           # period (HD 189733 b)
C_KMS              = 299792.458

OUTDIR = Path("data/synth_night")

START_ISO   = "2021-08-11T00:53:32"

CADENCE_S   = 360 

# -----------------------------------------------------------------------------

def rv_planet_kms(phase: np.ndarray, Kp_kms: float) -> np.ndarray:
    """Simple sinusoidal planet RV as a function of orbital phase."""
    return Kp_kms * np.sin(2 * np.pi * phase)

def gauss_profile(v_kms: np.ndarray, sigma_kms: float) -> np.ndarray:
    return np.exp(-0.5 * (v_kms / sigma_kms) ** 2)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # wavelength grid around the line
    lam = np.arange(CENTER_A - WINDOW_A, CENTER_A + WINDOW_A + DLAM_A, DLAM_A)
    nlam = lam.size

    # make times & phases (spread over ~one transit)
    bjd = T0 + np.linspace(-0.02, 0.02, N_EXP) * P_DAYS
    phase = ((bjd - T0) / P_DAYS) % 1.0

    # choose which indices are in transit (middle block)
    it_mask = np.zeros(N_EXP, dtype=bool)
    it_mask[10:10 + N_IN_TRANSIT] = True

    # constant barycentric and stellar RV for synthetic data
    berv = np.zeros(N_EXP)
    rv_star = np.zeros(N_EXP)

    # precompute velocity grid relative to CENTER_A
    # for a shifted line: v = c * (lambda/lambda0 - 1)
    v_from_lam = C_KMS * (lam / CENTER_A - 1.0)

    rng = np.random.default_rng(42)
    
    start_dt = datetime.fromisoformat(START_ISO)

    for i in range(N_EXP):
        # start with flat continuum
        flux = np.ones(nlam, dtype=float)

        if it_mask[i]:
            # planet rv for this exposure
            vp = rv_planet_kms(phase[i], Kp_KMS)
            # absorption line centered at vp in velocity space
            line = gauss_profile(v_from_lam - vp, SIGMA_KMS)
            flux -= DEPTH_FRAC * line

        # add white noise
        flux += rng.normal(0.0, WHITE_NOISE_FRAC, size=nlam)

        # write minimal S1D-like FITS
        hdu = fits.PrimaryHDU(flux.astype("f4"))
        h = hdu.header
        h["BJD_TDB"] = float(bjd[i])
        h["BJD"] = float(bjd[i])
        h["BERV"] = float(berv[i])
        h["RV_STAR"] = float(rv_star[i])
        h["CRVAL1"] = lam[0]
        h["CDELT1"] = DLAM_A
#        h["NAXIS1"] = nlam
        h["CUNIT1"] = "Angstrom"
        h["OBJECT"] = "HD189733"
        h["ITRANSIT"] = int(it_mask[i])

                # store the wavelength array in extension 1 for convenience
        hdu_lam = fits.ImageHDU(lam.astype("f8"), name="LAMBDA")

        hdul = fits.HDUList([hdu, hdu_lam])
        ts  = (start_dt + timedelta(seconds=i * CADENCE_S)).strftime("%Y-%m-%dT%H:%M:%S")
        out = OUTDIR / f"ESPRE.{ts}_HD189733_S1D_cpl-7.1.2_DRS.fits"
        hdul.writeto(out, overwrite=True)

    print(f"Wrote {N_EXP} synthetic S1D files to {OUTDIR}/")

if __name__ == "__main__":
    main()
