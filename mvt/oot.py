import numpy as np

def build_oot_master(waves, fluxes, in_transit_mask):
    """Median OOT master on native pixel grid. Assumes same wavelength sampling."""
    waves = list(waves); fluxes = list(fluxes)
    if len(fluxes) == 0:
        raise ValueError("No exposures")
    oot = [f for f, it in zip(fluxes, in_transit_mask) if not it]
    if len(oot) == 0:
        raise ValueError("No OOT exposures; cannot build master")
    F = np.vstack(oot)
    master = np.nanmedian(F, axis=0)
    return waves[0], master