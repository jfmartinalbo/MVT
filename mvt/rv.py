import numpy as np

C_KMS = 299792.458

def shift_wave_by_velocity(wave, v_kms):
    """Doppler shift wavelengths by velocity v (km/s), classical approx.
    λ_rest = λ_obs / (1 + v/c)  (use +v to remove the observed redshift)"""
    v = 0.0 if v_kms is None else float(v_kms)
    return np.asarray(wave, float) / (1.0 + v / C_KMS)

def to_stellar_rest(wave, rv_star_kms=None, berv_kms=None):
    """Shift observed wavelengths to stellar rest frame using RV* + BERV."""
    v = 0.0
    if rv_star_kms is not None:
        v += float(rv_star_kms)
    if berv_kms is not None:
        v += float(berv_kms)
    return shift_wave_by_velocity(wave, v)

def planet_rv_kms(phase, Kp_kms):
    return float(Kp_kms) * np.sin(2.0 * np.pi * np.asarray(phase, float))