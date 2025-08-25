from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

@dataclass
class Ephemeris:
    T0_bjdtdb: float
    period_days: float
    impact_b: float
    k_rprs: float
    a_over_rs: float
    inc_deg: float
    Kp_kms: float

@dataclass
class Contacts:
    T1: float
    T2: float
    T3: float
    T4: float

def phases_from_bjd(bjd: np.ndarray, T0_bjdtdb: float, period_days: float) -> np.ndarray:
    phi = ((np.asarray(bjd, float) - T0_bjdtdb) / period_days + 0.5) % 1.0 - 0.5
    return phi

def compute_contacts_auto(ephem: Ephemeris) -> Contacts:
    P = ephem.period_days
    aRs = ephem.a_over_rs
    k = ephem.k_rprs
    i = math.radians(ephem.inc_deg)
    b = aRs * math.cos(i)
    if b >= 1.0 + k:
        raise ValueError("No transit: impact parameter >= 1 + k")
    # Winn (2010) circular-orbit formulas
    term14 = math.sqrt(((1.0 + k) ** 2 - b * b) / (aRs * aRs - b * b))
    term23_sq = ((1.0 - k) ** 2 - b * b) / (aRs * aRs - b * b)
    term23 = math.sqrt(term23_sq) if term23_sq > 0 else 0.0
    T14 = (P / math.pi) * math.asin(term14)
    T23 = (P / math.pi) * math.asin(term23) if term23 > 0 else 0.0
    T12 = 0.5 * (T14 - T23)
    T0 = ephem.T0_bjdtdb
    T1 = T0 - 0.5 * T14
    T2 = T1 + T12
    T3 = T2 + T23
    T4 = T1 + T14
    return Contacts(T1=T1, T2=T2, T3=T3, T4=T4)

def contacts_from_yaml(ephem: Ephemeris, cfg_contacts: Optional[Dict]) -> Contacts:
    if not cfg_contacts or cfg_contacts.get("mode", "auto") == "auto":
        base = compute_contacts_auto(ephem)
    else:
        base = compute_contacts_auto(ephem)
    if cfg_contacts:
        for key in ("T1", "T2", "T3", "T4"):
            v = cfg_contacts.get(f"{key}_bjdtdb")
            if v is not None:
                setattr(base, key, float(v))
    if not (base.T1 < base.T2 < base.T3 < base.T4):
        raise ValueError("Contacts are not monotonic T1<T2<T3<T4")
    return base

def in_transit_mask(bjd: np.ndarray, contacts: Contacts) -> np.ndarray:
    bjd = np.asarray(bjd, float)
    return (bjd >= contacts.T1) & (bjd <= contacts.T4)

def kp_radial_velocity(phases: np.ndarray, Kp_kms: float) -> np.ndarray:
    return Kp_kms * np.sin(2.0 * np.pi * phases)