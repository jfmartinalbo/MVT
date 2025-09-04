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
    """
    Devuelve siempre contactos en UNIDADES DE FASE.
    Acepta en YAML:
      - Fase directa:   contacts: {T1: -0.010, T2: -0.004, T3: 0.004, T4: 0.010}
      - En BJD_TDB:     contacts: {T1_bjdtdb: ..., ...}
    Si se dan en BJD, se convierten a fase usando (bjd - T0)/P → (-0.5, 0.5].
    """
    # base (en fase) a partir de la geometría
    base = compute_contacts_auto(ephem)  # devuelve T1..T4 en fase

    if not cfg_contacts:
        return base

    # 1) Overrides en fase si existen
    for key in ("T1", "T2", "T3", "T4"):
        v = cfg_contacts.get(key)
        if v is not None:
            setattr(base, key, float(v))

    # 2) Overrides en BJD si existen → convertir a fase
    for key in ("T1", "T2", "T3", "T4"):
        v_bjd = cfg_contacts.get(f"{key}_bjdtdb")
        if v_bjd is not None:
            phi = float(
                phases_from_bjd(
                    np.array([v_bjd], dtype=float),
                    ephem.T0_bjdtdb,
                    ephem.period_days
                )[0]
            )
            # normaliza a (-0.5, 0.5]
            phi = ((phi + 0.5) % 1.0) - 0.5
            setattr(base, key, phi)

    # 3) Validación (monótonos en fase)
    if not (base.T1 < base.T2 < base.T3 < base.T4):
        raise ValueError(f"Contacts are not monotonic in phase: {base}")

    return base

def in_transit_mask(
    bjd: np.ndarray,
    contacts: Contacts,
    ephem: Optional[Ephemeris] = None,
) -> np.ndarray:
    """
    Return a boolean mask of in-transit exposures.

    - If contacts.T1..T4 look like phases (|value| < ~0.5), we compare in phase.
      In that case we need `ephem` to convert BJD -> phase.
    - Otherwise we assume contacts are in BJD_TDB and compare in time.
    """
    bjd = np.asarray(bjd, float)
    T = np.array([contacts.T1, contacts.T2, contacts.T3, contacts.T4], float)

    # Heuristic: if all |T_i| < 0.5, treat contacts as phases
    if np.all(np.isfinite(T)) and np.all(np.abs(T) < 0.5):
        if ephem is None:
            raise ValueError("in_transit_mask: contacts are in phase; Ephemeris is required.")
        ph = phases_from_bjd(bjd, ephem.T0_bjdtdb, ephem.period_days)
        return (ph >= contacts.T1) & (ph <= contacts.T4)
    else:
        # Treat contacts as BJD_TDB
        return (bjd >= contacts.T1) & (bjd <= contacts.T4)

def kp_radial_velocity(phases: np.ndarray, Kp_kms: float) -> np.ndarray:
    return Kp_kms * np.sin(2.0 * np.pi * phases)