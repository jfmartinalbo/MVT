from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .timephase import Ephemeris, phases_from_bjd, compute_contacts_auto
from .rv import to_stellar_rest, planet_rv_kms

# Velocidad de la luz en km/s (aprox. exacta para nuestras necesidades)
C_KMS = 299792.458


# -------------------------------------------------------------------------
# Utilidades básicas
# -------------------------------------------------------------------------

def frac_residual(flux, oot):
    """
    Devuelve el **residual fraccionario** r = F / F_OOT - 1.

    Parámetros
    ----------
    flux : array-like
        Flujo de una exposición (unidimensional).
    oot : array-like
        "Master" out-of-transit con la MISMA rejilla nativa (mismo tamaño).

    Notas importantes
    -----------------
    - Este cociente asume que `flux` y `oot` están muestreados en la misma
      rejilla de longitudes de onda (mismo índice i → misma λ_i).
      Si tus S1D varían la rejilla entre exposiciones, hay que **re-muestrear
      previamente** para construir `oot`.
    """
    return np.asarray(flux, float) / np.asarray(oot, float) - 1.0

def cut_window(wave, arr, center_A, half_A):
    """
    Recorta `arr` (flux/residual) y `wave` a la ventana [center_A ± half_A].

    Devolvemos la pareja (wave_recortada, arr_recortada) para operar solo
    alrededor de la línea de interés.
    """
    m = (wave >= center_A - half_A) & (wave <= center_A + half_A)
    return wave[m], arr[m]

# -------------------------------------------------------------------------
# Corrección de continuo (por bandas laterales) — helpers locales
# -------------------------------------------------------------------------
def _sideband_mask_local(w, center_A, half_A, gap_kms=30.0):
    """
    Construye una máscara booleana para seleccionar **bandas laterales**:
    regiones alejadas del centro de línea más allá de ±gap_kms, pero aún
    dentro de la ventana [center_A ± half_A].

    gap_kms se traduce a Δλ mediante la aproximación Doppler clásica:
    Δλ = λ0 * (gap_kms / c).
    """
    gap_A = center_A * (gap_kms / C_KMS)
    mL = (w >= center_A - half_A) & (w <= center_A - gap_A)
    mR = (w >= center_A + gap_A) & (w <= center_A + half_A)
    return mL | mR

def _normalize_by_sidebands_local(w, r_raw, center_A, half_A):
    """
    Normaliza una **curva de cocientes** f/oot (aún NO centrada en 1) usando
    las bandas laterales de ESA MISMA exposición y rejilla.

    Estrategia:
      1) Ajuste lineal en bandas laterales: y ≈ intercept + slope*(w-centerA)
         → `cont(w)` como estimación de continuo multiplicativo residual.
      2) Dividimos r_raw por `cont(w)` para aplanar el continuo.
      3) Fallback: si no hay suficientes puntos, usar mediana de sidebands.

    Entradas
    --------
    w : ndarray
        Wavelength en la rejilla (idealmente ya en reposo estelar).
    r_raw : ndarray
        Cociente f_i / oot_master en la rejilla de la exposición (≥ 0).
        OJO: aquí es el cociente **positivo**, no (f/oot - 1).
    center_A, half_A : float
        Definen la ventana de línea y por tanto las bandas laterales.

    Salida
    ------
    r_norm : ndarray
        r_raw / cont(w) → todavía en "cociente" (≈1 en continuo).
        El “residual final” será r_norm - 1.
    """
    # Selección de sidebands y limpieza de no-finitos
    m = _sideband_mask_local(w, center_A, half_A) & np.isfinite(r_raw)
    if np.count_nonzero(m) >= 5:
        # Ajuste lineal sobre w-center_A para estabilidad numérica
        x = w[m] - center_A
        y = r_raw[m]
        slope, intercept = np.polyfit(x, y, 1)   # y ≈ intercept + slope*x

        # Continuo multiplicativo estimado y topes de seguridad
        cont = intercept + slope * (w - center_A)
        cont = np.clip(cont, 1e-3, 1e3)          # evita divisiones raras
        return r_raw / cont

    # Fallback: scale by sideband median if not enough points
    med = np.nanmedian(r_raw[m]) if np.any(m) else 1.0
    return r_raw / med

def residual_on_vgrid_with_shift(wave, resid, center_A, v_grid_kms, v_shift_kms):
    """
    Interpola un residual (en función de λ) a una rejilla de velocidades
    `v_grid_kms`, previa conversión λ→v y **tras restar** el corrimiento
    `v_shift_kms` (p.ej. velocidad planetaria para llevar la línea a v≈0).

    Notas
    -----
    - La conversión usa Doppler clásico (válido para |v|≪c):
        v = c * (λ - λ0)/λ0
    - Usamos np.interp con `left/right=np.nan`: fuera del rango se devolverán
      NaNs y luego los procedimientos de “stack” usarán nanmedian/nanpercentile.
    """
    """
    Interpola un residual (en función de λ) a una rejilla de velocidades
    `v_grid_kms`, previa conversión λ→v y **tras restar** el corrimiento
    `v_shift_kms` (p.ej. velocidad planetaria para llevar la línea a v≈0).

    Notas
    -----
    - La conversión usa Doppler clásico (válido para |v|≪c):
        v = c * (λ - λ0)/λ0
    - Usamos np.interp con `left/right=np.nan`: fuera del rango se devolverán
      NaNs y luego los procedimientos de “stack” usarán nanmedian/nanpercentile.
    """


    v = C_KMS * (np.asarray(wave, float) - float(center_A)) / float(center_A)
    x = v - float(v_shift_kms)
    return np.interp(v_grid_kms, x, resid, left=np.nan, right=np.nan)

# -------------------------------------------------------------------------
# Función principal: construir residuos sobre rejilla de velocidad
# -------------------------------------------------------------------------
def make_residuals_and_vgrid(waves: List[np.ndarray],
                             fluxes: List[np.ndarray],
                             bjds: np.ndarray,
                             ephem: Ephemeris,
                             bervs=None,
                             rv_stars=None,
                             center_A: float = 5889.951,
                             half_width_A: float = 0.30,
                             vmin: float = -150.0,
                             vmax: float = +150.0,
                             dv: float = 0.5,
                             *,
                             contacts=None,
                             it_mask: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Construye perfiles residuos alineados en la **rejilla de velocidad**
    del marco planetario y devuelve también esa rejilla común.

    Pasos (resumen)
    ---------------
    1) Determinar exposiciones in-transit (mask) **en espacio de fase**.
    2) Pasar cada rejilla λ[i] a **reposo estelar** si hay (BERV, RV_*).
    3) Construir OOT master en la rejilla nativa (suponiendo misma rejilla).
    4) Residuales por exposición:
         r_raw = f_i/oot_master  → normalizar continuo con sidebands →
         residual final = r_norm - 1.
    5) Definir rejilla de velocidades v_grid = [vmin:dv:vmax].
    6) Para cada exposición:
         recortar ventana [λ0 ± Δλ] → pasar a v_local → restar v_planeta
         → interpolar a v_grid → residual en la rejilla común.
    7) Devolver (v_grid, lista_de_perfiles_residuales).

    Unidades/convenciones
    ---------------------
    - `waves[i]` y `fluxes[i]` deben ser 1D y **coherentes entre sí**.
    - `center_A` y `half_width_A` en **Ångström**.
    - `vmin`, `vmax`, `dv` en **km/s**.
    - La salida son perfiles **adimensionales** (F/F_OOT - 1) en v_grid.

    Supuestos y precauciones
    ------------------------
    - **Importante**: para construir `oot_master` hacemos una mediana por
      columnas de `F_oot = [f_j]` tal cual. Esto **asume** que las rejillas
      nativas de todas las exposiciones son idénticas (mismo N y misma λ_i).
      Si no es tu caso, antes hay que re-muestrear los `f_j` a una rejilla
      común en λ (o construir OOT directamente en velocidad).
    - Si no hay OOT suficientes se lanza `ValueError`.
    """    
    bjds = np.asarray(bjds, float)

    # 1) Derive (or accept) in-transit mask in PHASE space (contacts are phases)
    if it_mask is None:
        if contacts is None:
            contacts = compute_contacts_auto(ephem)  # T1..T4 in phase units
        phases = phases_from_bjd(bjds, ephem.T0_bjdtdb, ephem.period_days)
        it_mask = (phases >= contacts.T1) & (phases <= contacts.T4)
    else:
        it_mask = np.asarray(it_mask, bool)

    # 2) Shift each to stellar rest if RV info supplied
    w_rest = []
    for i, w in enumerate(waves):
        if bervs is not None and rv_stars is not None:
            w0 = to_stellar_rest(w, rv_star_kms=rv_stars[i], berv_kms=bervs[i])
        else:
            w0 = np.asarray(w, float)
        w_rest.append(w0)

    # 3) Build OOT master on native grid
    oot_mask = ~it_mask

    F_oot = [f for f, m in zip(fluxes, oot_mask) if m]
    if len(F_oot) == 0:
        n_it = int(np.sum(it_mask)); n_oot = int(np.sum(oot_mask))
        raise ValueError(f"No OOT exposures available (IT={n_it}, OOT={n_oot}). "
                         "Check contact phases / mask input.")

    # Comprobación de saneado básico (evitar todo-NaN)
    _valid = [f for f in F_oot if f is not None and np.size(f) > 0 and np.isfinite(f).any()]
    if not _valid:
        raise ValueError("No valid out-of-transit flux arrays (all empty or NaN).")

    # ¡Asume misma rejilla nativa! (ver notas en docstring)
    oot_master = np.nanmedian(np.vstack(F_oot), axis=0)

    # 4) Residuals per exposure (normalize by sidebands first)
    #    Usamos sidebands de esa misma exposición para aplanar r_raw = f/oot.
    resid = []
    for i, f in enumerate(fluxes):
        # r_raw es el cociente positivo (≈1 en continuo)
        r_raw  = frac_residual(f, oot_master) + 1.0           # = f / oot_master
        # Normaliza continuo por bandas laterales en la rejilla de esta expo
        r_norm = _normalize_by_sidebands_local(w_rest[i], r_raw, center_A, half_width_A)
        # Residual final (F/F_OOT - 1), ya con continuo aplanado
        resid.append(r_norm - 1.0)                             # final residual
        
    # 5) Velocity grid common for all exposures
    v_grid = np.arange(vmin, vmax + dv, dv, dtype=float)

    # 6) Planet velocities per exposure (requires phases) --> (llevar la señal a v≈0)
    phases = phases_from_bjd(bjds, ephem.T0_bjdtdb, ephem.period_days)
    v_p = planet_rv_kms(phases, ephem.Kp_kms)

    # 7) Ventana + λ→v + (−v_planeta) + interpolación a v_grid
    resid_v = []
    for i in range(len(resid)):
        # Recorte a ventana de línea en λ (ya en reposo estelar)
        w_i, r_i = cut_window(w_rest[i], resid[i], center_A, half_width_A)

        # --- OPCIONAL: quita tendencia lineal usando sidebands ---
        #   - define velocidad local en la ventana (sin marco planetario)
        #   - usa |v| >= 30 km/s como bandas laterales
        #   - ajusta recta y la resta del residual de esta exposición
        v_loc = (C_KMS * (w_i - center_A) / center_A).astype(float)
        sb = (np.abs(v_loc) >= 30.0)              # bandas laterales
        if np.isfinite(r_i[sb]).sum() >= 5:
            a, b = np.polyfit(v_loc[sb], r_i[sb], 1)
            r_i = r_i - (a * v_loc + b)
        # --- fin opcional ---

        # Interpolación a la rejilla común **substraída** la velocidad planetaria
        rv_i = residual_on_vgrid_with_shift(w_i, r_i, center_A, v_grid, v_p[i])
        resid_v.append(rv_i)

    # Salidas para el resto del pipeline (stack, ajuste, nulos, inyección,…)
    return v_grid, resid_v