import numpy as np

def nanmedian_stack(arrs):
    A = np.vstack(arrs)
    return np.nanmedian(A, axis=0)

def nanpercentile_band(arrs, low=16, high=84):
    A = np.vstack(arrs)
    lo = np.nanpercentile(A, low, axis=0)
    hi = np.nanpercentile(A, high, axis=0)
    return lo, hi

def median_stack_with_band(arrs):
    A = np.vstack(arrs)
    med = np.nanmedian(A, axis=0)
    lo = np.nanpercentile(A, 16, axis=0)
    hi = np.nanpercentile(A, 84, axis=0)
    return med, lo, hi

def robust_nanmedian(arr, axis=0, min_valid=1):
    """
    Mediana ignorando NaNs, pero devuelve NaN allí donde el nº de valores
    finitos a lo largo de 'axis' es < min_valid.

    Parameters
    ----------
    arr : array_like
    axis : int, default 0
    min_valid : int, default 1
        Mínimo de elementos finitos requeridos para calcular la mediana.

    Returns
    -------
    med : ndarray or float
        Mediana por 'axis' con NaN donde no hay suficientes datos.
    """
    a = np.asarray(arr, dtype=float)

    # Evita warning "All-NaN slice encountered"
    with np.errstate(all="ignore"):
        med = np.nanmedian(a, axis=axis)

    if min_valid <= 1:
        return med

    finite = np.isfinite(a)
    cnt = np.sum(finite, axis=axis)

    # Scalar vs vector
    if np.ndim(med) == 0:
        return med if cnt >= min_valid else np.nan

    med = np.array(med, dtype=float, copy=True)
    med[cnt < min_valid] = np.nan
    return med

# Si usas __all__, añádelo:
# __all__ = [..., "robust_nanmedian"]