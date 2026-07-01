"""Shared region-box masking for Amazon/Africa 5P workflows.

Region boxes are defined with longitude in 0–360° (e.g. Amazon lon 270–330).
E3SM grids use −180…180; convert before masking so Amazon cells are found.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

Box = Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max


def lon_to_360(lon: np.ndarray) -> np.ndarray:
    """Map longitudes to 0–360° for region-box tests (NaN preserved)."""
    out = np.asarray(lon, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    out[out < 0] += 360.0
    return out


def region_mask_arrays(
    lat: np.ndarray,
    lon: np.ndarray,
    box: Box,
) -> np.ndarray:
    """Boolean mask for points inside ``box`` (lon bounds are 0–360)."""
    lat_min, lat_max, lon_min, lon_max = box
    lat = np.asarray(lat, dtype=np.float64)
    lon360 = lon_to_360(lon)
    return (
        np.isfinite(lat)
        & np.isfinite(lon360)
        & (lat >= lat_min)
        & (lat <= lat_max)
        & (lon360 >= lon_min)
        & (lon360 <= lon_max)
    )


def region_mask_df(df: pd.DataFrame, box: Box) -> np.ndarray:
    """Region mask from a DataFrame with Longitude/Latitude columns."""
    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("DataFrame must contain 'Longitude' and 'Latitude' columns.")
    return region_mask_arrays(
        df["Latitude"].to_numpy(),
        df["Longitude"].to_numpy(),
        box,
    )


def reference_site_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    rlon: float,
    rlat: float,
    atol: float = 1e-4,
) -> np.ndarray:
    """Match a reference site; ``rlon`` may be 0–360 while ``lon`` is −180…180."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    candidates: Sequence[float] = [rlon]
    if rlon > 180.0:
        candidates = (rlon, rlon - 360.0)
    elif rlon < 0.0:
        candidates = (rlon, rlon + 360.0)
    out = np.zeros(len(lon), dtype=bool)
    for clon in candidates:
        out |= np.isclose(lon, clon, atol=atol) & np.isclose(lat, rlat, atol=atol)
    return out
