#!/usr/bin/env python3
"""
Compare AI CNP predictions with reference (model) or with CSV exports.

Design:
  - Primary: AI predictions NetCDF vs model (ELM) NetCDF.
    Use --ai-predictions and --model (and --compare-to model to avoid CSV).

  - Optional: AI predictions NetCDF vs CSV predictions (same run, two formats).
    Use --csv-predictions for a consistency check. Use --compare-to csv or auto.

How AI vs model comparison works (different internal formats):

  The AI NetCDF and the model restart NetCDF do not have the same internal
  layout. The script bridges them as follows.

  Required for both:
    - grid1d_lon, grid1d_lat (1D arrays of length n_gridcell)
    - Same variable names (e.g. cwdc_vr, tlai) for the fields to compare

  AI NetCDF (e.g. from ai_predictions_to_netcdf.py):
    - One gridcell per land point; dimension "gridcell".
    - Scalars: (gridcell).
    - PFT variables: (pft, gridcell) with pft size 16.
    - Soil/column variables: (column, levgrnd, gridcell) with column size 1,
      levgrnd size 10 (one column per gridcell).

  Model (ELM) NetCDF:
    - Many columns and PFT instances; each is linked to a gridcell via
      cols1d_gridcell_index and pfts1d_gridcell_index (1-based or 0-based).
    - Scalars: often (gridcell).
    - Column variables: (column, levgrnd) — column index maps to gridcell.
    - PFT variables: (pft) — pft index maps to gridcell.

  The script:
    1. Builds a spatial mapping: for each AI gridcell (lon, lat), finds the
       closest model gridcell (nearest-neighbor).
    2. For each model gridcell g: gets AI value from the AI gridcell that
       maps to g; gets model value from the model’s column/pft indices for g
       (e.g. first column and PFT 1–16 for that gridcell).
    3. Plots and stats are on the model grid (one value per model gridcell).

  So the AI NetCDF must be in the “plotting” format produced by
  ai_predictions_to_netcdf.py (grid1d_lon/lat, gridcell, pft/gridcell,
  column/levgrnd/gridcell). If your AI NetCDF has a different layout, convert
  it first or extend this script to support that layout.
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
import argparse
from pathlib import Path
import sys
from typing import List
import csv
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list

# Default paths
FALLBACK_AI_PREDICTIONS = './comparison_results/ai_predictions_for_plotting.nc'
#FALLBACK_MODEL = '/global/cfs/cdirs/m4814/daweigao/14_Code/all_dataset_1_degree/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc'
FALLBACK_MODEL = '/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/20251201_TRENDY2024_default_ICB1850CNPRDCTCBC.elm.r.0801-01-01-00000.nc'
FALLBACK_OUTPUT_DIR = "./ai_model_comparison_plots"
FALLBACK_CSV_PREDICTIONS = './cnp_inference_entire_dataset/cnp_predictions'

# Default variables to plot unless '--variables all' is used
VARIABLES = ['cwdc_vr', 'soil3c_vr', 'tlai', 'deadstemc', 'cpool', 'npool', 'ppool', 'primp_vr', 'secondp_vr', 'litr2c_vr', 'litr2n_vr', 'litr2p_vr', 'soil1c_vr', 'soil1n_vr', 'soil1p_vr']

# Layers for column-type variables (AI has 10 layers, model has 15)
LEVGRND_LAYERS = [0,1,2,3,4,5,6,7,8,9]  # Layers 0, 4, 9 (corresponding to AI layers 1, 5, 10)

# PFTs to plot (AI has PFT1-16, model has PFT0-16)
# Note: AI PFT0 = Model PFT1, AI PFT1 = Model PFT2, etc.
PFT_PICK_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # PFT1, PFT2, PFT3, PFT4, PFT5 (0-indexed, so 0=PFT1, 1=PFT2, etc.)

CSV_LONGITUDE_NAMES = ("Longitude", "Long", "long", "lon", "LON")
CSV_LATITUDE_NAMES = ("Latitude", "Lat", "lat", "LAT")
CSV_COORD_COLUMNS = CSV_LONGITUDE_NAMES + CSV_LATITUDE_NAMES


QUALITY_THRESHOLDS = {
    'good': 0.9,  # R² >= 0.9 for good quality
    'ok': 0.7     # R² >= 0.7 for ok quality
}


def _extract_coords_from_df(df: pd.DataFrame):
    lon = lat = None
    for col in CSV_LONGITUDE_NAMES:
        if col in df.columns:
            lon = pd.to_numeric(df[col], errors='coerce').to_numpy()
            break
    for col in CSV_LATITUDE_NAMES:
        if col in df.columns:
            lat = pd.to_numeric(df[col], errors='coerce').to_numpy()
            break
    return lon, lat


def _drop_coord_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in CSV_COORD_COLUMNS if c in df.columns], errors='ignore')


def _get_case_insensitive(mapping, key):
    if mapping is None:
        return None
    if key in mapping:
        return mapping[key]
    key_lower = key.lower()
    for k, v in mapping.items():
        if k.lower() == key_lower:
            return v
    return None


def _classify_quality_from_r2(value: float) -> str:
    """Classify quality based on R² value (similar to generate_prediction_quality_report.py)"""
    if value is None or not np.isfinite(value):
        return 'unknown'
    val = float(value)
    if val >= QUALITY_THRESHOLDS['good']:
        return 'good'
    if val >= QUALITY_THRESHOLDS['ok']:
        return 'ok'
    return 'bad'


def load_csv_predictions(csv_path: str) -> dict:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV predictions source not found: {csv_path}")

    store = {
        'source': str(path),
        'scalar_df': None,
        'scalar_columns': {},
        'scalar_vars': set(),
        'scalar_coords': None,
        'pft_data': {},
        'pft_columns': {},
        'pft_vars': set(),
        'pft_coords': {},
        'soil_data': {},
        'soil_columns': {},
        'soil_vars': set(),
        'soil_coords': {},
        'static_inverse': None,
        'flat_df': None,
        'flat_columns': {},
        'flat_vars': set(),
        'flat_coords': None,
    }

    if path.is_dir():
        scalar_path = path / 'predictions_scalar.csv'
        if scalar_path.exists():
            df = pd.read_csv(scalar_path)
            lon, lat = _extract_coords_from_df(df)
            data = _drop_coord_columns(df)
            col_map = {}
            scalar_vars = set()
            for col in data.columns:
                base = col[2:] if col.startswith('Y_') else col
                scalar_vars.add(base)
                if base not in col_map:
                    col_map[base] = col
                if base.lower() not in col_map:
                    col_map[base.lower()] = col
            store['scalar_df'] = data
            store['scalar_columns'] = col_map
            store['scalar_vars'] = scalar_vars
            store['scalar_coords'] = (lon, lat)

        pft_dir = path / 'pft_1d_predictions'
        if pft_dir.exists():
            for csv_file in sorted(pft_dir.glob('predictions_*.csv')):
                var_name = csv_file.stem.replace('predictions_Y_', '')
                df = pd.read_csv(csv_file)
                lon, lat = _extract_coords_from_df(df)
                data = _drop_coord_columns(df)
                cols = [c for c in data.columns if '_pft' in c]
                if not cols:
                    continue
                def _pft_idx(col_name: str) -> int:
                    try:
                        return int(col_name.split('_pft')[-1])
                    except Exception:
                        return 999999
                cols = sorted(cols, key=_pft_idx)
                store['pft_data'][var_name] = data
                store['pft_columns'][var_name] = cols
                store['pft_vars'].add(var_name)
                store['pft_coords'][var_name] = (lon, lat)

        soil_dir = path / 'soil_2d_predictions'
        if soil_dir.exists():
            for csv_file in sorted(soil_dir.glob('predictions_*.csv')):
                var_name = csv_file.stem.replace('predictions_Y_', '')
                df = pd.read_csv(csv_file)
                lon, lat = _extract_coords_from_df(df)
                data = _drop_coord_columns(df)
                cols = [c for c in data.columns if '_layer' in c]
                if not cols:
                    continue
                def _layer_idx(col_name: str) -> int:
                    try:
                        return int(col_name.split('_layer')[-1])
                    except Exception:
                        return 999999
                cols = sorted(cols, key=_layer_idx)
                store['soil_data'][var_name] = data
                store['soil_columns'][var_name] = cols
                store['soil_vars'].add(var_name)
                store['soil_coords'][var_name] = (lon, lat)

        static_path = path / 'test_static_inverse.csv'
        if static_path.exists():
            try:
                store['static_inverse'] = pd.read_csv(static_path)
            except Exception as exc:
                print(f"Warning: failed to load test_static_inverse.csv: {exc}")

    elif path.is_file():
        df = pd.read_csv(path)
        lon, lat = _extract_coords_from_df(df)
        data = _drop_coord_columns(df)
        col_map = {}
        flat_vars = set()
        for col in data.columns:
            base = col[2:] if col.startswith('Y_') else col
            flat_vars.add(base)
            if base not in col_map:
                col_map[base] = col
            if base.lower() not in col_map:
                col_map[base.lower()] = col
        store['flat_df'] = data
        store['flat_columns'] = col_map
        store['flat_vars'] = flat_vars
        store['flat_coords'] = (lon, lat)
    else:
        raise ValueError(f"Unsupported CSV predictions path: {csv_path}")

    available = set()
    for key in ('scalar_vars', 'pft_vars', 'soil_vars', 'flat_vars'):
        available.update(store.get(key, set()))
    store['available_vars'] = available
    print(f"Loaded CSV predictions from {path} with {len(available)} variables")
    return store


def build_variable_category_map(variable_list_path: str) -> dict:
    if not variable_list_path:
        return {}
    try:
        parsed = parse_cnp_io_list(variable_list_path)
    except Exception as exc:
        print(f"Warning: unable to parse variable list for category mapping: {exc}")
        return {}
    mapping = {}
    for name in parsed.get('scalar_variables', []) or []:
        mapping[name] = 'scalar'
    for name in parsed.get('pft_1d_variables', []) or []:
        mapping[name] = 'pft1d'
    for name in parsed.get('variables_2d_soil', []) or []:
        mapping[name] = 'soil2d'
    return mapping


def infer_variable_category(var: str, dims, category_map: dict) -> str:
    if category_map and var in category_map:
        return category_map[var]
    if dims and any(dim in ('levgrnd', 'column') for dim in dims):
        return 'soil2d'
    if dims and any(dim == 'pft' for dim in dims):
        return 'pft1d'
    return 'scalar'


def csv_has_variable(store: dict, var: str, category: str = None) -> bool:
    if store is None:
        return False
    if category == 'scalar':
        if store.get('scalar_df') is not None and _get_case_insensitive(store.get('scalar_columns'), var):
            return True
        if store.get('flat_df') is not None and _get_case_insensitive(store.get('flat_columns'), var):
            return True
        return False
    if category == 'pft1d':
        return _get_case_insensitive(store.get('pft_data'), var) is not None
    if category == 'soil2d':
        return _get_case_insensitive(store.get('soil_data'), var) is not None
    return (csv_has_variable(store, var, 'scalar') or
            csv_has_variable(store, var, 'pft1d') or
            csv_has_variable(store, var, 'soil2d'))


def extract_csv_scalar(store: dict, var: str):
    df = store.get('scalar_df')
    columns = store.get('scalar_columns')
    col = _get_case_insensitive(columns, var) if columns else None
    if df is None or col is None:
        df = store.get('flat_df')
        columns = store.get('flat_columns')
        col = _get_case_insensitive(columns, var) if columns else None
    if df is None or col is None:
        col = _get_case_insensitive(columns, f'Y_{var}') if columns else None
        if df is None or col is None:
            return None
    return pd.to_numeric(df[col], errors='coerce').to_numpy()


def extract_csv_pft(store: dict, var: str):
    data_dict = store.get('pft_data')
    df = _get_case_insensitive(data_dict, var)
    if df is None:
        return None
    columns_map = store.get('pft_columns')
    cols = _get_case_insensitive(columns_map, var)
    if not cols:
        cols = [c for c in df.columns if '_pft' in c]
        if not cols:
            return None
        def _pft_idx(col_name: str) -> int:
            try:
                return int(col_name.split('_pft')[-1])
            except Exception:
                return 999999
        cols = sorted(cols, key=_pft_idx)
    arr = np.full((len(cols), len(df)), np.nan, dtype=float)
    for idx, col in enumerate(cols):
        arr[idx, :] = pd.to_numeric(df[col], errors='coerce').to_numpy()
    return arr


def extract_csv_soil(store: dict, var: str):
    data_dict = store.get('soil_data')
    df = _get_case_insensitive(data_dict, var)
    if df is None:
        return None
    columns_map = store.get('soil_columns')
    cols = _get_case_insensitive(columns_map, var)
    if not cols:
        cols = list(df.columns)
    arr = np.full((len(cols), len(df)), np.nan, dtype=float)
    for idx, col in enumerate(cols):
        arr[idx, :] = pd.to_numeric(df[col], errors='coerce').to_numpy()
    return arr


def _safe_get(ds, name):
    """Safely get a variable from dataset, with error handling."""
    if name not in ds:
        raise KeyError(f"Missing required variable/coordinate: {name}")
    return ds[name]

def _to_zero_based_index(idx_raw, n_grid):
    """Convert gridcell indices to zero-based indexing."""
    idx = np.asarray(idx_raw, dtype=np.int64).copy()
    if idx.size == 0:
        return np.full_like(idx, -1)
    is_one_based = (np.any(idx == n_grid) or (np.nanmin(idx) == 1))
    if is_one_based:
        idx = idx - 1
    idx[(idx < 0) | (idx >= n_grid)] = -1
    return idx

def _build_gridcell_groups(one_d_to_grid, n_grid):
    """Build groups of indices for each gridcell."""
    groups = [[] for _ in range(n_grid)]
    for idx, g in enumerate(one_d_to_grid):
        if 0 <= g < n_grid:
            groups[g].append(idx)
    return groups

def _create_ai_to_model_mapping(ds_ai, ds_model, n_grid):
    """Create spatial mapping from AI gridcells to model gridcells."""
    from scipy.spatial.distance import cdist
    
    # Get AI coordinates
    ai_lon = ds_ai['grid1d_lon'].values
    ai_lat = ds_ai['grid1d_lat'].values
    
    # Get model coordinates (already extracted as grid_lon, grid_lat)
    model_lon = ds_model['grid1d_lon'].values
    model_lat = ds_model['grid1d_lat'].values
    
    # Create coordinate arrays
    ai_coords = np.column_stack([ai_lon, ai_lat])
    model_coords = np.column_stack([model_lon, model_lat])
    
    # Find closest model gridcell for each AI gridcell
    distances = cdist(ai_coords, model_coords)
    ai_to_model_mapping = np.argmin(distances, axis=1)
    
    print(f"Created AI-to-model spatial mapping:")
    print(f"  AI gridcells: {len(ai_lon)}")
    print(f"  Model gridcells: {len(model_lon)}")
    print(f"  Mapping range: AI gridcell 0->model gridcell {ai_to_model_mapping[0]}")
    print(f"  Mapping range: AI gridcell {len(ai_lon)-1}->model gridcell {ai_to_model_mapping[-1]}")
    
    return ai_to_model_mapping

def _to_nan_fillvalue(arr, fill_threshold=1e35):
    """Convert fill values to NaN."""
    a = np.asarray(arr, dtype=float)
    a[np.abs(a) >= fill_threshold] = np.nan
    return a

def _gridcell_lonlat(ds):
    """Extract gridcell longitude and latitude."""
    lon = _safe_get(ds, "grid1d_lon").values
    lat = _safe_get(ds, "grid1d_lat").values
    return lon, lat

def _to_zero_based_index(idx_raw, n_grid):
    """Convert one-based indices to zero-based indices."""
    idx = np.asarray(idx_raw, dtype=np.int64).copy()
    if idx.size == 0:
        return np.full_like(idx, -1)
    is_one_based = (np.any(idx == n_grid) or (np.nanmin(idx) == 1))
    if is_one_based:
        idx = idx - 1
    idx[(idx < 0) | (idx >= n_grid)] = -1
    return idx

def _build_gridcell_groups(one_d_to_grid, n_grid):
    """Build mapping from gridcell to indices."""
    groups = [[] for _ in range(n_grid)]
    for idx, g in enumerate(one_d_to_grid):
        if 0 <= g < n_grid:
            groups[g].append(idx)
    return groups

def _plot_map(ax, lon, lat, data, title, vmin=None, vmax=None, cmap="viridis", norm=None):
    """Plot a map with the given data."""
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)
    im = ax.scatter(lon, lat, c=data, s=5, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)

def _percent_diff_categories(model_vals: np.ndarray, ai_vals: np.ndarray) -> np.ndarray:
    """Create percent difference categories for visualization."""
    model = np.asarray(model_vals, dtype=float)
    ai = np.asarray(ai_vals, dtype=float)
    cat = np.full(model.shape, np.nan, dtype=float)
    finite = np.isfinite(model) & np.isfinite(ai)
    if not np.any(finite):
        return cat
    
    # Threshold: zero-out percentage where |model| <= 2% of mean(|model|)
    mean_abs_model = np.nanmean(np.abs(model[finite])) if np.any(finite) else 0.0
    threshold = 0.02 * mean_abs_model
    denom = np.abs(model[finite])
    
    # Compute percent (AI - model)/model in %
    pct = (ai[finite] - model[finite]) / np.where(denom > 0, denom, 1.0) * 100.0
    
    # Apply threshold rule
    pct[denom <= threshold] = 0.0
    sign = np.sign(pct)
    mag = np.abs(pct)
    bins = np.zeros_like(mag, dtype=int)
    bins[(mag >= 0) & (mag < 5)] = 1
    bins[(mag >= 5) & (mag < 15)] = 2
    bins[(mag >= 15) & (mag < 30)] = 3
    bins[mag >= 30] = 4
    bins[(mag == 0)] = 0
    categories = sign.astype(int) * bins
    cat[finite] = categories.astype(float)
    return cat

def _plot_tripanel(var, label_suffix, lon, lat, data_ai, data_model, out_dir,
                   label_ai="AI Predictions", label_model="Model Results", plot: bool = True):
    """Create a tri-panel comparison plot and return statistics.

    When plot=False, figures are not generated; only statistics are computed and returned.
    """
    diff = data_ai - data_model
    vmin_orig = np.nanmin([np.nanmin(data_ai), np.nanmin(data_model)])
    vmax_orig = np.nanmax([np.nanmax(data_ai), np.nanmax(data_model)])
    diff_abs = np.nanmax(np.abs(diff))

    # Compute stats and metrics
    finite_ai = np.isfinite(data_ai)
    finite_model = np.isfinite(data_model)
    mask = finite_ai & finite_model
    n = int(mask.sum())

    def _nan_min(a):
        return float(np.nanmin(a)) if np.any(np.isfinite(a)) else float('nan')

    def _nan_max(a):
        return float(np.nanmax(a)) if np.any(np.isfinite(a)) else float('nan')

    sum_ai = float(np.nansum(data_ai))
    std_ai = float(np.nanstd(data_ai)) if np.any(np.isfinite(data_ai)) else float('nan')
    min_ai = _nan_min(data_ai)
    max_ai = _nan_max(data_ai)

    sum_model = float(np.nansum(data_model))
    std_model = float(np.nanstd(data_model)) if np.any(np.isfinite(data_model)) else float('nan')
    min_model = _nan_min(data_model)
    max_model = _nan_max(data_model)

    if n > 0:
        y = data_ai[mask]
        yhat = data_model[mask]
        mse = float(np.mean((y - yhat) ** 2))
        rmse = float(np.sqrt(mse))
        mean_y = float(np.mean(y))
        nrmse = (rmse / mean_y) if mean_y != 0 else float('inf')
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
    else:
        rmse = float('nan')
        nrmse = float('nan')
        r2 = float('nan')

    print(f"Stats for {var}{label_suffix}:")
    print(f"  {label_ai}: sum={sum_ai:.6g} std={std_ai:.6g} min={min_ai:.6g} max={max_ai:.6g}")
    print(f"  {label_model}: sum={sum_model:.6g} std={std_model:.6g} min={min_model:.6g} max={max_model:.6g}")
    print(f"  Metrics (AI vs Model): n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}")

    stats = {
        "ai_sum": sum_ai,
        "ai_std": std_ai,
        "ai_min": min_ai,
        "ai_max": max_ai,
        "model_sum": sum_model,
        "model_std": std_model,
        "model_min": min_model,
        "model_max": max_model,
        "n": n,
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
    }

    if not plot:
        return stats

    fig = plt.figure(figsize=(12, 20))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.3)

    # Panel 1: AI Predictions
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    _plot_map(ax1, lon, lat, data_ai, f"{var} - {label_ai}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    # Panel 2: Model Results
    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    _plot_map(ax2, lon, lat, data_model, f"{var} - {label_model}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    # Panel 3: Difference (AI - Model)
    ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    if not np.isfinite(diff_abs) or diff_abs <= 0:
        msg = "No Difference" if diff_abs == 0 else "All NaN"
        ax3.text(0.5, 0.5, msg, ha="center", va="center", transform=ax3.transAxes,
                 fontsize=14, fontweight="bold", color="gray")
        ax3.set_title(f"{var} - Difference (AI - Model)", fontsize=14, fontweight="bold")
        ax3.set_global()
        gl = ax3.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0, vmax=diff_abs)
        _plot_map(ax3, lon, lat, diff, f"{var} - Difference (AI - Model)", cmap="RdBu_r", norm=norm)

    # Panel 4: Percent Difference Categories
    ax4 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
    # Treat model as reference, AI as comparison
    cat = _percent_diff_categories(data_model, data_ai)
    colors = [
        "#08519c",  # -4: 30%+
        "#3182bd",  # -3: 15–30%
        "#6baed6",  # -2: 5–15%
        "#c6dbef",  # -1: 0–5%
        "#bdbdbd",  #  0: 0
        "#fcbba1",  # +1: 0–5%
        "#fc9272",  # +2: 5–15%
        "#fb6a4a",  # +3: 15–30%
        "#cb181d",  # +4: 30%+
    ]
    cmap = ListedColormap(colors)
    boundaries = [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm_cat = BoundaryNorm(boundaries, cmap.N)

    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS)
    ax4.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax4.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)
    im4 = ax4.scatter(lon, lat, c=cat, s=5, cmap=cmap, norm=norm_cat, transform=ccrs.PlateCarree())
    ax4.set_title(f"{var} - Percent Diff bins ((AI-Model)/Model)", fontsize=14, fontweight="bold")
    ax4.set_global()
    gl4 = ax4.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl4.top_labels = False
    gl4.right_labels = False
    cbar = plt.colorbar(im4, ax=ax4, shrink=0.7, pad=0.05, ticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels([
        "-30%+", "-15–30%", "-5–15%", "-0–5%", "0", "0–5%", "5–15%", "15–30%", "30%+"
    ])

    plt.suptitle(f"{var} {label_suffix}", fontsize=16, fontweight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{var}{label_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return stats

def _load_default_paths(variable_list_path: str):
    defaults = {}
    if variable_list_path:
        try:
            vl_path = Path(variable_list_path)
            if vl_path.exists():
                parsed = parse_cnp_io_list(variable_list_path)
                for key in ('ai_predictions_default', 'model_default', 'comparison_output_dir', 'csv_predictions_default'):
                    value = parsed.get(key) if isinstance(parsed, dict) else None
                    if value:
                        defaults[key] = value
        except Exception as exc:
            print(f"Warning: Failed to load default paths from {variable_list_path}: {exc}")
    return defaults

def parse_variable_list_file(variable_list_path: str) -> List[str]:
    """Parse the CNP IO list file to extract all variables."""
    print(f"Parsing variable list file: {variable_list_path}")
    
    try:
        parsed = parse_cnp_io_list(variable_list_path)
        
        # Extract all variables from all sections
        all_variables = []
        
        # Add scalar variables
        if 'scalar_variables' in parsed:
            all_variables.extend(parsed['scalar_variables'])
            print(f"  Found {len(parsed['scalar_variables'])} scalar variables")
        
        # Add PFT 1D variables
        if 'pft_1d_variables' in parsed:
            all_variables.extend(parsed['pft_1d_variables'])
            print(f"  Found {len(parsed['pft_1d_variables'])} PFT 1D variables")
        
        # Add soil 2D variables
        if 'variables_2d_soil' in parsed:
            all_variables.extend(parsed['variables_2d_soil'])
            print(f"  Found {len(parsed['variables_2d_soil'])} soil 2D variables")
        
        print(f"  Total variables to plot: {len(all_variables)}")
        print(f"  Variables: {all_variables}")
        
        return all_variables
        
    except Exception as e:
        print(f"Warning: Could not parse variable list file: {e}")
        print("  Falling back to default variables")
        return VARIABLES


def discover_common_variables(ds_ai, ds_model):
    """Discover variables that exist in both datasets."""
    ai_vars = set(ds_ai.data_vars.keys())
    model_vars = set(ds_model.data_vars.keys())
    common_vars = ai_vars.intersection(model_vars)
    
    # Filter out coordinate variables and metadata
    exclude_patterns = ['lon', 'lat', 'index', 'period', 'time', 'bnds']
    filtered_vars = []
    
    for var in common_vars:
        if not any(pattern in var.lower() for pattern in exclude_patterns):
            filtered_vars.append(var)
    
    return sorted(filtered_vars)

def main():
    parser = argparse.ArgumentParser(
        description='Compare AI predictions with model results using tri-panel plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python ai_model_comparison_plot.py

  # Use variable list file to plot all variables
  python ai_model_comparison_plot.py \\
    --variable-list CNP_IO_demo1.txt

  # Specify custom paths
  python ai_model_comparison_plot.py \\
    --ai-predictions ./my_ai_predictions.nc \\
    --model ./my_model.nc \\
    --output-dir ./my_comparison_plots

  # Plot specific variables only
  python ai_model_comparison_plot.py --variables cwdc_vr tlai
        """
    )
    
    parser.add_argument('--ai-predictions', default=None,
                       help=f'Path to AI predictions NetCDF file [default: AI_PREDICTIONS_DEFAULT in variable list or {FALLBACK_AI_PREDICTIONS}]')
    parser.add_argument('--model', default=None,
                       help=f'Path to model results NetCDF file [default: MODEL_DEFAULT in variable list or {FALLBACK_MODEL}]')
    parser.add_argument('--output-dir', default=None,
                       help=f'Output directory for plots [default: COMPARISON_OUTPUT_DIR in variable list or {FALLBACK_OUTPUT_DIR}]')
    parser.add_argument('--variable-list', type=str,
                       help='Path to CNP_IO_list file to extract all variables for plotting')
    parser.add_argument('--variables', nargs='*', default=VARIABLES,
                       help=f"Variables to plot, or 'all' to plot every available variable [default: {VARIABLES}]")
    parser.add_argument('--layers', nargs='*', type=int, default=LEVGRND_LAYERS,
                       help=f'Layers to plot for column variables [default: {LEVGRND_LAYERS}]')
    parser.add_argument('--pfts', nargs='*', type=int, default=PFT_PICK_LIST,
                       help=f'PFTs to plot [default: {PFT_PICK_LIST}]')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plot generation; compute and save statistics only')
    parser.add_argument('--stats-file', type=str,
                       help='Path to write CSV of statistics (defaults to output dir stats.txt)')
    parser.add_argument('--stats-format', type=str, choices=['csv', 'txt', 'both'], default='txt',
                       help='Format of statistics output: csv, txt, or both [default: txt]')
    parser.add_argument('--csv-predictions', type=str, default=None,
                       help='Path to CSV predictions (directory or file) containing original AI outputs for comparison')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only compute statistics; disables plotting and processes all layers/PFTs when used with --variable-list')

    args = parser.parse_args()

    defaults_from_config = _load_default_paths(args.variable_list)

    def _resolve_default(current_value, config_key, fallback):
        candidate = current_value or defaults_from_config.get(config_key)
        if candidate:
            return str(candidate)
        return fallback

    args.ai_predictions = _resolve_default(args.ai_predictions, 'ai_predictions_default', FALLBACK_AI_PREDICTIONS)
    user_provided_model = (args.model is not None)
    args.model = _resolve_default(args.model, 'model_default', FALLBACK_MODEL)
    args.output_dir = _resolve_default(args.output_dir, 'comparison_output_dir', FALLBACK_OUTPUT_DIR)
    args.csv_predictions = _resolve_default(args.csv_predictions, 'csv_predictions_default', FALLBACK_CSV_PREDICTIONS)

    # If user passed --model (or --restart-file), compare to that file; otherwise use CSV if path exists
    use_csv_predictions = bool(args.csv_predictions) and not user_provided_model
    if getattr(args, 'restart_file', None):
        use_csv_predictions = False
    if args.stats_only:
        args.no_plot = True

    # Validate input files
    if not Path(args.ai_predictions).exists():
        raise FileNotFoundError(f"AI predictions file not found: {args.ai_predictions}")
    if use_csv_predictions:
        if not Path(args.csv_predictions).exists():
            raise FileNotFoundError(f"CSV predictions source not found: {args.csv_predictions}")
    else:
        if not Path(args.model).exists():
            raise FileNotFoundError(f"Model file not found: {args.model}")

    print("="*60)
    if use_csv_predictions:
        print("CSV vs NetCDF Comparison")
    else:
        print("AI vs Model Comparison")
    print("="*60)
    print(f"AI predictions (NetCDF): {args.ai_predictions}")
    if use_csv_predictions:
        print(f"CSV predictions: {args.csv_predictions}")
    else:
        print(f"Model results: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variables to plot: {args.variables}")
    print(f"Layers to plot: {args.layers}")
    print(f"PFTs to plot: {args.pfts}")
    print("="*60)
    
    # Open datasets once
    ds_ai = xr.open_dataset(args.ai_predictions)
    csv_predictions = None
    if use_csv_predictions:
        csv_predictions = load_csv_predictions(args.csv_predictions)
        ds_model = None
    else:
        ds_model = xr.open_dataset(args.model)

    # Determine variable selection behavior
    requested_all = False
    if args.variables:
        requested_all = (len(args.variables) == 1 and str(args.variables[0]).lower() == 'all')
    if args.stats_only and args.variable_list:
        requested_all = True

    variable_category_map = build_variable_category_map(args.variable_list) if args.variable_list else {}

    ai_vars = set(ds_ai.data_vars.keys())

    if requested_all:
        if args.variable_list:
            if not Path(args.variable_list).exists():
                ds_ai.close()
                if ds_model is not None:
                    ds_model.close()
                raise FileNotFoundError(f"Variable list file not found: {args.variable_list}")
            all_variables = parse_variable_list_file(args.variable_list)
            if use_csv_predictions:
                available_vars = [
                    var for var in all_variables
                    if var in ai_vars and csv_has_variable(csv_predictions, var, variable_category_map.get(var))
                ]
            else:
                model_vars = set(ds_model.data_vars.keys())
                available_vars = [var for var in all_variables if var in ai_vars and var in model_vars]
            if available_vars:
                args.variables = available_vars
                print(f"Using all variables from variable list ({len(available_vars)}): {available_vars}")
            else:
                print("Warning: No variables from variable list found in available datasets!")
                ds_ai.close()
                if ds_model is not None:
                    ds_model.close()
                return
        else:
            if use_csv_predictions:
                csv_vars = set(csv_predictions.get('available_vars', set()))
                discovered_vars = sorted(ai_vars.intersection(csv_vars))
                if discovered_vars:
                    args.variables = discovered_vars
                    print(f"Using all common variables between NetCDF and CSV ({len(discovered_vars)}): {discovered_vars}")
                else:
                    print("Warning: No common variables found between AI NetCDF predictions and CSV source!")
                    ds_ai.close()
                    return
            else:
                discovered_vars = discover_common_variables(ds_ai, ds_model)
                if discovered_vars:
                    args.variables = discovered_vars
                    print(f"Using all common variables ({len(discovered_vars)}): {discovered_vars}")
                else:
                    print("Warning: No common variables found between AI predictions and model!")
                    ds_ai.close(); ds_model.close()
                    return
    else:
        forced = list(VARIABLES)
        if use_csv_predictions:
            selected = [
                v for v in forced
                if v in ai_vars and csv_has_variable(csv_predictions, v, variable_category_map.get(v))
            ]
            if not selected:
                print("Warning: None of the default variables are present in both NetCDF and CSV data!")
                ds_ai.close()
                return
            missing = [v for v in forced if v not in selected]
            if missing:
                print(f"Note: Skipping default variables missing from CSV source: {missing}")
        else:
            model_vars = set(ds_model.data_vars.keys())
            selected = [v for v in forced if v in ai_vars and v in model_vars]
            if not selected:
                print("Warning: None of the default variables are present in both datasets!")
                ds_ai.close(); ds_model.close()
                return
            missing = [v for v in forced if v not in selected]
            if missing:
                print(f"Note: Skipping missing default variables not present in both datasets: {missing}")
        args.variables = selected
        print(f"Using default subset of variables ({len(selected)}): {selected}")

    if use_csv_predictions:
        grid_lon, grid_lat = _gridcell_lonlat(ds_ai)
        n_grid = ds_ai.sizes['gridcell']
        print(f"Using AI gridcell count: {n_grid}")
        print(f"AI coordinates: lon range [{grid_lon.min():.3f}, {grid_lon.max():.3f}], lat range [{grid_lat.min():.3f}, {grid_lat.max():.3f}]")
        grid_to_cols = None
        grid_to_pfts = None
        ai_to_model_mapping = None
    else:
        grid_lon, grid_lat = _gridcell_lonlat(ds_model)
        n_grid = ds_model.sizes['gridcell']
        print(f"Using MODEL gridcell count: {n_grid}")
        print(f"Model coordinates: lon range [{grid_lon.min():.3f}, {grid_lon.max():.3f}], lat range [{grid_lat.min():.3f}, {grid_lat.max():.3f}]")
        col2grid = _to_zero_based_index(_safe_get(ds_model, 'cols1d_gridcell_index').values, n_grid)
        pft2grid = _to_zero_based_index(_safe_get(ds_model, 'pfts1d_gridcell_index').values, n_grid)
        grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
        grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)
        ai_to_model_mapping = _create_ai_to_model_mapping(ds_ai, ds_model, n_grid)
        print(f"Model mappings: total columns: {col2grid.size} | total pfts: {pft2grid.size}")
        print(f"Example: gridcell 0 -> columns {grid_to_cols[0][:5]}, pfts {grid_to_pfts[0][:5]}")

    # Create output directory
    if args.variable_list and requested_all:
        # Use a more descriptive output directory name when using variable list
        var_list_name = Path(args.variable_list).stem
        output_dir = Path(args.output_dir) / f"comparison_{var_list_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Update the output directory for the plotting function
    args.output_dir = str(output_dir)
    
    print(f"\nStart processing: {len(args.variables)} variables")
    stats_rows = []
    for var in args.variables:
        if var not in ds_ai.data_vars:
            print(f"Skip {var} (not found in AI NetCDF)")
            continue

        da_ai = ds_ai[var]
        dims = da_ai.dims
        category = infer_variable_category(var, dims, variable_category_map)
        print(f"\nVariable {var}, dims: {dims}")
        print(f"  NetCDF shape: {da_ai.shape}")

        if use_csv_predictions:
            if not csv_has_variable(csv_predictions, var, category):
                print(f"  Skip {var} (not present in CSV source)")
                continue

            label_ai = "NetCDF Predictions"
            label_csv = "CSV Predictions"

            if category == 'soil2d' and 'levgrnd' in dims:
                da_ai_sel = da_ai
                if 'column' in da_ai_sel.dims:
                    da_ai_sel = da_ai_sel.isel(column=0)
                try:
                    da_ai_sel = da_ai_sel.transpose('levgrnd', 'gridcell', ...)
                except ValueError:
                    da_ai_sel = da_ai_sel.transpose(..., 'levgrnd', 'gridcell')
                ai_vals = _to_nan_fillvalue(da_ai_sel.values)
                if ai_vals.ndim == 1:
                    ai_vals = ai_vals[np.newaxis, :]
                csv_vals = extract_csv_soil(csv_predictions, var)
                if csv_vals is None:
                    print(f"  Skip {var} (CSV layers unavailable)")
                    continue
                layer_count = ai_vals.shape[0]
                csv_layer_count = csv_vals.shape[0]
                if csv_vals.shape[1] != ai_vals.shape[1]:
                    min_len = min(ai_vals.shape[1], csv_vals.shape[1], len(grid_lon))
                    if min_len == 0:
                        print(f"  Skip {var} (no grid overlap between NetCDF and CSV)")
                        continue
                    print(f"  Warning: grid mismatch for {var}; trimming to {min_len} cells")
                else:
                    min_len = ai_vals.shape[1]
                lon_subset = grid_lon[:min_len]
                lat_subset = grid_lat[:min_len]
                for lev in range(layer_count):
                    ai_layer = ai_vals[lev, :min_len]
                    if lev < csv_layer_count:
                        csv_layer = csv_vals[lev, :min_len]
                    else:
                        csv_layer = np.full(min_len, np.nan, dtype=float)
                    plot_flag = (not args.no_plot) and (lev in args.layers)
                    stats = _plot_tripanel(var, f"_lev{lev}", lon_subset, lat_subset, ai_layer, csv_layer, args.output_dir,
                                           label_ai=label_ai, label_model=label_csv, plot=plot_flag)
                    stats_rows.append({
                        'variable': var,
                        'suffix': f"_lev{lev}",
                        **stats,
                    })

            elif category == 'pft1d' or ('pft' in dims):
                try:
                    da_ai_p = da_ai.transpose('pft', 'gridcell', ...)
                except ValueError:
                    da_ai_p = da_ai.transpose(..., 'pft', 'gridcell')
                ai_vals = _to_nan_fillvalue(da_ai_p.values)
                if ai_vals.ndim == 1:
                    ai_vals = ai_vals[np.newaxis, :]
                csv_vals = extract_csv_pft(csv_predictions, var)
                if csv_vals is None:
                    print(f"  Skip {var} (CSV PFT entries unavailable)")
                    continue
                pft_count = ai_vals.shape[0]
                csv_pft_count = csv_vals.shape[0]
                if csv_vals.shape[1] != ai_vals.shape[1]:
                    min_len = min(ai_vals.shape[1], csv_vals.shape[1], len(grid_lon))
                    if min_len == 0:
                        print(f"  Skip {var} (no grid overlap between NetCDF and CSV)")
                        continue
                    print(f"  Warning: grid mismatch for {var}; trimming to {min_len} cells")
                else:
                    min_len = ai_vals.shape[1]
                lon_subset = grid_lon[:min_len]
                lat_subset = grid_lat[:min_len]
                for k in range(pft_count):
                    ai_slice = ai_vals[k, :min_len]
                    if k < csv_pft_count:
                        csv_slice = csv_vals[k, :min_len]
                    else:
                        csv_slice = np.full(min_len, np.nan, dtype=float)
                    plot_flag = (not args.no_plot) and (k in args.pfts)
                    stats = _plot_tripanel(var, f"_pft{k+1}", lon_subset, lat_subset, ai_slice, csv_slice, args.output_dir,
                                           label_ai=label_ai, label_model=label_csv, plot=plot_flag)
                    stats_rows.append({
                        'variable': var,
                        'suffix': f"_pft{k+1}",
                        **stats,
                    })

            elif 'gridcell' in dims:
                try:
                    da_ai_gc = da_ai.transpose(..., 'gridcell')
                except ValueError:
                    da_ai_gc = da_ai
                ai_vals = _to_nan_fillvalue(np.asarray(da_ai_gc.values))
                ai_vals = np.reshape(ai_vals, (-1, ai_vals.shape[-1])) if ai_vals.ndim > 1 else ai_vals
                ai_vals = ai_vals[-1] if ai_vals.ndim > 1 else ai_vals
                csv_vals = extract_csv_scalar(csv_predictions, var)
                if csv_vals is None:
                    print(f"  Skip {var} (CSV scalar column unavailable)")
                    continue
                csv_vals = np.asarray(csv_vals, dtype=float)
                min_len = min(len(ai_vals), len(csv_vals), len(grid_lon))
                if min_len == 0:
                    print(f"  Skip {var} (no overlapping records)")
                    continue
                if len(ai_vals) != len(csv_vals):
                    print(f"  Warning: record mismatch for {var}; trimming to {min_len}")
                lon_subset = grid_lon[:min_len]
                lat_subset = grid_lat[:min_len]
                ai_trim = ai_vals[:min_len]
                csv_trim = csv_vals[:min_len]
                stats = _plot_tripanel(var, '', lon_subset, lat_subset, ai_trim, csv_trim, args.output_dir,
                                       label_ai=label_ai, label_model=label_csv, plot=(not args.no_plot))
                stats_rows.append({
                    'variable': var,
                    'suffix': '',
                    **stats,
                })
            else:
                print(f"  Skip {var} (unsupported dimensions for CSV comparison: {dims})")

        else:
            if var not in ds_model.data_vars:
                print(f"Skip {var} (not found in model file)")
                continue

            da_model = ds_model[var]
            print(f"  Model shape: {da_model.shape}")

            if ('column' in dims) and ('levgrnd' in dims):
                if len(dims) == 3:
                    if 'gridcell' in dims:
                        da_ai_cl = da_ai.transpose('column', 'levgrnd', ...)
                        da_model_cl = da_model.transpose('column', 'levgrnd', ...)
                    else:
                        da_ai_cl = da_ai.transpose('column', 'levgrnd')
                        da_model_cl = da_model.transpose('column', 'levgrnd')
                else:
                    da_ai_cl = da_ai.transpose('column', 'levgrnd')
                    da_model_cl = da_model.transpose('column', 'levgrnd')

                vals_ai = _to_nan_fillvalue(da_ai_cl.values)
                vals_model = _to_nan_fillvalue(da_model_cl.values)

                print(f"  Column variable: AI shape {vals_ai.shape}, Model shape {vals_model.shape}")
                print(f"  Grid mapping: {len(grid_to_cols)} gridcells with columns")
                print(f"  Sample gridcell 0 has columns: {grid_to_cols[0][:5] if grid_to_cols[0] else 'none'}")

                total_layers = int(da_ai_cl.sizes['levgrnd']) if 'levgrnd' in da_ai_cl.sizes else 0
                for lev in range(total_layers):
                    if lev < 0 or lev >= total_layers:
                        continue

                    ai_grid = np.full(n_grid, np.nan, dtype=float)
                    model_grid = np.full(n_grid, np.nan, dtype=float)

                    for g in range(n_grid):
                        cols = grid_to_cols[g]
                        if len(cols) == 0:
                            continue
                        ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                        if len(ai_gridcell_idx) > 0:
                            ai_gridcell_idx = ai_gridcell_idx[0]
                            ai_col_idx = 0
                            if vals_ai.ndim == 3:
                                ai_grid[g] = vals_ai[ai_col_idx, lev, ai_gridcell_idx]
                            else:
                                ai_grid[g] = vals_ai[ai_col_idx, lev]
                        else:
                            ai_grid[g] = np.nan

                        if g < len(grid_to_cols) and len(grid_to_cols[g]) > 0:
                            model_col_idx = grid_to_cols[g][0]
                            if model_col_idx < vals_model.shape[0]:
                                if vals_model.ndim == 2:
                                    if lev < vals_model.shape[1]:
                                        model_grid[g] = vals_model[model_col_idx, lev]
                                else:
                                    model_grid[g] = vals_model[model_col_idx]

                    do_plot = (not args.no_plot) and (lev in args.layers)
                    stats = _plot_tripanel(var, f"_lev{lev}", grid_lon, grid_lat, ai_grid, model_grid, args.output_dir,
                                           label_ai='AI Predictions', label_model='Model Results', plot=do_plot)
                    stats_rows.append({
                        'variable': var,
                        'suffix': f"_lev{lev}",
                        **stats,
                    })

            elif ('pft' in dims):
                if len(dims) == 2 and 'gridcell' in dims:
                    da_ai_p = da_ai.transpose('pft', ...)
                    da_model_p = da_model.transpose('pft', ...)
                else:
                    da_ai_p = da_ai.transpose('pft')
                    da_model_p = da_model.transpose('pft')

                vals_ai = _to_nan_fillvalue(da_ai_p.values)
                vals_model = _to_nan_fillvalue(da_model_p.values)

                total_pfts = vals_ai.shape[0]
                for k in range(total_pfts):
                    ai_grid = np.full(n_grid, np.nan, dtype=float)
                    model_grid = np.full(n_grid, np.nan, dtype=float)

                    if vals_ai.ndim == 2:
                        for g in range(n_grid):
                            ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                            if len(ai_gridcell_idx) > 0:
                                ai_gridcell_idx = ai_gridcell_idx[0]
                                ai_grid[g] = vals_ai[k, ai_gridcell_idx]
                    else:
                        for g in range(n_grid):
                            ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                            if len(ai_gridcell_idx) > 0:
                                ai_grid[g] = vals_ai[k]

                    for g in range(n_grid):
                        if g < len(grid_to_pfts) and len(grid_to_pfts[g]) > 0:
                            gridcell_pfts = grid_to_pfts[g][:16]
                            adjusted_k = k + 1
                            if adjusted_k < len(gridcell_pfts):
                                model_pft_idx = gridcell_pfts[adjusted_k]
                                if model_pft_idx < vals_model.shape[0]:
                                    model_grid[g] = vals_model[model_pft_idx]

                    do_plot = (not args.no_plot) and (k in args.pfts)
                    stats = _plot_tripanel(var, f"_pft{k+1}", grid_lon, grid_lat, ai_grid, model_grid, args.output_dir,
                                           label_ai='AI Predictions', label_model='Model Results', plot=do_plot)
                    stats_rows.append({
                        'variable': var,
                        'suffix': f"_pft{k+1}",
                        **stats,
                    })

            elif 'gridcell' in dims:
                if len(dims) == 1:
                    da_ai_gc = da_ai
                    da_model_gc = da_model
                else:
                    da_ai_gc = da_ai.transpose(..., 'gridcell')
                    da_model_gc = da_model.transpose(..., 'gridcell')

                vals_ai = _to_nan_fillvalue(da_ai_gc.values)
                vals_model = _to_nan_fillvalue(da_model_gc.values)

                ai_grid = np.full(n_grid, np.nan, dtype=float)

                if vals_ai.ndim == 1:
                    for g in range(n_grid):
                        ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                        if len(ai_gridcell_idx) > 0:
                            ai_gridcell_idx = ai_gridcell_idx[0]
                            ai_grid[g] = vals_ai[ai_gridcell_idx]
                else:
                    for g in range(n_grid):
                        ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                        if len(ai_gridcell_idx) > 0:
                            ai_gridcell_idx = ai_gridcell_idx[0]
                            if vals_ai.ndim == 2:
                                ai_grid[g] = vals_ai[0, ai_gridcell_idx]
                            else:
                                ai_grid[g] = vals_ai[ai_gridcell_idx]

                model_grid = vals_model

                stats = _plot_tripanel(var, '', grid_lon, grid_lat, ai_grid, model_grid, args.output_dir,
                                       label_ai='AI Predictions', label_model='Model Results', plot=(not args.no_plot))
                stats_rows.append({
                    'variable': var,
                    'suffix': '',
                    **stats,
                })

            else:
                print(f"  Skip {var} (unsupported dimensions: {dims})")

    # Write statistics outputs (CSV/TXT)
    if stats_rows:
        csv_out = args.stats_file if (args.stats_file and args.stats_file.lower().endswith('.csv')) else os.path.join(args.output_dir, 'stats.csv')
        txt_out = args.stats_file if (args.stats_file and args.stats_file.lower().endswith('.txt')) else os.path.join(args.output_dir, 'stats.txt')
        os.makedirs(args.output_dir, exist_ok=True)

        if args.stats_format in ('csv', 'both'):
            fieldnames = [
                'variable', 'suffix', 'ai_sum', 'ai_std', 'ai_min', 'ai_max',
                'model_sum', 'model_std', 'model_min', 'model_max', 'n', 'rmse', 'nrmse', 'r2'
            ]
            with open(csv_out, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in stats_rows:
                    writer.writerow(row)
            print(f"Saved statistics CSV: {csv_out}")

        if args.stats_format in ('txt', 'both'):
            from collections import defaultdict
            grouped = defaultdict(list)
            for row in stats_rows:
                grouped[row['variable']].append(row)

            def _suffix_key(s):
                if s == '':
                    return (0, 0, 0)
                if s.startswith('_lev'):
                    try:
                        return (1, int(s.replace('_lev', '')), 0)
                    except Exception:
                        return (1, 999999, 0)
                if s.startswith('_pft'):
                    try:
                        return (2, int(s.replace('_pft', '')), 0)
                    except Exception:
                        return (2, 999999, 0)
                return (3, 0, 0)

            lines = []
            header = 'NetCDF vs CSV Statistics Report' if use_csv_predictions else 'AI vs Model Statistics Report'
            lines.append(header)
            lines.append('=' * 80)
            for var in sorted(grouped.keys()):
                rows = sorted(grouped[var], key=lambda r: _suffix_key(r.get('suffix', '')))
                lines.append('')
                lines.append(f"Variable: {var}")
                lines.append('-' * 80)
                for row in rows:
                    title = f"{var}{row.get('suffix','')}"
                    lines.append(title)
                    if use_csv_predictions:
                        lines.append("  NetCDF: sum={ai_sum:.6g} std={ai_std:.6g} min={ai_min:.6g} max={ai_max:.6g}".format(**row))
                        lines.append("  CSV:    sum={model_sum:.6g} std={model_std:.6g} min={model_min:.6g} max={model_max:.6g}".format(**row))
                    else:
                        lines.append("  AI:    sum={ai_sum:.6g} std={ai_std:.6g} min={ai_min:.6g} max={ai_max:.6g}".format(**row))
                        lines.append("  Model: sum={model_sum:.6g} std={model_std:.6g} min={model_min:.6g} max={model_max:.6g}".format(**row))
                    lines.append("  Compare: n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}".format(**row))
                    lines.append('')

            with open(txt_out, 'w') as f:
                f.write("\n".join(lines))
            print(f"Saved statistics report: {txt_out}")

        # Generate a stacked bar chart summarizing quality by variable (CSV vs NetCDF)
        try:
            stats_df = pd.DataFrame(stats_rows)
            if not stats_df.empty:
                stats_df['quality'] = stats_df['r2'].apply(_classify_quality_from_r2)
                quality_counts = (
                    stats_df.groupby(['variable', 'quality']).size().unstack(fill_value=0)
                )
                if not quality_counts.empty:
                    totals = quality_counts.sum(axis=1).replace(0, np.nan)
                    quality_pct = (quality_counts.div(totals, axis=0) * 100.0).fillna(0.0)
                    desired_order = ['good', 'ok', 'bad', 'unknown']
                    available_cols = [c for c in desired_order if c in quality_pct.columns]
                    missing_cols = [c for c in desired_order if c not in available_cols]
                    for col in missing_cols:
                        quality_pct[col] = 0.0
                    quality_pct = quality_pct[available_cols + missing_cols] if missing_cols else quality_pct[available_cols]
                    if 'good' in quality_pct.columns:
                        quality_pct = quality_pct.sort_values(by='good', ascending=False)
                    colors = {
                        'good': '#2ecc71',
                        'ok': '#f39c12',
                        'bad': '#e74c3c',
                        'unknown': '#7f8c8d'
                    }
                    plot_cols = [c for c in ['good', 'ok', 'bad', 'unknown'] if c in quality_pct.columns]
                    if plot_cols:
                        ax = quality_pct[plot_cols].plot(
                            kind='bar',
                            stacked=True,
                            figsize=(14, 10),
                            color=[colors.get(col, 'gray') for col in plot_cols]
                        )
                        title = 'CSV vs NetCDF Agreement by Variable' if use_csv_predictions else 'AI vs Model Agreement by Variable'
                        plt.title(title, fontsize=16)
                        plt.xlabel('Variable', fontsize=14)
                        plt.ylabel('Percentage (%)', fontsize=14)
                        plt.xticks(rotation=90)
                        plt.legend(title='Category')
                        plt.tight_layout()
                        quality_fig = Path(args.output_dir) / 'comparison_quality_by_variable.png'
                        plt.savefig(quality_fig, dpi=300)
                        plt.close()
                        print(f"Saved quality summary figure: {quality_fig}")
        except Exception as exc:
            print(f"Warning: Failed to generate quality summary figure ({exc})")
    else:
        print("No statistics to write.")

    ds_ai.close()
    if ds_model is not None:
        ds_model.close()
    if args.no_plot:
        print(f"\nCompleted without plotting. Output directory: {args.output_dir}")
    else:
        print(f"\nAll plots done! Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()