import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
import argparse
import glob
from pathlib import Path
import sys
import json
from scipy.spatial.distance import cdist

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list

# Default file paths (fallback values)
FALLBACK_DATA_DIR = '/global/cfs/cdirs/m4814/daweigao/14_Code/all_dataset_1_degree/'
FALLBACK_REFERENCE_FILE = FALLBACK_DATA_DIR + '20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc'
FALLBACK_AI_PREDICTIONS = './comparison_results/ai_predictions_for_plotting.nc'
FALLBACK_OUTPUT_DIR = './ai_restart_comparison_plots'
LABEL_NEW = 'AI Restart'
LABEL_OLD_DEFAULT = 'Original Restart'

QUALITY_THRESHOLDS = {
    'good': 0.9,  # R² >= 0.9 for good quality
    'ok': 0.7     # R² >= 0.7 for ok quality
}


def _classify_quality_from_r2(value):
    """Classify quality based on R² value (similar to generate_prediction_quality_report.py)"""
    if value is None or not np.isfinite(value):
        return 'unknown'
    val = float(value)
    if val >= QUALITY_THRESHOLDS['good']:
        return 'good'
    if val >= QUALITY_THRESHOLDS['ok']:
        return 'ok'
    return 'bad'

def _safe_get(ds, name):
    if name not in ds:
        raise KeyError(f"Missing required variable/coordinate: {name}")
    return ds[name]

def _to_nan_fillvalue(arr, fill_threshold=1e35):
    a = np.asarray(arr, dtype=float)
    a[np.abs(a) >= fill_threshold] = np.nan
    return a

def _gridcell_lonlat(ds):
    lon = _safe_get(ds, "grid1d_lon").values
    lat = _safe_get(ds, "grid1d_lat").values
    return lon, lat

def _to_zero_based_index(idx_raw, n_grid):
    idx = np.asarray(idx_raw, dtype=np.int64).copy()
    if idx.size == 0:
        return np.full_like(idx, -1)
    is_one_based = (np.any(idx == n_grid) or (np.nanmin(idx) == 1))
    if is_one_based:
        idx = idx - 1
    idx[(idx < 0) | (idx >= n_grid)] = -1
    return idx

def _build_gridcell_groups(one_d_to_grid, n_grid):
    groups = [[] for _ in range(n_grid)]
    for idx, g in enumerate(one_d_to_grid):
        if 0 <= g < n_grid:
            groups[g].append(idx)
    return groups

def _build_target_to_source_mapping(ds_source, ds_target):
    src_lon = _safe_get(ds_source, 'grid1d_lon').values
    src_lat = _safe_get(ds_source, 'grid1d_lat').values
    tgt_lon = _safe_get(ds_target, 'grid1d_lon').values
    tgt_lat = _safe_get(ds_target, 'grid1d_lat').values
    if (len(src_lon) == len(tgt_lon) and
            np.allclose(src_lon, tgt_lon) and
            np.allclose(src_lat, tgt_lat)):
        return np.arange(len(tgt_lon), dtype=int)
    mapping = np.argmin(cdist(np.column_stack([tgt_lon, tgt_lat]),
                               np.column_stack([src_lon, src_lat])), axis=1)
    return mapping

def _plot_map(ax, lon, lat, data, title, vmin=None, vmax=None, cmap="viridis", norm=None):
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
    bins[(mag >= 0) & (mag < 10)] = 1
    bins[(mag >= 10) & (mag < 30)] = 2
    bins[mag >= 30] = 3
    bins[(mag == 0)] = 0
    categories = sign.astype(int) * bins
    cat[finite] = categories.astype(float)
    return cat


def _plot_tripanel(var, label_suffix, lon, lat, data_new, data_old, out_dir,
                   label_new="AI Restart", label_old="Original Restart", plot=True):
    diff = data_new - data_old
    vmin_orig = np.nanmin([np.nanmin(data_new), np.nanmin(data_old)])
    vmax_orig = np.nanmax([np.nanmax(data_new), np.nanmax(data_old)])
    diff_abs = np.nanmax(np.abs(diff))

    finite_new = np.isfinite(data_new)
    finite_old = np.isfinite(data_old)
    mask = finite_new & finite_old
    n = int(mask.sum())

    def _nan_min(a):
        return float(np.nanmin(a)) if np.any(np.isfinite(a)) else float('nan')

    def _nan_max(a):
        return float(np.nanmax(a)) if np.any(np.isfinite(a)) else float('nan')

    sum_new = float(np.nansum(data_new))
    std_new = float(np.nanstd(data_new)) if np.any(np.isfinite(data_new)) else float('nan')
    min_new = _nan_min(data_new)
    max_new = _nan_max(data_new)

    sum_old = float(np.nansum(data_old))
    std_old = float(np.nanstd(data_old)) if np.any(np.isfinite(data_old)) else float('nan')
    min_old = _nan_min(data_old)
    max_old = _nan_max(data_old)

    if n > 0:
        y = data_new[mask]
        yhat = data_old[mask]
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
    print(f"  {label_new}: sum={sum_new:.6g} std={std_new:.6g} min={min_new:.6g} max={max_new:.6g}")
    print(f"  {label_old}: sum={sum_old:.6g} std={std_old:.6g} min={min_old:.6g} max={max_old:.6g}")
    print(f"  Metrics ({label_new} vs {label_old}): n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}")

    stats = {
        'new_sum': sum_new,
        'new_std': std_new,
        'new_min': min_new,
        'new_max': max_new,
        'old_sum': sum_old,
        'old_std': std_old,
        'old_min': min_old,
        'old_max': max_old,
        'n': n,
        'rmse': rmse,
        'nrmse': nrmse,
        'r2': r2,
    }

    if not plot:
        return stats

    fig = plt.figure(figsize=(12, 20))
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    _plot_map(ax1, lon, lat, data_new, f"{var} - {label_new}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    _plot_map(ax2, lon, lat, data_old, f"{var} - {label_old}", vmin=vmin_orig, vmax=vmax_orig, cmap="viridis")

    ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    if not np.isfinite(diff_abs) or diff_abs <= 0:
        msg = "No Difference" if diff_abs == 0 else "All NaN"
        ax3.text(0.5, 0.5, msg, ha="center", va="center", transform=ax3.transAxes,
                 fontsize=14, fontweight="bold", color="gray")
        ax3.set_title(f"{var} - Analysis", fontsize=14, fontweight="bold")
        ax3.set_global()
        gl = ax3.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0, vmax=diff_abs)
        _plot_map(ax3, lon, lat, diff, f"{var} - Diff ({label_new} - {label_old})", cmap="RdBu_r", norm=norm)

    ax4 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
    cat = _percent_diff_categories(data_old, data_new)
    colors = [
        "#08519c",
        "#6baed6",
        "#c6dbef",
        "#bdbdbd",
        "#fcbba1",
        "#fb6a4a",
        "#cb181d",
    ]
    cmap = ListedColormap(colors)
    boundaries = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm_cat = BoundaryNorm(boundaries, cmap.N)

    ax4.add_feature(cfeature.COASTLINE)
    ax4.add_feature(cfeature.BORDERS)
    ax4.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax4.add_feature(cfeature.LAND, color="lightgray", alpha=0.3)
    im4 = ax4.scatter(lon, lat, c=cat, s=5, cmap=cmap, norm=norm_cat, transform=ccrs.PlateCarree())
    ax4.set_title(f"{var} - Percent Diff bins (({label_new}-{label_old})/{label_old})", fontsize=14, fontweight="bold")
    ax4.set_global()
    gl4 = ax4.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
    gl4.top_labels = False
    gl4.right_labels = False
    cbar = plt.colorbar(im4, ax=ax4, shrink=0.7, pad=0.05, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels([
        "-30%+", "-10–30%", "-0–10%", "0", "0–10%", "10–30%", "30%+"
    ])

    plt.suptitle(f"{var} {label_suffix}", fontsize=16, fontweight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    path_out = os.path.join(out_dir, f"{var}{label_suffix}.png")
    plt.savefig(path_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path_out}")

    return stats

def _load_default_paths(variable_list_path: str):
    defaults = {}
    if variable_list_path:
        try:
            vl_path = Path(variable_list_path)
            if vl_path.exists():
                parsed = parse_cnp_io_list(variable_list_path)
                if isinstance(parsed, dict):
                    for key in ('ai_predictions_default', 'model_default', 'comparison_output_dir', 'ai_restart_default', 'fallback_data_dir', 'fallback_reference_file', 'fallback_reference_filename'):
                        value = parsed.get(key)
                        if value:
                            defaults[key] = value
        except Exception as exc:
            print(f"Warning: Failed to load default paths from {variable_list_path}: {exc}")
    return defaults

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare AI-enhanced restart file with original model file')
    parser.add_argument('--variable-list', type=str, required=False,
                       help='Path to CNP_IO list file (optional, will auto-detect from config.json if not provided)')
    parser.add_argument('--ai-restart', type=str, default=None,
                       help='Path to AI-enhanced restart file (auto-detected if not specified)')
    parser.add_argument('--original-restart', type=str, default=None,
                       help='Path to reference dataset (original restart or AI predictions)')
    parser.add_argument('--layers', type=str, default='0,3,5',
                       help='Comma-separated list of soil layers to plot (default: 0,3,5)')
    parser.add_argument('--pfts', type=str, default='1,2,4,5',
                       help='Comma-separated list of PFTs to plot (default: all PFT0-PFT15)')
    parser.add_argument('--plot-all', action='store_true',
                       help='If set, plot all variables for all 10 layers (0-9) and all 16 PFTs (1-16, skip pft0)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots/statistics')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only compute statistics (no plots)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--stats-file', type=str, default=None,
                       help='Optional path to write statistics file (csv/txt)')
    parser.add_argument('--stats-format', type=str, choices=['csv', 'txt', 'both'], default='txt',
                       help='Statistics output format (default: txt)')
    return parser.parse_args()

def find_ai_restart_file():
    """Find AI-enhanced restart file in current directory."""
    current_dir = Path('.')
    pattern = '*updated_restart_CNP_IO*'
    matching_files = list(current_dir.glob(pattern))
    
    if matching_files:
        # Return the most recent file
        return str(sorted(matching_files, key=lambda x: x.stat().st_mtime)[-1])
    else:
        return None


def main():
    args = parse_arguments()

    defaults = _load_default_paths(args.variable_list)
    
    # Set FALLBACK_DATA_DIR and FALLBACK_REFERENCE_FILE from config if available
    global FALLBACK_DATA_DIR, FALLBACK_REFERENCE_FILE
    if 'fallback_data_dir' in defaults:
        FALLBACK_DATA_DIR = defaults['fallback_data_dir']
    
    # Priority order for FALLBACK_REFERENCE_FILE:
    # 1. Explicit fallback_reference_file in config
    # 2. fallback_data_dir + fallback_reference_filename in config  
    # 3. fallback_data_dir + default filename
    if 'fallback_reference_file' in defaults:
        FALLBACK_REFERENCE_FILE = defaults['fallback_reference_file']
    elif 'fallback_data_dir' in defaults and 'fallback_reference_filename' in defaults:
        FALLBACK_REFERENCE_FILE = FALLBACK_DATA_DIR + defaults['fallback_reference_filename']
    elif 'fallback_data_dir' in defaults:
        # Fallback: construct reference file from data dir if not explicitly set
        FALLBACK_REFERENCE_FILE = FALLBACK_DATA_DIR + '20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc'

    def _resolve(value, keys, fallback):
        if value:
            return str(value)
        for key in keys:
            val = defaults.get(key)
            if val:
                return str(val)
        return fallback

    if args.stats_only:
        args.no_plot = True
    plot_enabled = not args.no_plot

    ai_restart_path = _resolve(args.ai_restart, ('ai_restart_default',), None)
    if not ai_restart_path:
        ai_restart_path = find_ai_restart_file()
        if not ai_restart_path:
            print('Error: No AI-enhanced restart file found. Please specify with --ai-restart')
            return

    reference_path = _resolve(args.original_restart, ('ai_predictions_default', 'model_default'), FALLBACK_REFERENCE_FILE)
    output_dir_str = _resolve(args.output_dir, ('comparison_output_dir',), FALLBACK_OUTPUT_DIR)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_new = LABEL_NEW
    label_old = LABEL_OLD_DEFAULT
    default_ai_predictions = defaults.get('ai_predictions_default', FALLBACK_AI_PREDICTIONS)
    try:
        if Path(reference_path).resolve() == Path(default_ai_predictions).resolve():
            label_old = 'AI Predictions'
    except Exception:
        pass

    print(f'Using AI-enhanced restart: {ai_restart_path}')
    print(f'Using reference dataset: {reference_path}')
    print(f'Output directory: {output_dir}')

    global LEVGRND_LAYERS, PFT_PICK_LIST
    if args.plot_all:
        LEVGRND_LAYERS = list(range(10))
        PFT_PICK_LIST = list(range(16))
    else:
        if args.stats_only:
            LEVGRND_LAYERS = list(range(10))
            PFT_PICK_LIST = list(range(16))
        else:
            LEVGRND_LAYERS = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
            PFT_PICK_LIST = [int(x.strip()) for x in args.pfts.split(',') if x.strip()]
    print(f'Layers to process: {LEVGRND_LAYERS}')
    print(f'PFTs to process: {PFT_PICK_LIST}')
    print('-' * 80)

    if args.variable_list:
        print('Parsing CNP_IO list...')
        cnp_io_vars = parse_cnp_io_list(Path(args.variable_list))
        pft_1d_variables = cnp_io_vars.get('pft_1d_variables', [])
        variables_2d_soil = cnp_io_vars.get('variables_2d_soil', []) or cnp_io_vars.get('x_list_columns_2d', [])
    else:
        print('Auto-detecting CNP_IO variables from config.json...')
        pft_1d_variables, variables_2d_soil = auto_detect_variable_list_from_config(ai_restart_path)
    variables = pft_1d_variables + variables_2d_soil
    if not variables:
        print('Error: No variables found in CNP_IO list or config.json')
        return
    print(f'Variables to process ({len(variables)}): {variables}')

    ds_new = xr.open_dataset(ai_restart_path)
    ds_old = xr.open_dataset(reference_path)

    grid_mapping = _build_target_to_source_mapping(ds_old, ds_new)

    grid_lon, grid_lat = _gridcell_lonlat(ds_new)
    n_grid = ds_new.sizes['gridcell']

    col2grid = _to_zero_based_index(_safe_get(ds_new, 'cols1d_gridcell_index').values, n_grid)
    pft2grid = _to_zero_based_index(_safe_get(ds_new, 'pfts1d_gridcell_index').values, n_grid)
    grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
    grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)

    # Build reference-side grid groups when reference is a restart file (column/pft indexed, no gridcell on variables)
    n_grid_old = ds_old.sizes['gridcell']
    ref_has_col_index = 'cols1d_gridcell_index' in ds_old
    ref_has_pft_index = 'pfts1d_gridcell_index' in ds_old
    grid_to_cols_old = None
    grid_to_pfts_old = None
    if ref_has_col_index:
        col2grid_old = _to_zero_based_index(_safe_get(ds_old, 'cols1d_gridcell_index').values, n_grid_old)
        grid_to_cols_old = _build_gridcell_groups(col2grid_old, n_grid_old)
    if ref_has_pft_index:
        pft2grid_old = _to_zero_based_index(_safe_get(ds_old, 'pfts1d_gridcell_index').values, n_grid_old)
        grid_to_pfts_old = _build_gridcell_groups(pft2grid_old, n_grid_old)

    print(f'Total gridcells: {n_grid} | total columns: {col2grid.size} | total pfts: {pft2grid.size}')
    print(f'Example: gridcell 0 -> columns {grid_to_cols[0][:5]}, pfts {grid_to_pfts[0][:5]}')
    if grid_to_cols_old is not None:
        print(f'Reference: restart-style (column-indexed); n_grid_old={n_grid_old}')
    if grid_to_pfts_old is not None:
        print(f'Reference: restart-style (pft-indexed)')

    stats_rows = []
    debug_enabled = not args.stats_only


    for var in variables:
        if (var not in ds_new.data_vars) or (var not in ds_old.data_vars):
            print(f'Skip {var} (not found in both files)')
            continue

        da_new = ds_new[var]
        da_old = ds_old[var]
        dims = da_new.dims
        print(f"\nVariable {var}, dims: {dims}")

        if ('column' in dims) and ('levgrnd' in dims):
            da_new_cl = da_new.transpose('column', 'levgrnd', ...)
            vals_new = _to_nan_fillvalue(da_new_cl.values)

            ref_restart_style = ('gridcell' not in da_old.dims) and (grid_to_cols_old is not None)
            if not ref_restart_style and 'gridcell' not in da_old.dims:
                print('  Skip (reference dataset lacks gridcell dimension and not column-indexed restart)')
                continue

            if ref_restart_style:
                # Reference is restart file: (column, levgrnd); use grid_to_cols_old + grid_mapping
                da_old_cl = da_old.transpose('column', 'levgrnd', ...)
                vals_old = _to_nan_fillvalue(da_old_cl.values)
            else:
                old_order = [dim for dim in da_old.dims if dim != 'gridcell'] + ['gridcell']
                da_old_cl = da_old.transpose(*old_order)
                vals_old = _to_nan_fillvalue(da_old_cl.values)
                dims_old = da_old_cl.dims
                axis_grid = dims_old.index('gridcell')
                axis_column = dims_old.index('column') if 'column' in dims_old else None
                axis_lev = dims_old.index('levgrnd') if 'levgrnd' in dims_old else None

            for lev in LEVGRND_LAYERS:
                if lev < 0 or lev >= da_new_cl.sizes['levgrnd']:
                    print(f'  Layer {lev} out of range, skipped')
                    continue
                if not ref_restart_style and lev >= da_old_cl.sizes['levgrnd']:
                    continue

                new_grid = np.full(n_grid, np.nan, dtype=float)
                old_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    cols = grid_to_cols[g]
                    if len(cols) == 0:
                        continue
                    c0 = cols[0]
                    if vals_new.shape[0] == 1:
                        new_grid[g] = vals_new[0, lev]
                    else:
                        new_grid[g] = vals_new[c0, lev]

                    src_g = int(grid_mapping[g]) if g < len(grid_mapping) else -1
                    if src_g < 0 or src_g >= n_grid_old:
                        continue

                    if ref_restart_style:
                        cols_old = grid_to_cols_old[src_g]
                        if len(cols_old) == 0:
                            continue
                        c0_old = cols_old[0]
                        if c0_old < vals_old.shape[0] and lev < vals_old.shape[1]:
                            old_grid[g] = vals_old[c0_old, lev]
                    else:
                        idx = [slice(None)] * vals_old.ndim
                        if axis_column is not None:
                            col_sel = min(c0, vals_old.shape[axis_column] - 1)
                            idx[axis_column] = col_sel
                        if axis_lev is not None:
                            if lev >= vals_old.shape[axis_lev]:
                                continue
                            idx[axis_lev] = lev
                        idx[axis_grid] = src_g
                        old_grid[g] = vals_old[tuple(idx)]

                if debug_enabled:
                    print(f'[DEBUG] {var} lev{lev}: new min={np.nanmin(new_grid)} max={np.nanmax(new_grid)} mean={np.nanmean(new_grid)}')
                    print(f'[DEBUG] {var} lev{lev}: ref min={np.nanmin(old_grid)} max={np.nanmax(old_grid)} mean={np.nanmean(old_grid)}')

                stats = _plot_tripanel(var, f'_lev{lev}', grid_lon, grid_lat, new_grid, old_grid, str(output_dir),
                                       label_new=label_new, label_old=label_old, plot=plot_enabled)
                stats_rows.append({'variable': var, 'suffix': f'_lev{lev}', 'label_new': label_new, 'label_old': label_old, 'quality': _classify_quality_from_r2(stats['r2']), **stats})

        elif 'pft' in dims:
            ref_pft_restart_style = ('gridcell' not in da_old.dims) and (grid_to_pfts_old is not None)
            if not ref_pft_restart_style and 'gridcell' not in da_old.dims:
                print('  Skip (reference dataset lacks gridcell dimension and not pft-indexed restart)')
                continue

            da_new_p = da_new.transpose(..., 'pft')
            vals_new = _to_nan_fillvalue(da_new_p.values)

            if ref_pft_restart_style:
                da_old_p = da_old.transpose('pft', ...)
                vals_old = _to_nan_fillvalue(da_old_p.values)
            else:
                da_old_p = da_old.transpose('pft', 'gridcell')
                vals_old = _to_nan_fillvalue(da_old_p.values)
            total_pfts = da_new_p.sizes.get('pft', vals_new.shape[0])

            for k in PFT_PICK_LIST:
                if k < 0 or k >= total_pfts:
                    print(f'  PFT {k} out of range, skipped')
                    continue
                if args.plot_all and k == 0:
                    continue

                new_grid = np.full(n_grid, np.nan, dtype=float)
                old_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    pfts = grid_to_pfts[g]
                    if len(pfts) == 0:
                        continue
                    if k < len(pfts):
                        p_idx = pfts[k]
                        if p_idx < vals_new.shape[0]:
                            new_grid[g] = vals_new[p_idx]

                    src_g = int(grid_mapping[g]) if g < len(grid_mapping) else -1
                    if ref_pft_restart_style:
                        if src_g < 0 or src_g >= n_grid_old or grid_to_pfts_old is None:
                            continue
                        pfts_old = grid_to_pfts_old[src_g]
                        if k < len(pfts_old):
                            p_idx_old = pfts_old[k]
                            if p_idx_old < vals_old.shape[0]:
                                old_grid[g] = vals_old[p_idx_old]
                    else:
                        if src_g < 0 or src_g >= vals_old.shape[1]:
                            continue
                        ai_pft_idx = k - 1
                        if ai_pft_idx >= 0 and ai_pft_idx < vals_old.shape[0]:
                            old_grid[g] = vals_old[ai_pft_idx, src_g]

                stats = _plot_tripanel(var, f'_pft{k}', grid_lon, grid_lat, new_grid, old_grid, str(output_dir),
                                       label_new=label_new, label_old=label_old, plot=plot_enabled)
                stats_rows.append({'variable': var, 'suffix': f'_pft{k}', 'label_new': label_new, 'label_old': label_old, 'quality': _classify_quality_from_r2(stats['r2']), **stats})
        else:
            print(f'  Skip {var} (unsupported dimensions)')
    ds_new.close()
    ds_old.close()

    if stats_rows:
        stats_format = args.stats_format
        stats_dir = output_dir
        csv_out = None
        txt_out = None
        if args.stats_file:
            stats_path = Path(args.stats_file)
            if stats_path.suffix.lower() == '.csv':
                csv_out = stats_path
            elif stats_path.suffix.lower() == '.txt':
                txt_out = stats_path
            else:
                csv_out = stats_path.with_suffix('.csv')
                txt_out = stats_path.with_suffix('.txt')
        else:
            csv_out = stats_dir / 'restart_stats.csv'
            txt_out = stats_dir / 'restart_stats.txt'

        if stats_format in ('csv', 'both') and csv_out:
            fieldnames = ['variable', 'suffix', 'label_new', 'label_old', 'quality', 'new_sum', 'new_std', 'new_min', 'new_max', 'old_sum', 'old_std', 'old_min', 'old_max', 'n', 'rmse', 'nrmse', 'r2']
            with csv_out.open('w', newline='') as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in stats_rows:
                    writer.writerow(row)
            print(f'Saved statistics CSV: {csv_out}')

        if stats_format in ('txt', 'both') and txt_out:
            from collections import defaultdict
            grouped = defaultdict(list)
            for row in stats_rows:
                grouped[row['variable']].append(row)
            lines = []
            lines.append('AI Restart Comparison Statistics')
            lines.append('=' * 80)
            for var in sorted(grouped.keys()):
                rows = sorted(grouped[var], key=lambda r: r.get('suffix', ''))
                lines.append('')
                lines.append(f'Variable: {var}')
                lines.append('-' * 80)
                for row in rows:
                    lines.append(f"{var}{row.get('suffix', '')}")
                    lines.append(f"  {row['label_new']}: sum={row['new_sum']:.6g} std={row['new_std']:.6g} min={row['new_min']:.6g} max={row['new_max']:.6g}")
                    lines.append(f"  {row['label_old']}: sum={row['old_sum']:.6g} std={row['old_std']:.6g} min={row['old_min']:.6g} max={row['old_max']:.6g}")
                    lines.append(f"  Compare: n={row['n']} rmse={row['rmse']:.6g} nrmse={row['nrmse']:.6g} r2={row['r2']:.6g}")
                    lines.append(f"  Quality: {row['quality']}")
                    lines.append('')
            txt_out.write_text('\n'.join(lines))
            print(f'Saved statistics report: {txt_out}')

        # Generate quality summary figure
        categories = ['good', 'ok', 'bad', 'unknown']
        category_colors = {
            'good': '#2ecc71',
            'ok': '#f39c12',
            'bad': '#e74c3c',
            'unknown': '#7f8c8d'
        }
        from collections import defaultdict
        quality_counts = defaultdict(lambda: {cat: 0 for cat in categories})
        for row in stats_rows:
            cat = row.get('quality', 'unknown') or 'unknown'
            if cat not in categories:
                cat = 'unknown'
            quality_counts[row['variable']][cat] += 1
        labels = sorted(quality_counts.keys())
        totals = [sum(quality_counts[var].values()) for var in labels]
        if labels and any(total > 0 for total in totals):
            percentages = []
            for idx, var in enumerate(labels):
                total = totals[idx]
                pct = {}
                for cat in categories:
                    if total > 0:
                        pct[cat] = quality_counts[var].get(cat, 0) / total * 100.0
                    else:
                        pct[cat] = 0.0
                percentages.append(pct)
            fig_width = max(12.0, len(labels) * 0.4)
            fig, ax = plt.subplots(figsize=(fig_width, 8))
            positions = np.arange(len(labels))
            bottom = np.zeros(len(labels), dtype=float)
            for cat in categories:
                heights = [pct[cat] for pct in percentages]
                ax.bar(positions, heights, bottom=bottom, color=category_colors.get(cat, 'gray'), label=cat.capitalize())
                bottom += heights
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_ylabel('Percentage (%)')
            ax.set_ylim(0, 100)
            ax.set_title('Restart Comparison Quality by Variable')
            ax.legend(title='Category')
            fig.tight_layout()
            quality_fig = output_dir / 'restart_quality_by_variable.png'
            plt.savefig(quality_fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved quality summary figure: {quality_fig}')

    print('\n' + '=' * 80)
    print('AI RESTART COMPARISON SUMMARY:')
    print('=' * 80)
    print(f'Reference dataset: {os.path.abspath(reference_path)}')
    print(f'AI-enhanced restart: {os.path.abspath(ai_restart_path)}')
    print(f'Labels: {label_old} vs {label_new}')
    print(f'Output directory: {output_dir.resolve()}')
    print(f'Variables processed: {variables}')
    print(f'Layers processed: {LEVGRND_LAYERS}')
    print(f'PFTs processed: {PFT_PICK_LIST}')
    print('=' * 80)
    if plot_enabled:
        print('\nAll plots done! Output dir:', output_dir)
    else:
        print('\nPlots disabled. Statistics written to output directory.')

if __name__ == "__main__":
    main()
    
    # Print usage examples
    print("\n" + "="*60)
    print("USAGE EXAMPLES:")
    print("="*60)
    print("1. Basic usage with auto-detection:")
    print("   python ai_restart_comparison.py")
    print()
    print("2. Plot all variables for all 10 layers (0-9) and all 16 PFTs (1-16, skip pft0):")
    print("   python ai_restart_comparison.py --plot-all")
    print()
    print("3. Full custom configuration:")
    print("   python ai_restart_comparison.py --variable-list ../../CNP_IO_demo1.txt --ai-restart ai_file.nc --original-restart orig_file.nc --layers 0,5,9 --pfts 0,1,2,3,4,5")
    print("="*60)