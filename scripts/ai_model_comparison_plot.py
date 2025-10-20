#!/usr/bin/env python3
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
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

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list

# Default paths
DEFAULT_AI_PREDICTIONS = './comparison_results/ai_predictions_for_plotting.nc'
DEFAULT_MODEL = '/global/cfs/cdirs/m4814/daweigao/14_Code/all_dataset_1_degree/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc'
DEFAULT_OUTPUT_DIR = "./ai_model_comparison_plots"

# Default variables to plot unless '--variables all' is used
VARIABLES = ['cwdc_vr', 'soil3c_vr', 'tlai', 'deadstemc']

# Layers for column-type variables (AI has 10 layers, model has 15)
LEVGRND_LAYERS = [0, 4, 9]  # Layers 0, 4, 9 (corresponding to AI layers 1, 5, 10)

# PFTs to plot (AI has PFT1-16, model has PFT0-16)
# Note: AI PFT0 = Model PFT1, AI PFT1 = Model PFT2, etc.
PFT_PICK_LIST = [0, 1, 2, 3, 4]  # PFT1, PFT2, PFT3, PFT4, PFT5 (0-indexed, so 0=PFT1, 1=PFT2, etc.)

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
    
    parser.add_argument('--ai-predictions', default=DEFAULT_AI_PREDICTIONS,
                       help=f'Path to AI predictions NetCDF file [default: {DEFAULT_AI_PREDICTIONS}]')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help=f'Path to model results NetCDF file [default: {DEFAULT_MODEL}]')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for plots [default: {DEFAULT_OUTPUT_DIR}]')
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
    parser.add_argument('--stats-only', action='store_true',
                       help='Only compute statistics and write CSV (sum/std/min/max per variable and layer/PFT); saves into output_dir/stats')
    
    args = parser.parse_args()
    # If stats-only is requested, force no-plot and CSV output into a dedicated stats folder
    if getattr(args, 'stats_only', False):
        args.no_plot = True
        # Route outputs to a stats subfolder for cleaner organization
        # The final output_dir will be resolved below after potential variable-list handling
        _stats_only_requested = True
    else:
        _stats_only_requested = False
    
    # Validate input files
    if not Path(args.ai_predictions).exists():
        raise FileNotFoundError(f"AI predictions file not found: {args.ai_predictions}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    print("="*60)
    print("AI vs Model Comparison")
    print("="*60)
    print(f"AI predictions: {args.ai_predictions}")
    print(f"Model results: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variables to plot: {args.variables}")
    print(f"Layers to plot: {args.layers}")
    print(f"PFTs to plot: {args.pfts}")
    print("="*60)
    
    # Open datasets once
    ds_ai = xr.open_dataset(args.ai_predictions)
    ds_model = xr.open_dataset(args.model)

    # Determine variable selection behavior
    requested_all = False
    if args.variables:
        requested_all = (len(args.variables) == 1 and str(args.variables[0]).lower() == 'all')
    # In stats-only mode, if a variable list is provided, treat as "all" variables from the list
    if getattr(args, 'stats_only', False) and args.variable_list:
        requested_all = True

    if requested_all:
        if args.variable_list:
            if not Path(args.variable_list).exists():
                ds_ai.close(); ds_model.close()
                raise FileNotFoundError(f"Variable list file not found: {args.variable_list}")
            # Parse the variable list file to get all variables
            all_variables = parse_variable_list_file(args.variable_list)
            # Filter to only include variables that exist in both datasets
            ai_vars = set(ds_ai.data_vars.keys())
            model_vars = set(ds_model.data_vars.keys())
            available_vars = [var for var in all_variables if var in ai_vars and var in model_vars]
            if available_vars:
                args.variables = available_vars
                print(f"Using all variables from variable list ({len(available_vars)}): {available_vars}")
            else:
                print("Warning: No variables from variable list found in both datasets!")
                ds_ai.close(); ds_model.close()
                return
        else:
            # Discover all common variables across datasets
            discovered_vars = discover_common_variables(ds_ai, ds_model)
            if discovered_vars:
                args.variables = discovered_vars
                print(f"Using all common variables ({len(discovered_vars)}): {discovered_vars}")
            else:
                print("Warning: No common variables found between AI predictions and model!")
                ds_ai.close(); ds_model.close()
                return
    else:
        # Force the script to only plot the default subset unless 'all' is requested
        forced = ['cwdc_vr', 'soil3c_vr', 'tlai', 'deadstemc']
        ai_vars = set(ds_ai.data_vars.keys())
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
    
    # Get grid information from the MODEL file as the master coordinate system
    # This ensures AI predictions can be properly ingested into the model
    grid_lon, grid_lat = _gridcell_lonlat(ds_model)
    n_grid = ds_model.sizes["gridcell"]
    
    print(f"Using MODEL gridcell count: {n_grid}")
    print(f"Model coordinates: lon range [{grid_lon.min():.3f}, {grid_lon.max():.3f}], lat range [{grid_lat.min():.3f}, {grid_lat.max():.3f}]")
    
    # Build mappings from the model file for extracting model data
    col2grid = _to_zero_based_index(_safe_get(ds_model, "cols1d_gridcell_index").values, n_grid)
    pft2grid = _to_zero_based_index(_safe_get(ds_model, "pfts1d_gridcell_index").values, n_grid)
    
    grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
    grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)
    
    # Create spatial mapping from AI gridcells to model gridcells
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
    
    # If stats-only, place outputs under a dedicated stats subdirectory
    if _stats_only_requested:
        output_dir = output_dir / 'stats'
        os.makedirs(output_dir, exist_ok=True)

    # Update the output directory for the plotting function
    args.output_dir = str(output_dir)
    
    print(f"\nStart processing: {len(args.variables)} variables")
    stats_rows = []
    for var in args.variables:
        if (var not in ds_ai.data_vars) or (var not in ds_model.data_vars):
            print(f"Skip {var} (not found in both files)")
            continue

        da_ai = ds_ai[var]
        da_model = ds_model[var]
        dims = da_ai.dims
        print(f"\nVariable {var}, dims: {dims}")
        print(f"  AI shape: {da_ai.shape}")
        print(f"  Model shape: {da_model.shape}")

        if ("column" in dims) and ("levgrnd" in dims):
            # Handle column-type variables (e.g., soil variables)
            # Handle different dimension orders
            if len(dims) == 3:
                # If we have (column, levgrnd, gridcell) or similar, transpose to (column, levgrnd)
                if "gridcell" in dims:
                    da_ai_cl = da_ai.transpose("column", "levgrnd", ...)
                    da_model_cl = da_model.transpose("column", "levgrnd", ...)
                else:
                    da_ai_cl = da_ai.transpose("column", "levgrnd")
                    da_model_cl = da_model.transpose("column", "levgrnd")
            else:
                da_ai_cl = da_ai.transpose("column", "levgrnd")
                da_model_cl = da_model.transpose("column", "levgrnd")
            
            vals_ai = _to_nan_fillvalue(da_ai_cl.values)
            vals_model = _to_nan_fillvalue(da_model_cl.values)
            
            # Debug: Print data structure information
            print(f"  Column variable: AI shape {vals_ai.shape}, Model shape {vals_model.shape}")
            print(f"  Grid mapping: {len(grid_to_cols)} gridcells with columns")
            print(f"  Sample gridcell 0 has columns: {grid_to_cols[0][:5] if grid_to_cols[0] else 'none'}")

            # Compute statistics for all layers; only plot selected layers
            total_layers = int(da_ai_cl.sizes["levgrnd"]) if "levgrnd" in da_ai_cl.sizes else 0
            for lev in range(total_layers):
                if lev < 0 or lev >= total_layers:
                    continue

                ai_grid = np.full(n_grid, np.nan, dtype=float)
                model_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    cols = grid_to_cols[g]
                    if len(cols) == 0:
                        continue
                    
                    # For AI data: map from model gridcell g to corresponding AI gridcell
                    ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                    if len(ai_gridcell_idx) > 0:
                        ai_gridcell_idx = ai_gridcell_idx[0]
                        ai_col_idx = 0
                        # Handle AI data indexing - shape is (column, levgrnd, gridcell)
                        if vals_ai.ndim == 3:
                            ai_grid[g] = vals_ai[ai_col_idx, lev, ai_gridcell_idx]
                        else:
                            ai_grid[g] = vals_ai[ai_col_idx, lev]
                    else:
                        ai_grid[g] = np.nan
                    
                    # For model: use the first column of this gridcell
                    if g < len(grid_to_cols) and len(grid_to_cols[g]) > 0:
                        model_col_idx = grid_to_cols[g][0]
                        if model_col_idx < vals_model.shape[0]:
                            if vals_model.ndim == 2:
                                if lev < vals_model.shape[1]:
                                    model_grid[g] = vals_model[model_col_idx, lev]
                            else:
                                model_grid[g] = vals_model[model_col_idx]

                # Plot only if requested layer in args.layers and plotting enabled
                do_plot = (not args.no_plot) and (lev in args.layers)
                stats = _plot_tripanel(var, f"_lev{lev}", grid_lon, grid_lat, ai_grid, model_grid, args.output_dir,
                               label_ai="AI Predictions", label_model="Model Results", plot=do_plot)
                stats_rows.append({
                    "variable": var,
                    "suffix": f"_lev{lev}",
                    **stats,
                })

        elif ("pft" in dims):
            # Handle PFT-type variables
            # Handle different dimension orders
            if len(dims) == 2 and "gridcell" in dims:
                # If we have (pft, gridcell), transpose to (pft, ...)
                da_ai_p = da_ai.transpose("pft", ...)
                da_model_p = da_model.transpose("pft", ...)
            else:
                da_ai_p = da_ai.transpose("pft")
                da_model_p = da_model.transpose("pft")
            
            vals_ai = _to_nan_fillvalue(da_ai_p.values)
            vals_model = _to_nan_fillvalue(da_model_p.values)

            # Compute statistics for all PFTs in AI data; only plot selected ones
            total_pfts = vals_ai.shape[0]
            for k in range(total_pfts):
                ai_grid = np.full(n_grid, np.nan, dtype=float)
                model_grid = np.full(n_grid, np.nan, dtype=float)

                # Map AI data for PFT k to model gridcells
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

                # Model data: use first 16 PFTs per gridcell; AI PFT k -> Model PFT (k+1)
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
                               label_ai="AI Predictions", label_model="Model Results", plot=do_plot)
                stats_rows.append({
                    "variable": var,
                    "suffix": f"_pft{k+1}",
                    **stats,
                })

        elif "gridcell" in dims:
            # Handle gridcell-level variables (e.g., GPP, NPP)
            # Handle different dimension orders
            if len(dims) == 1:
                da_ai_gc = da_ai
                da_model_gc = da_model
            else:
                # If we have multiple dimensions including gridcell, transpose to put gridcell last
                da_ai_gc = da_ai.transpose(..., "gridcell")
                da_model_gc = da_model.transpose(..., "gridcell")
            
            vals_ai = _to_nan_fillvalue(da_ai_gc.values)
            vals_model = _to_nan_fillvalue(da_model_gc.values)
            
            # Map AI data to model gridcell positions using spatial mapping
            ai_grid = np.full(n_grid, np.nan, dtype=float)
            
            if vals_ai.ndim == 1:
                # For each model gridcell, find the corresponding AI gridcell and extract data
                for g in range(n_grid):
                    ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                    if len(ai_gridcell_idx) > 0:
                        ai_gridcell_idx = ai_gridcell_idx[0]  # Take the first match
                        ai_grid[g] = vals_ai[ai_gridcell_idx]
            else:
                # Handle multi-dimensional AI data
                for g in range(n_grid):
                    ai_gridcell_idx = np.where(ai_to_model_mapping == g)[0]
                    if len(ai_gridcell_idx) > 0:
                        ai_gridcell_idx = ai_gridcell_idx[0]  # Take the first match
                        # Extract data for this gridcell (handle different dimension orders)
                        if vals_ai.ndim == 2:
                            ai_grid[g] = vals_ai[0, ai_gridcell_idx]  # Assume first dimension is not gridcell
                        else:
                            ai_grid[g] = vals_ai[ai_gridcell_idx]
            
            # Model data is already in the correct gridcell order
            model_grid = vals_model
            
            stats = _plot_tripanel(var, "", grid_lon, grid_lat, ai_grid, model_grid, args.output_dir,
                           label_ai="AI Predictions", label_model="Model Results", plot=(not args.no_plot))
            stats_rows.append({
                "variable": var,
                "suffix": "",
                **stats,
            })

        else:
            print(f"  Skip {var} (unsupported dimensions: {dims})")

    # Write statistics outputs (CSV/TXT)
    if stats_rows:
        # Resolve output paths
        if _stats_only_requested:
            # In stats-only mode, write a concise CSV with the requested metrics under stats folder
            csv_out = os.path.join(args.output_dir, "summary_stats.csv")
            txt_out = None
            output_mode = 'csv'
        else:
            csv_out = args.stats_file if (args.stats_file and args.stats_file.lower().endswith('.csv')) else os.path.join(args.output_dir, "stats.csv")
            txt_out = args.stats_file if (args.stats_file and args.stats_file.lower().endswith('.txt')) else os.path.join(args.output_dir, "stats.txt")
            output_mode = args.stats_format
        os.makedirs(args.output_dir, exist_ok=True)

        # CSV output
        if (output_mode in ('csv', 'both')):
            if _stats_only_requested:
                fieldnames = [
                    "variable", "suffix",
                    "ai_sum", "ai_std", "ai_min", "ai_max",
                    "model_sum", "model_std", "model_min", "model_max"
                ]
                # Reduce rows to requested columns only
                filtered_rows = []
                for row in stats_rows:
                    filtered_rows.append({
                        "variable": row.get("variable"),
                        "suffix": row.get("suffix"),
                        "ai_sum": row.get("ai_sum"),
                        "ai_std": row.get("ai_std"),
                        "ai_min": row.get("ai_min"),
                        "ai_max": row.get("ai_max"),
                        "model_sum": row.get("model_sum"),
                        "model_std": row.get("model_std"),
                        "model_min": row.get("model_min"),
                        "model_max": row.get("model_max"),
                    })
                rows_to_write = filtered_rows
            else:
                fieldnames = [
                    "variable", "suffix", "ai_sum", "ai_min", "ai_max",
                    "model_sum", "model_min", "model_max", "n", "rmse", "nrmse", "r2"
                ]
                rows_to_write = stats_rows
            with open(csv_out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows_to_write:
                    writer.writerow(row)
            print(f"Saved statistics CSV: {csv_out}")

        # TXT output (readable, grouped by variable and suffix)
        if (not _stats_only_requested) and (output_mode in ('txt', 'both')):
            # Group stats by variable then suffix
            from collections import defaultdict
            grouped = defaultdict(list)
            for row in stats_rows:
                grouped[row["variable"]].append(row)

            def _suffix_key(s):
                # Sort order: gridcell-level (empty) first, then lev by number, then pft by number
                if s == "":
                    return (0, 0, 0)
                if s.startswith("_lev"):
                    try:
                        return (1, int(s.replace("_lev", "")), 0)
                    except Exception:
                        return (1, 999999, 0)
                if s.startswith("_pft"):
                    try:
                        return (2, int(s.replace("_pft", "")), 0)
                    except Exception:
                        return (2, 999999, 0)
                return (3, 0, 0)

            lines = []
            lines.append("AI vs Model Statistics Report")
            lines.append("=" * 80)
            for var in sorted(grouped.keys()):
                rows = sorted(grouped[var], key=lambda r: _suffix_key(r.get("suffix", "")))
                lines.append("")
                lines.append(f"Variable: {var}")
                lines.append("-" * 80)
                for row in rows:
                    title = f"{var}{row.get('suffix','')}"
                    lines.append(title)
                    lines.append("  AI:    sum={ai_sum:.6g} min={ai_min:.6g} max={ai_max:.6g}".format(**row))
                    lines.append("  Model: sum={model_sum:.6g} min={model_min:.6g} max={model_max:.6g}".format(**row))
                    lines.append("  Compare: n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}".format(**row))
                    lines.append("")

            with open(txt_out, "w") as f:
                f.write("\n".join(lines))
            print(f"Saved statistics report: {txt_out}")
    else:
        print("No statistics to write.")

    ds_ai.close()
    ds_model.close()
    if args.no_plot:
        print(f"\nCompleted without plotting. Output directory (for CSV): {args.output_dir}")
    else:
        print(f"\nAll plots done! Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
