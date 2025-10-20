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

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list

# Default file paths
DATA_DIR = '/global/cfs/cdirs/m4814/daweigao/14_Code/all_dataset_1_degree/'
DEFAULT_FILE_OLD = DATA_DIR + '20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc'

LABEL_NEW = "AI Generated"
LABEL_OLD = "Original Model"
OUTPUT_DIR = "./ai_restart_comparison_plots"

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
                   label_new="AI Enhanced", label_old="Original Model"):
    diff = data_new - data_old
    vmin_orig = np.nanmin([np.nanmin(data_new), np.nanmin(data_old)])
    vmax_orig = np.nanmax([np.nanmax(data_new), np.nanmax(data_old)])
    diff_abs = np.nanmax(np.abs(diff))

    # Compute stats and metrics (old vs new) with NaN safety
    finite_new = np.isfinite(data_new)
    finite_old = np.isfinite(data_old)
    mask = finite_new & finite_old
    n = int(mask.sum())

    def _nan_min(a):
        return float(np.nanmin(a)) if np.any(np.isfinite(a)) else float('nan')

    def _nan_max(a):
        return float(np.nanmax(a)) if np.any(np.isfinite(a)) else float('nan')

    sum_new = float(np.nansum(data_new))
    min_new = _nan_min(data_new)
    max_new = _nan_max(data_new)

    sum_old = float(np.nansum(data_old))
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
    print(f"  {label_new}: sum={sum_new:.6g} min={min_new:.6g} max={max_new:.6g}")
    print(f"  {label_old}: sum={sum_old:.6g} min={min_old:.6g} max={max_old:.6g}")
    print(f"  Metrics (AI vs Model): n={n} rmse={rmse:.6g} nrmse={nrmse:.6g} r2={r2:.6g}")

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

    # Percent-difference categorical map (fourth panel)
    ax4 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
    # Treat data_old as model, data_new as AI
    cat = _percent_diff_categories(data_old, data_new)
    colors = [
        "#08519c",  # -3: 30%+
        "#6baed6",  # -2: 10–30%
        "#c6dbef",  # -1: 0–10%
        "#bdbdbd",  #  0: 0
        "#fcbba1",  # +1: 0–10%
        "#fb6a4a",  # +2: 10–30%
        "#cb181d",  # +3: 30%+
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
    path = os.path.join(out_dir, f"{var}{label_suffix}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def auto_detect_variable_list_from_config(ai_restart_path: str):
    run_dir = Path(ai_restart_path).parent if ai_restart_path else Path('.')
    for parent in [run_dir] + list(run_dir.parents):
        config_path = parent / 'cnp_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                data_info = config.get('data_info', {})
                vars_1d = data_info.get('variables_1d_pft', [])
                # Try both keys for 2d soil variables
                vars_2d = data_info.get('variables_2d_soil', [])
                if not vars_2d:
                    vars_2d = data_info.get('x_list_columns_2d', [])
                print(f"Auto-detected variables from {config_path}")
                print(f"  1D PFT variables: {vars_1d}")
                print(f"  2D soil variables: {vars_2d}")
                return list(vars_1d), list(vars_2d)
            except Exception as e:
                print(f"Warning: Failed to parse {config_path}: {e}")
    print("Warning: Could not auto-detect variable list. No variables will be plotted.")
    return [], []

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare AI-enhanced restart file with original model file')
    parser.add_argument('--variable-list', type=str, required=False,
                       help='Path to CNP_IO list file (optional, will auto-detect from config.json if not provided)')
    parser.add_argument('--ai-restart', type=str, default=None,
                       help='Path to AI-enhanced restart file (auto-detected if not specified)')
    parser.add_argument('--original-restart', type=str, default=None,
                       help='Path to original model restart file (default: 780 year model results)')
    parser.add_argument('--layers', type=str, default='0,3,5',
                       help='Comma-separated list of soil layers to plot (default: 0,3,5)')
    parser.add_argument('--pfts', type=str, default='1,2,4,5',
                       help='Comma-separated list of PFTs to plot (default: all PFT0-PFT15)')
    parser.add_argument('--plot-all', action='store_true',
                       help='If set, plot all variables for all 10 layers (0-9) and all 16 PFTs (1-16, skip pft0)')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only compute statistics (sum/std/min/max) for all variables and all layers/PFTs; no plots')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory for plots or stats (default: {OUTPUT_DIR})')
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
    
    # Resolve output directory (stats-only goes to a stats subfolder)
    out_dir_base = Path(args.output_dir)
    if args.stats_only:
        output_dir = out_dir_base / 'stats'
    else:
        output_dir = out_dir_base
    os.makedirs(output_dir, exist_ok=True)
    # Parse layers and PFTs
    global LEVGRND_LAYERS, PFT_PICK_LIST
    if args.plot_all:
        LEVGRND_LAYERS = list(range(10))  # 0-9
        PFT_PICK_LIST = list(range(16))   # 0-15 (will skip pft0 in plotting)
    else:
        LEVGRND_LAYERS = [int(x.strip()) for x in args.layers.split(',')]
        PFT_PICK_LIST = [int(x.strip()) for x in args.pfts.split(',')]
    
    # Set file paths
    if args.ai_restart:
        FILE_NEW = args.ai_restart
    else:
        FILE_NEW = find_ai_restart_file()
        if not FILE_NEW:
            print("Error: No AI-enhanced restart file found. Please specify with --ai-restart")
            return
    
    if args.original_restart:
        FILE_OLD = args.original_restart
    else:
        FILE_OLD = DEFAULT_FILE_OLD
    
    print(f"Using AI-enhanced restart: {FILE_NEW}")
    print(f"Using original restart: {FILE_OLD}")
    print(f"Layers to plot: {LEVGRND_LAYERS}")
    print(f"PFTs to plot: {PFT_PICK_LIST}")
    print("-" * 80)
    
    # Parse CNP_IO list to get variables
    if args.variable_list:
        print("Parsing CNP_IO list...")
        cnp_io_vars = parse_cnp_io_list(Path(args.variable_list))
        pft_1d_variables = cnp_io_vars.get('pft_1d_variables', [])
        variables_2d_soil = cnp_io_vars.get('variables_2d_soil', [])
        if not variables_2d_soil:
            variables_2d_soil = cnp_io_vars.get('x_list_columns_2d', [])
        print(f"  PFT1D: {pft_1d_variables}")
        print(f"  Soil2D: {variables_2d_soil}")
    else:
        print("Auto-detecting CNP_IO variables from config.json...")
        pft_1d_variables, variables_2d_soil = auto_detect_variable_list_from_config(FILE_NEW)
    # Combine all variables to plot
    VARIABLES = pft_1d_variables + variables_2d_soil
    if not VARIABLES:
        print("Error: No variables found in CNP_IO list or config.json")
        return
    print(f"Variables to plot: {VARIABLES}")
    print(f"  PFT1D: {pft_1d_variables}")
    print(f"  Soil2D: {variables_2d_soil}")
    
    ds_new = xr.open_dataset(FILE_NEW)
    ds_old = xr.open_dataset(FILE_OLD)

    grid_lon, grid_lat = _gridcell_lonlat(ds_new)
    n_grid = ds_new.sizes["gridcell"]

    col2grid = _to_zero_based_index(_safe_get(ds_new, "cols1d_gridcell_index").values, n_grid)
    pft2grid = _to_zero_based_index(_safe_get(ds_new, "pfts1d_gridcell_index").values, n_grid)

    grid_to_cols = _build_gridcell_groups(col2grid, n_grid)
    grid_to_pfts = _build_gridcell_groups(pft2grid, n_grid)

    print(f"Total gridcells: {n_grid} | total columns: {col2grid.size} | total pfts: {pft2grid.size}")
    print(f"Example: gridcell 0 -> columns {grid_to_cols[0][:5]}, pfts {grid_to_pfts[0][:5]}")

    stats_rows = []
    print(f"\nStart {'statistics' if args.stats_only else 'plotting'}: {len(VARIABLES)} variables")
    for var in VARIABLES:
        if (var not in ds_new.data_vars) or (var not in ds_old.data_vars):
            print(f"Skip {var} (not found in both files)")
            continue

        da_new = ds_new[var]
        da_old = ds_old[var]
        dims = da_new.dims
        print(f"\nVariable {var}, dims: {dims}")

        if ("column" in dims) and ("levgrnd" in dims):
            # Soil2D variable
            da_new_cl = da_new.transpose("column", "levgrnd")
            da_old_cl = da_old.transpose("column", "levgrnd")
            vals_new = _to_nan_fillvalue(da_new_cl.values)
            vals_old = _to_nan_fillvalue(da_old_cl.values)

            # Iterate all layers if stats-only; otherwise iterate requested layers
            if args.stats_only:
                lev_iter = range(int(da_new_cl.sizes["levgrnd"]))
            else:
                lev_iter = LEVGRND_LAYERS

            for lev in lev_iter:
                if lev < 0 or lev >= da_new_cl.sizes["levgrnd"]:
                    print(f"  Layer {lev} out of range, skipped")
                    continue

                new_grid = np.full(n_grid, np.nan, dtype=float)
                old_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    cols = grid_to_cols[g]
                    if len(cols) == 0:
                        continue
                    c0 = cols[0]  # First column only
                    # Use column 0 for AI prediction file if it has only one column
                    if vals_new.shape[0] == 1:
                        new_grid[g] = vals_new[0, lev]
                    else:
                        new_grid[g] = vals_new[c0, lev]
                    # Always use mapping for old/model file
                    old_grid[g] = vals_old[c0, lev]

                if args.stats_only:
                    # Compute stats only
                    new_sum = float(np.nansum(new_grid))
                    new_std = float(np.nanstd(new_grid)) if np.any(np.isfinite(new_grid)) else float('nan')
                    new_min = float(np.nanmin(new_grid)) if np.any(np.isfinite(new_grid)) else float('nan')
                    new_max = float(np.nanmax(new_grid)) if np.any(np.isfinite(new_grid)) else float('nan')
                    old_sum = float(np.nansum(old_grid))
                    old_std = float(np.nanstd(old_grid)) if np.any(np.isfinite(old_grid)) else float('nan')
                    old_min = float(np.nanmin(old_grid)) if np.any(np.isfinite(old_grid)) else float('nan')
                    old_max = float(np.nanmax(old_grid)) if np.any(np.isfinite(old_grid)) else float('nan')
                    stats_rows.append({
                        "variable": var,
                        "suffix": f"_lev{lev}",
                        "new_sum": new_sum, "new_std": new_std, "new_min": new_min, "new_max": new_max,
                        "old_sum": old_sum, "old_std": old_std, "old_min": old_min, "old_max": old_max,
                    })
                else:
                    # Plotting mode
                    _plot_tripanel(var, f"_lev{lev}", grid_lon, grid_lat, new_grid, old_grid, str(out_dir_base),
                                   label_new=LABEL_NEW, label_old=LABEL_OLD)

        elif ("pft" in dims) and (len(dims) == 1):
            # PFT1D variable
            da_new_p = da_new.transpose("pft")
            da_old_p = da_old.transpose("pft")
            vals_new = _to_nan_fillvalue(da_new_p.values)
            vals_old = _to_nan_fillvalue(da_old_p.values)

            # Iterate all PFT TYPES (0..15) if stats-only; otherwise iterate requested PFTs
            # Note: da_new_p.sizes["pft"] is the total number of PFT entries across all gridcells (very large).
            #       For comparison we want PFT type indices 0..15 which are mapped per-gridcell via grid_to_pfts.
            if args.stats_only:
                pft_iter = range(16)
            else:
                pft_iter = PFT_PICK_LIST

            for k in pft_iter:
                if k < 0 or k >= da_new_p.sizes["pft"]:
                    print(f"  PFT {k} out of range, skipped")
                    continue
                if args.plot_all and k == 0:
                    continue  # skip pft0 for plot-all
                new_grid = np.full(n_grid, np.nan, dtype=float)
                old_grid = np.full(n_grid, np.nan, dtype=float)

                for g in range(n_grid):
                    pfts = grid_to_pfts[g]
                    if len(pfts) == 0:
                        continue
                    if k < len(pfts):
                        p_idx = pfts[k]
                        new_grid[g] = vals_new[p_idx]
                        old_grid[g] = vals_old[p_idx]

                if args.stats_only:
                    new_sum = float(np.nansum(new_grid))
                    new_std = float(np.nanstd(new_grid)) if np.any(np.isfinite(new_grid)) else float('nan')
                    new_min = float(np.nanmin(new_grid)) if np.any(np.isfinite(new_grid)) else float('nan')
                    new_max = float(np.nanmax(new_grid)) if np.any(np.isfinite(new_grid)) else float('nan')
                    old_sum = float(np.nansum(old_grid))
                    old_std = float(np.nanstd(old_grid)) if np.any(np.isfinite(old_grid)) else float('nan')
                    old_min = float(np.nanmin(old_grid)) if np.any(np.isfinite(old_grid)) else float('nan')
                    old_max = float(np.nanmax(old_grid)) if np.any(np.isfinite(old_grid)) else float('nan')
                    stats_rows.append({
                        "variable": var,
                        "suffix": f"_pft{k}",
                        "new_sum": new_sum, "new_std": new_std, "new_min": new_min, "new_max": new_max,
                        "old_sum": old_sum, "old_std": old_std, "old_min": old_min, "old_max": old_max,
                    })
                else:
                    _plot_tripanel(var, f"_pft{k}", grid_lon, grid_lat, new_grid, old_grid, str(out_dir_base),
                                   label_new=LABEL_NEW, label_old=LABEL_OLD)

        else:
            print(f"  Skip {var} (only supports (column, levgrnd) and (pft,))")

    # Write stats CSV if stats-only
    if args.stats_only and stats_rows:
        import csv
        csv_path = output_dir / 'summary_stats.csv'
        fieldnames = [
            'variable', 'suffix',
            'new_sum', 'new_std', 'new_min', 'new_max',
            'old_sum', 'old_std', 'old_min', 'old_max'
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in stats_rows:
                writer.writerow(row)
        print(f"Saved statistics CSV: {csv_path}")

    ds_new.close()
    ds_old.close()

    # Print summary
    print("\n" + "="*80)
    print("AI RESTART COMPARISON SUMMARY:")
    print("="*80)
    print(f"Original restart: {os.path.abspath(FILE_OLD)}")
    print(f"AI-enhanced restart: {os.path.abspath(FILE_NEW)}")
    print(f"Labels: {LABEL_OLD} vs {LABEL_NEW}")
    print(f"Output directory: {os.path.abspath(str(output_dir))}")
    print(f"Variables processed: {VARIABLES}")
    if args.stats_only:
        print("Mode: stats-only (all layers and all PFTs)")
    else:
        print(f"Layers plotted: {LEVGRND_LAYERS}")
        print(f"PFTs plotted: {PFT_PICK_LIST}")
    print("="*80)
    if args.stats_only:
        print("\nCompleted without plotting. Stats CSV saved to:", str(output_dir))
    else:
        print("\nAll plots done! Output dir:", str(out_dir_base))

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
