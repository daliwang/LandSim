import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from glob import glob

# Canonical definitions: "All layers" = 10 layers; "All PFTs" = PFT 1 through PFT 16 only (not others).
NUM_LAYERS = 10   # All layers (soil/2D) = exactly 10 layers
NUM_PFTS = 16     # All PFTs (1D) = pft1 to pft16 only

# Minimum gridcells after natveg filter to apply the filter; below this, we keep unfiltered data to avoid unstable metrics.
MIN_GRIDCELLS_FOR_NATVEG_FILTER = 10
# Minimum valid (non-NaN) gt/pred pairs to generate a plot for a PFT or layer.
MIN_VALID_PAIRS_FOR_PLOT = 3

# Plot output subfolders (under plots/ or top_bad_plots/):
#   aggregate_all  - one scatter per variable combining all PFTs (1D) or all layers (2D)
#   aggregate_bad  - one scatter per variable combining only bad/selected PFTs or layers (when using top-bad report)
#   by_pft_layer   - one scatter per PFT (1D), per layer (2D), or per scalar variable
SUBDIR_ALLLAYER = "aggregate_all"
SUBDIR_BADLAYER = "aggregate_bad"
SUBDIR_INDIVIDUAL = "by_pft_layer"

def plot_gt_vs_pred(gt, pred, title, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(gt, pred, alpha=0.5)
    plt.plot([gt.min(), gt.max()], [gt.min(), gt.max()], 'r--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(title)
    plt.tight_layout()
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def _parse_top_bad_report(report_path):
    """Parse quality_summary_report.txt to extract variables and associated PFT indices or layer numbers.

    Returns a mapping: { variable: { 'pfts': set[int], 'layers': set[int] } }
    """
    selection = {}
    if not os.path.exists(report_path):
        print(f"Top-bad report not found: {report_path}")
        return selection

    in_section = False
    try:
        with open(report_path, 'r') as f:
            for line in f:
                stripped = line.strip('\n')
                header = stripped.strip()
                # Detect start of section (support both old and new headings)
                if header.startswith('Top variables by bad-count'):
                    in_section = True
                    continue
                # Section ends at next heading or blank line followed by a heading; we keep it simple
                if in_section and header.startswith('## '):
                    break
                if in_section and stripped.startswith('  '):
                    # Example formats:
                    #   ppool: 15; pfts: 1, 2, 3, ...
                    #   smin_no3_vr: 10; layers: 1, 2, ...
                    #   var: 5
                    m = re.match(r"\s+([A-Za-z0-9_]+):\s*([0-9]+)(;.*)?$", stripped)
                    if not m:
                        continue
                    var = m.group(1)
                    details = m.group(3) or ''
                    pfts = set()
                    layers = set()
                    if 'pfts:' in details:
                        m_p = re.search(r"pfts:\s*([0-9,\s]+)", details)
                        if m_p:
                            nums = [n.strip() for n in m_p.group(1).split(',') if n.strip()]
                            for n in nums:
                                try:
                                    pfts.add(int(n))
                                except Exception:
                                    pass
                    if 'layers:' in details:
                        m_l = re.search(r"layers:\s*([0-9,\s]+)", details)
                        if m_l:
                            nums = [n.strip() for n in m_l.group(1).split(',') if n.strip()]
                            for n in nums:
                                try:
                                    layers.add(int(n))
                                except Exception:
                                    pass
                    selection[var] = { 'pfts': pfts, 'layers': layers }
    except Exception as e:
        print(f"Failed to parse top-bad report {report_path}: {e}")
        return {}
    return selection

def _load_gridcell_metadata(results_dir):
    """Load gridcell-level PCT_NATVEG and PCT_NAT_PFT_* from test_static_inverse.csv if present.
    Returns a DataFrame with same row order as GT/pred CSVs, or None if not available.
    Used to exclude gridcells with no natural veg (PCT_NATVEG=0) or 100% PFT0 (PCT_NAT_PFT_0=100).
    """
    # Prefer test_static_inverse.csv written by run_inference_all (same row order as predictions)
    candidates = [
        os.path.join(results_dir, 'cnp_predictions', 'test_static_inverse.csv'),
        os.path.join(results_dir, 'cnp_predictions', 'gridcell_metadata.csv'),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            # Need PCT_NATVEG and PCT_NAT_PFT_0 for exclusion; PCT_NAT_PFT_1..16 for low-coverage flag
            pct_natveg = None
            for c in ['PCT_NATVEG', 'pct_natveg']:
                if c in df.columns:
                    pct_natveg = df[c].values
                    break
            pct_pft0 = None
            for c in ['PCT_NAT_PFT_0', 'pct_nat_pft_0']:
                if c in df.columns:
                    pct_pft0 = df[c].values
                    break
            if pct_natveg is None or pct_pft0 is None:
                continue
            # Build include mask: include where (PCT_NATVEG > 0) and (PCT_NAT_PFT_0 < 100)
            include = (np.asarray(pct_natveg, dtype=float) > 0) & (np.asarray(pct_pft0, dtype=float) < 100)
            # PCT_NAT_PFT_1..16 for pft_pct_low
            pct_pft_cols = {}
            for i in range(1, 17):
                c = f'PCT_NAT_PFT_{i}'
                if c in df.columns:
                    pct_pft_cols[i] = df[c].values.astype(float)
            return {
                'include_mask': include,
                'pct_pft': pct_pft_cols,
                'n_rows': len(df),
            }
        except Exception as e:
            print(f"Warning: Could not load gridcell metadata from {path}: {e}")
    return None


def _parse_worst_vars_report(report_path):
    """Parse quality_summary_report.txt to extract variables from
    the '## Variables with Worst Predictions' section.
    Returns a mapping { variable: { 'pfts': set(), 'layers': set() } }
    """
    selection = {}
    if not os.path.exists(report_path):
        print(f"Worst-variables report not found: {report_path}")
        return selection
    in_section = False
    try:
        with open(report_path, 'r') as f:
            for line in f:
                stripped = line.strip('\n')
                header = stripped.strip()
                if header.startswith('## Variables with Worst Predictions'):
                    in_section = True
                    continue
                # Section ends at next heading
                if in_section and header.startswith('## '):
                    break
                if in_section and stripped and not stripped.startswith('#'):
                    # Expected line format: "<var>: <good>% good, <ok>% ok, <bad>% bad"
                    m = re.match(r"\s*([A-Za-z0-9_]+):\s*", stripped)
                    if not m:
                        continue
                    var = m.group(1)
                    selection[var] = { 'pfts': set(), 'layers': set() }
    except Exception as e:
        print(f"Failed to parse worst variables section {report_path}: {e}")
        return {}
    return selection

def main_with_flag(results_dir, plot_scatter, plot_loss, top_bad_only=False, top_bad_report=None, plots_dir_override=None, worst_only=False, use_natveg_filter=True):
    # Create plots subdirectory and subfolders for organization
    plots_dir = plots_dir_override or os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for sub in (SUBDIR_ALLLAYER, SUBDIR_BADLAYER, SUBDIR_INDIVIDUAL):
        os.makedirs(os.path.join(plots_dir, sub), exist_ok=True)
    
    # NEW: Create stats file
    stats_path = os.path.join(results_dir, "validation_stats.csv")
    stats_data = []

    # Optional: restrict plotting to selected variables (top-bad or worst list)
    selection = None
    if top_bad_only or worst_only:
        report_path = top_bad_report or os.path.join(results_dir, 'analysis', 'quality_summary_report.txt')
        if worst_only:
            selection = _parse_worst_vars_report(report_path)
            if selection:
                print(f"Plotting restricted to worst variables from: {report_path}")
        if (not selection) and top_bad_only:
            selection = _parse_top_bad_report(report_path)
            if selection:
                print(f"Plotting restricted to top-bad variables from: {report_path}")
        # If neither parser returned a selection, proceed unrestricted
        if not selection:
            print("No selections parsed from report; proceeding without restriction.")
            selection = None
    
    # Load gridcell metadata for PFT/2D filtering only when explicitly requested (opt-in; default off to avoid changing metrics)
    gridcell_metadata = _load_gridcell_metadata(results_dir) if use_natveg_filter else None
    if gridcell_metadata is not None:
        print("Using gridcell metadata for PFT/2D: excluding no-natveg and 100% PFT0 gridcells")
    
    # Check for new directory structure first
    pft_gt_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_ground_truth')
    pft_pred_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_predictions')
    
    # Handle 1D data with new structure
    if os.path.exists(pft_gt_dir) and os.path.exists(pft_pred_dir):
        print("Using new 1D directory structure")
        analyze_1d_new_structure(results_dir, '1D', plots_dir, stats_data, plot_scatter, selection, gridcell_metadata)
    else:
        # Fall back to old single-file format
        print("Using legacy 1D single-file format")
        gt_path = os.path.join(results_dir, 'cnp_predictions', 'ground_truth_1d.csv')
        pred_path = os.path.join(results_dir, 'cnp_predictions', 'predictions_1d.csv')
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            analyze_1d(gt_path, pred_path, '1D', plots_dir, results_dir, stats_data, plot_scatter)
        else:
            print("No 1D data found in either format")
    
    # Handle scalar data
    scalar_gt = os.path.join(results_dir, 'cnp_predictions', 'ground_truth_scalar.csv')
    scalar_pred = os.path.join(results_dir, 'cnp_predictions', 'predictions_scalar.csv')
    if os.path.exists(scalar_gt) and os.path.exists(scalar_pred):
        print("Analyzing scalar data...")
        analyze_pair(scalar_gt, scalar_pred, 'Scalar', plots_dir, stats_data, per_column=True, plot_scatter=plot_scatter, selection=selection)
    else:
        print("Scalar data files not found")
    
    # Handle 2D data if available
    soil_gt_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_ground_truth')
    soil_pred_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_predictions')
    if os.path.exists(soil_gt_dir) and os.path.exists(soil_pred_dir):
        print("Using new 2D directory structure")
        analyze_2d_new_structure(results_dir, '2D', plots_dir, stats_data, plot_scatter, selection, gridcell_metadata)
    else:
        # Fall back to old single-file format
        print("Using legacy 2D single-file format")
        soil_gt = os.path.join(results_dir, 'cnp_predictions', 'ground_truth_2d.csv')
        soil_pred = os.path.join(results_dir, 'cnp_predictions', 'predictions_2d.csv')
        if os.path.exists(soil_gt) and os.path.exists(soil_pred):
            analyze_2d(soil_gt, soil_pred, '2D', plots_dir, results_dir, stats_data, plot_scatter)
        else:
            print("No 2D data found in either format")
    
    # Plot train/val accuracy if available and requested
    if plot_loss:
        loss_csv = os.path.join(results_dir, 'cnp_training_losses.csv')
        if os.path.exists(loss_csv):
            plot_train_val_accuracy(loss_csv, plots_dir)
        else:
            print("Loss CSV not found for train/val accuracy plot.")
    
    # NEW: Save stats to CSV
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(stats_path, index=False)
        print(f"Saved validation stats to: {stats_path}")
    
    # Print test metrics if available
    test_metrics_path = os.path.join(results_dir, 'cnp_predictions','test_metrics.csv')
    if os.path.exists(test_metrics_path):
        print("\nTest Metrics:")
        print(pd.read_csv(test_metrics_path))
    else:
        print("test_metrics.csv not found.")

def analyze_pair(gt_path, pred_path, label, out_dir, stats_data, per_column=False, plot_scatter=True, selection=None):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    # Drop coordinate columns so they are never included in scatter plots or metrics
    _coord_cols = ['long', 'lat', 'Long', 'Lat', 'Longitude', 'Latitude']
    for c in _coord_cols:
        if c in gt.columns:
            gt = gt.drop(columns=[c])
        if c in pred.columns:
            pred = pred.drop(columns=[c])
    if per_column:
        # Coordinate columns to never include in scatter plots or metrics
        coord_cols = {'long', 'lat', 'Long', 'Lat', 'Longitude', 'Latitude'}
        # Per-column comparison for scalar
        for col in gt.columns:
            # Normalize variable name by stripping Y_ for selection matching
            col_norm = col[2:] if isinstance(col, str) and col.startswith('Y_') else col
            # Always skip coordinate columns - do not include in any scatter variable plots
            if str(col) in coord_cols or str(col_norm) in coord_cols:
                continue
            # If selection provided, only include scalar variables present in selection
            if selection is not None and col_norm not in selection:
                continue
            if col in pred.columns:
                print(f"Analyzing variable: {col}")
                gt_col = gt[col].values.flatten()
                pred_col = pred[col].values.flatten()
                
                # Calculate statistics
                gt_stats = {'min': np.nanmin(gt_col), 'max': np.nanmax(gt_col), 'sum': np.nansum(gt_col)}
                pred_stats = {'min': np.nanmin(pred_col), 'max': np.nanmax(pred_col), 'sum': np.nansum(pred_col)}
                
                # Print statistics
                print(f"  Ground Truth - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
                print(f"  Predictions - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
                
                rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
                mae = mean_absolute_error(gt_col, pred_col)
                r2 = r2_score(gt_col, pred_col)
                print(f"  {label} - {col}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                
                # Conditionally plot
                if plot_scatter:
                    plot_gt_vs_pred(gt_col, pred_col, f"{label} {col} GT vs Pred", os.path.join(out_dir, SUBDIR_INDIVIDUAL, f"{label}_{col}_gt_vs_pred.png"))
                
                # Collect stats
                stats_data.append({
                    'type': label,
                    'variable': col,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'gt_min': gt_stats['min'],
                    'gt_max': gt_stats['max'],
                    'gt_sum': gt_stats['sum'],
                    'pred_min': pred_stats['min'],
                    'pred_max': pred_stats['max'],
                    'pred_sum': pred_stats['sum']
                })
            else:
                print(f"Column {col} missing in predictions for {label}")
        return None
    else:
        # Flatten if needed
        gt_flat = gt.values.flatten()
        pred_flat = pred.values.flatten()
        
        # Calculate statistics
        gt_stats = {'min': np.nanmin(gt_flat), 'max': np.nanmax(gt_flat), 'sum': np.nansum(gt_flat)}
        pred_stats = {'min': np.nanmin(pred_flat), 'max': np.nanmax(pred_flat), 'sum': np.nansum(pred_flat)}
        
        # Print statistics
        print(f"Ground Truth - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
        print(f"Predictions - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))
        mae = mean_absolute_error(gt_flat, pred_flat)
        r2 = r2_score(gt_flat, pred_flat)
        print(f"{label} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Conditionally plot
        if plot_scatter:
            plot_gt_vs_pred(gt_flat, pred_flat, f"{label} GT vs Pred", os.path.join(out_dir, SUBDIR_INDIVIDUAL, f"{label}_gt_vs_pred.png"))
        
        # Collect stats
        stats_data.append({
            'type': label,
            'variable': 'all',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'gt_min': gt_stats['min'],
            'gt_max': gt_stats['max'],
            'gt_sum': gt_stats['sum'],
            'pred_min': pred_stats['min'],
            'pred_max': pred_stats['max'],
            'pred_sum': pred_stats['sum']
        })
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

def analyze_1d_new_structure(results_dir, label, out_dir, stats_data, plot_scatter=True, selection=None, gridcell_metadata=None):
    """Analyze 1D data using the new directory structure with individual variable files.
    If gridcell_metadata is provided, exclude gridcells where PCT_NATVEG=0 or PCT_NAT_PFT_0=100
    and set pft_pct_low for PFTs with mean coverage < 2%.
    """
    gt_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_ground_truth')
    pred_dir = os.path.join(results_dir, 'cnp_predictions', 'pft_1d_predictions')
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Missing 1D directories: {gt_dir} or {pred_dir}")
        return
    
    # Get all ground truth files
    gt_files = glob(os.path.join(gt_dir, 'ground_truth_Y_*.csv'))
    if not gt_files:
        print(f"No ground truth files found in {gt_dir}")
        return
    
    print(f"Found {len(gt_files)} 1D variables to analyze")
    
    for gt_file in gt_files:
        # Extract variable name from filename
        var_name = os.path.basename(gt_file).replace('ground_truth_Y_', '').replace('.csv', '')
        
        # Find corresponding prediction file
        pred_file = os.path.join(pred_dir, f'predictions_Y_{var_name}.csv')
        
        if not os.path.exists(pred_file):
            print(f"Missing prediction file for {var_name}: {pred_file}")
            continue
        
        # Read data (needed for both per-PFT and AllPFTs)
        gt_data = pd.read_csv(gt_file)
        pred_data = pd.read_csv(pred_file)
        # Drop coordinate columns so they are not included in scatter plots or metrics
        coord_cols = ['long', 'lat', 'Long', 'Lat', 'Longitude', 'Latitude']
        for col in coord_cols:
            if col in gt_data.columns:
                gt_data = gt_data.drop(columns=[col])
            if col in pred_data.columns:
                pred_data = pred_data.drop(columns=[col])
        
        # Ensure same shape
        if gt_data.shape != pred_data.shape:
            print(f"Shape mismatch for {var_name}: GT {gt_data.shape} vs Pred {pred_data.shape}")
            continue

        # Build row mask for PFT/2D: exclude PCT_NATVEG=0 and PCT_NAT_PFT_0=100
        include_mask = None
        if gridcell_metadata is not None and gridcell_metadata['n_rows'] == len(gt_data):
            include_mask = gridcell_metadata['include_mask']

        # When using selection (top-bad only): skip this variable for per-PFT plots if not selected,
        # but we still generate the AllPFTs plot for every variable so AllPFTs are never "missing".
        in_selection = selection is None or var_name in selection
        if in_selection:
            print(f"Analyzing variable: {var_name}")
        
        # Analyze each PFT column (only for selected variables when selection is set)
        num_pfts = gt_data.shape[1]
        if in_selection:
            print(f"  {var_name}: {num_pfts} PFT columns")
        
        for pft_idx, col_name in enumerate(gt_data.columns):
            if not in_selection:
                continue  # Only plot per-PFT for selected variables; AllPFTs still generated below
            # If selection provided, attempt to parse PFT index from col_name like 'Y_var_pftX'
            if selection is not None:
                sel = selection.get(var_name, None)
                if sel is not None and sel['pfts']:
                    pft_match = re.search(r"pft(\d+)$", col_name)
                    if pft_match:
                        try:
                            pft_num = int(pft_match.group(1))
                            if pft_num not in sel['pfts']:
                                continue
                        except Exception:
                            pass
            gt_col = gt_data[col_name].values
            pred_col = pred_data[col_name].values
            
            # Apply natveg filter: use only included gridcells (PCT_NATVEG>0 and PCT_NAT_PFT_0<100)
            if include_mask is not None:
                n_included = int(np.sum(include_mask))
                if n_included < MIN_GRIDCELLS_FOR_NATVEG_FILTER:
                    pass  # keep gt_col, pred_col unfiltered to avoid unstable metrics
                else:
                    gt_col = gt_col[include_mask]
                    pred_col = pred_col[include_mask]
            
            # Skip if all values are NaN
            if np.all(np.isnan(gt_col)) or np.all(np.isnan(pred_col)):
                continue
            
            # Remove NaN pairs
            valid_mask = ~(np.isnan(gt_col) | np.isnan(pred_col))
            if np.sum(valid_mask) < MIN_VALID_PAIRS_FOR_PLOT:
                continue
            
            gt_valid = gt_col[valid_mask]
            pred_valid = pred_col[valid_mask]
            
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_valid), 'max': np.nanmax(gt_valid), 'sum': np.nansum(gt_valid)}
            pred_stats = {'min': np.nanmin(pred_valid), 'max': np.nanmax(pred_valid), 'sum': np.nansum(pred_valid)}
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(gt_valid, pred_valid))
            mae = mean_absolute_error(gt_valid, pred_valid)
            r2 = r2_score(gt_valid, pred_valid)
            
            # PFT coverage < 2% flag for scatter coloring (PFT 1 = pft_idx 0 -> PCT_NAT_PFT_1)
            pft_pct_low = False
            if gridcell_metadata is not None and include_mask is not None:
                pft_num = pft_idx + 1  # 1-based
                if pft_num in gridcell_metadata.get('pct_pft', {}):
                    pct_vals = gridcell_metadata['pct_pft'][pft_num][include_mask]
                    if len(pct_vals) > 0:
                        pft_pct_low = float(np.nanmean(pct_vals)) < 2.0
            
            print(f"    {col_name}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"      GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"      Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            
            # Conditionally plot
            if plot_scatter:
                plot_gt_vs_pred(
                    gt_valid, pred_valid, 
                    f"{label} {col_name} GT vs Pred", 
                    os.path.join(out_dir, SUBDIR_INDIVIDUAL, f"{label}_{col_name}_gt_vs_pred.png")
                )
            
            # Collect stats
            row = {
                'type': '1D',
                'variable': var_name,
                'pft': col_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'gt_min': gt_stats['min'],
                'gt_max': gt_stats['max'],
                'gt_sum': gt_stats['sum'],
                'pred_min': pred_stats['min'],
                'pred_max': pred_stats['max'],
                'pred_sum': pred_stats['sum']
            }
            if gridcell_metadata is not None:
                row['pft_pct_low'] = pft_pct_low
            stats_data.append(row)
        
        # When not in top-bad mode: provide overall scatter across all PFTs (aggregate_all).
        # When selection is set (top_bad_plots), skip so we only get aggregate_bad and by_pft_layer.
        if selection is None:
            try:
                pft_cols = []
                for col in gt_data.columns:
                    m = re.search(r"pft(\d+)$", col)
                    if m and 1 <= int(m.group(1)) <= NUM_PFTS:
                        pft_cols.append(col)
                if pft_cols and all(c in pred_data.columns for c in pft_cols):
                    gt_all_pfts = gt_data[pft_cols].values.flatten()
                    pred_all_pfts = pred_data[pft_cols].values.flatten()
                else:
                    gt_all_pfts = gt_data.values.flatten()
                    pred_all_pfts = pred_data.values.flatten()
                valid_mask = ~(np.isnan(gt_all_pfts) | np.isnan(pred_all_pfts))
                if np.sum(valid_mask) >= 3 and plot_scatter:
                    plot_gt_vs_pred(
                        gt_all_pfts[valid_mask], pred_all_pfts[valid_mask],
                        f"{label} {var_name} AllPFTs (pft1–pft{NUM_PFTS}) GT vs Pred",
                        os.path.join(out_dir, SUBDIR_ALLLAYER, f"{label}_{var_name}_AllPFTs_gt_vs_pred.png")
                    )
            except Exception as e:
                print(f"  Skipped aggregated AllPFTs plot for {var_name}: {e}")

        # When selection (top-bad) is used: also plot just the bad PFTs aggregated together.
        if selection is not None and var_name in selection and plot_scatter:
            sel = selection.get(var_name, None)
            if sel and sel.get('pfts'):
                try:
                    bad_pft_nums = sorted(sel['pfts'])
                    pft_cols = []
                    for col in gt_data.columns:
                        m = re.search(r"pft(\d+)$", col)
                        if m:
                            pft_num = int(m.group(1))
                            if pft_num in bad_pft_nums:
                                pft_cols.append(col)
                    if pft_cols and all(c in pred_data.columns for c in pft_cols):
                        gt_bad = gt_data[pft_cols].values.flatten()
                        pred_bad = pred_data[pft_cols].values.flatten()
                        valid_mask = ~(np.isnan(gt_bad) | np.isnan(pred_bad))
                        if np.sum(valid_mask) >= 3:
                            pft_str = ",".join(str(p) for p in bad_pft_nums)
                            plot_gt_vs_pred(
                                gt_bad[valid_mask], pred_bad[valid_mask],
                                f"{label} {var_name} BadPFTs (pfts {pft_str}) GT vs Pred",
                                os.path.join(out_dir, SUBDIR_BADLAYER, f"{label}_{var_name}_BadPFTs_gt_vs_pred.png")
                            )
                except Exception as e:
                    print(f"  Skipped BadPFTs plot for {var_name}: {e}")
            
            # Also generate aggregate_all plot for this bad variable (all PFTs, not just bad ones)
            try:
                pft_cols_all = []
                for col in gt_data.columns:
                    m = re.search(r"pft(\d+)$", col)
                    if m and 1 <= int(m.group(1)) <= NUM_PFTS:
                        pft_cols_all.append(col)
                if pft_cols_all and all(c in pred_data.columns for c in pft_cols_all):
                    gt_all_pfts = gt_data[pft_cols_all].values.flatten()
                    pred_all_pfts = pred_data[pft_cols_all].values.flatten()
                    valid_mask = ~(np.isnan(gt_all_pfts) | np.isnan(pred_all_pfts))
                    if np.sum(valid_mask) >= 3:
                        plot_gt_vs_pred(
                            gt_all_pfts[valid_mask], pred_all_pfts[valid_mask],
                            f"{label} {var_name} AllPFTs (pft1–pft{NUM_PFTS}) GT vs Pred",
                            os.path.join(out_dir, SUBDIR_ALLLAYER, f"{label}_{var_name}_AllPFTs_gt_vs_pred.png")
                        )
            except Exception as e:
                print(f"  Skipped aggregate_all AllPFTs plot for {var_name}: {e}")

def analyze_1d(gt_path, pred_path, label, out_dir, results_dir, stats_data, plot_scatter=True, selection=None):
    """Legacy function for old single-file 1D format - kept for compatibility"""
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    # Drop coordinate columns so they are not included in scatter plots or metrics
    coord_cols = ['long', 'lat', 'Long', 'Lat', 'Longitude', 'Latitude']
    for col in coord_cols:
        if col in gt.columns:
            gt = gt.drop(columns=[col])
        if col in pred.columns:
            pred = pred.drop(columns=[col])
    # Read variable names from cnp_config.json
    config_path = os.path.join(results_dir, 'cnp_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        variable_names = config.get('data_info', {}).get('variables_1d_pft', None)
        if variable_names is None:
            print('[ERROR] Could not find variables_1d_pft in cnp_config.json!')
            return
    else:
        print(f'[ERROR] cnp_config.json not found in {results_dir}!')
        return
    num_vars = len(variable_names)
    num_pfts = NUM_PFTS  # All PFTs = pft1 to pft16 only (pft0 is dropped in training)
    num_samples = gt.shape[0]
    expected_cols = num_vars * num_pfts
    print(f"[DEBUG] 1D: num_samples={num_samples}, num_vars={num_vars}, num_pfts={num_pfts}, expected_cols={expected_cols}, actual_cols={gt.shape[1]}")
    if gt.shape[1] != expected_cols:
        print("[ERROR] Unexpected number of columns in 1D data!")
        print("Column names:", list(gt.columns))
        return
    gt_reshaped = gt.values.reshape(num_samples, num_vars, num_pfts)
    pred_reshaped = pred.values.reshape(num_samples, num_vars, num_pfts)
    # For each variable and each PFT column, compare
    for i, var in enumerate(variable_names):
        # Skip if restricting to top-bad variables and this variable is not selected
        if selection is not None and var not in selection:
            continue
        print(f"Analyzing variable: {var}")
        for j in range(num_pfts):
            # If selection provided, enforce PFT filtering (1-based indexing in report)
            if selection is not None:
                sel = selection.get(var, None)
                if sel is not None and sel['pfts'] and (j + 1) not in sel['pfts']:
                    continue
            gt_col = gt_reshaped[:, i, j]
            pred_col = pred_reshaped[:, i, j]
            # Use column name if available
            col_name = gt.columns[i * num_pfts + j] if (i * num_pfts + j) < len(gt.columns) else f"{var}_pft{j+1}"
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_col), 'max': np.nanmax(gt_col), 'sum': np.nansum(gt_col)}
            pred_stats = {'min': np.nanmin(pred_col), 'max': np.nanmax(pred_col), 'sum': np.nansum(pred_col)}
            rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
            mae = mean_absolute_error(gt_col, pred_col)
            r2 = r2_score(gt_col, pred_col)
            print(f"{label} - {col_name}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"  GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"  Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            # Plot using column name
            if plot_scatter:
                plot_gt_vs_pred(gt_col, pred_col, f"{label} {col_name} GT vs Pred", os.path.join(out_dir, SUBDIR_INDIVIDUAL, f"{label}_{col_name}_gt_vs_pred.png"))
            # Collect stats
            stats_data.append({
                'type': '1D',
                'variable': var,
                'pft': col_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'gt_min': gt_stats['min'],
                'gt_max': gt_stats['max'],
                'gt_sum': gt_stats['sum'],
                'pred_min': pred_stats['min'],
                'pred_max': pred_stats['max'],
                'pred_sum': pred_stats['sum']
            })

def analyze_2d_new_structure(results_dir, label, out_dir, stats_data, plot_scatter=True, selection=None, gridcell_metadata=None):
    """Analyze 2D data using the new directory structure with individual variable files.
    If gridcell_metadata is provided, exclude gridcells where PCT_NATVEG=0 or PCT_NAT_PFT_0=100.
    """
    gt_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_ground_truth')
    pred_dir = os.path.join(results_dir, 'cnp_predictions', 'soil_2d_predictions')
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print(f"Missing 2D directories: {gt_dir} or {pred_dir}")
        return
    
    # Get all ground truth files
    gt_files = glob(os.path.join(gt_dir, 'ground_truth_Y_*.csv'))
    if not gt_files:
        print(f"No 2D ground truth files found in {gt_dir}")
        return
    
    print(f"Found {len(gt_files)} 2D variables to analyze")
    
    for gt_file in gt_files:
        # Extract variable name from filename
        var_name = os.path.basename(gt_file).replace('ground_truth_Y_', '').replace('.csv', '')
        
        # Find corresponding prediction file
        pred_file = os.path.join(pred_dir, f'predictions_Y_{var_name}.csv')
        
        if not os.path.exists(pred_file):
            print(f"Missing prediction file for {var_name}: {pred_file}")
            continue
            
        # Skip if restricting to top-bad variables and this variable is not selected
        if selection is not None and var_name not in selection:
            continue
        print(f"Analyzing 2D variable: {var_name}")
        
        # Read data
        gt_data = pd.read_csv(gt_file)
        pred_data = pd.read_csv(pred_file)
        # Drop 'long' and 'lat' columns if present
        for col in ['long', 'lat', 'Long', 'Lat', 'Longitude', 'Latitude']:
            if col in gt_data.columns:
                gt_data = gt_data.drop(columns=[col])
            if col in pred_data.columns:
                pred_data = pred_data.drop(columns=[col])
        
        # Ensure same shape
        if gt_data.shape != pred_data.shape:
            print(f"Shape mismatch for {var_name}: GT {gt_data.shape} vs Pred {pred_data.shape}")
            continue

        # Build row mask: exclude PCT_NATVEG=0 and PCT_NAT_PFT_0=100
        include_mask = None
        if gridcell_metadata is not None and gridcell_metadata['n_rows'] == len(gt_data):
            include_mask = gridcell_metadata['include_mask']
        
        # 2D data: All layers = NUM_LAYERS (10) only.
        total_columns = gt_data.shape[1]
        expected_columns = 1 * NUM_LAYERS  # 10 layers
        
        if total_columns != expected_columns:
            print(f"  Warning: Expected {expected_columns} columns for 2D data, but found {total_columns}")
            if total_columns % NUM_LAYERS != 0:
                print(f"  Error: Number of columns ({total_columns}) is not divisible by {NUM_LAYERS}")
                continue
            num_columns = total_columns // NUM_LAYERS
            print(f"  Assuming {num_columns} columns with {NUM_LAYERS} layers each")
        else:
            num_columns = 1
            print(f"  {var_name}: {num_columns} columns, each with {NUM_LAYERS} layers ({total_columns} total columns)")
        
        # Analyze all NUM_LAYERS (10) layers of the first column (new prediction format stores only first column)
        first_column_idx = 0
        layers_to_analyze = NUM_LAYERS
        
        for layer_idx in range(layers_to_analyze):
            # If selection provided, enforce layer filtering (1-based indexing in report)
            if selection is not None:
                sel = selection.get(var_name, None)
                if sel is not None and sel['layers'] and (layer_idx + 1) not in sel['layers']:
                    continue
            # Calculate the correct column index: first_column * NUM_LAYERS + layer
            col_idx = layer_idx if num_columns == 1 else (first_column_idx * NUM_LAYERS + layer_idx)
            if col_idx >= total_columns:
                print(f"    Warning: Column index {col_idx} out of range for {total_columns} columns")
                continue
                
            gt_col = gt_data.iloc[:, col_idx].values
            pred_col = pred_data.iloc[:, col_idx].values
            
            # Apply natveg filter: use only included gridcells (PCT_NATVEG>0 and PCT_NAT_PFT_0<100)
            if include_mask is not None:
                n_included = int(np.sum(include_mask))
                if n_included >= MIN_GRIDCELLS_FOR_NATVEG_FILTER:
                    gt_col = gt_col[include_mask]
                    pred_col = pred_col[include_mask]
                # else: too few after filter; keep unfiltered to avoid unstable metrics
            
            # Skip if all values are NaN
            if np.all(np.isnan(gt_col)) or np.all(np.isnan(pred_col)):
                continue
                
            # Remove NaN pairs
            valid_mask = ~(np.isnan(gt_col) | np.isnan(pred_col))
            if np.sum(valid_mask) < MIN_VALID_PAIRS_FOR_PLOT:
                continue
                
            gt_valid = gt_col[valid_mask]
            pred_valid = pred_col[valid_mask]
            
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_valid), 'max': np.nanmax(gt_valid), 'sum': np.nansum(gt_valid)}
            pred_stats = {'min': np.nanmin(pred_valid), 'max': np.nanmax(pred_valid), 'sum': np.nansum(pred_valid)}
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(gt_valid, pred_valid))
            mae = mean_absolute_error(gt_valid, pred_valid)
            r2 = r2_score(gt_valid, pred_valid)
            
            print(f"    Layer {layer_idx+1}: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"      GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"      Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            
            # Conditionally plot
            if plot_scatter:
                plot_gt_vs_pred(
                    gt_valid, pred_valid, 
                    f"{label} {var_name} Layer{layer_idx+1} GT vs Pred", 
                    os.path.join(out_dir, SUBDIR_INDIVIDUAL, f"{label}_{var_name}_Layer{layer_idx+1}_gt_vs_pred.png")
                )
            
            # Collect stats
            stats_data.append({
                'type': '2D',
                'variable': var_name,
                'layer': layer_idx+1,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'gt_min': gt_stats['min'],
                'gt_max': gt_stats['max'],
                'gt_sum': gt_stats['sum'],
                'pred_min': pred_stats['min'],
                'pred_max': pred_stats['max'],
                'pred_sum': pred_stats['sum']
            })

        # When not in top-bad mode: overall scatter across all layers (aggregate_all).
        # When selection is set (top_bad_plots), skip so we only get aggregate_bad and by_pft_layer.
        if selection is None:
            try:
                gt_firstcol = gt_data.iloc[:, 0:NUM_LAYERS].values.flatten()
                pred_firstcol = pred_data.iloc[:, 0:NUM_LAYERS].values.flatten()
                valid_mask = ~(np.isnan(gt_firstcol) | np.isnan(pred_firstcol))
                if np.sum(valid_mask) >= 3 and plot_scatter:
                    plot_gt_vs_pred(
                        gt_firstcol[valid_mask], pred_firstcol[valid_mask],
                        f"{label} {var_name} FirstCol({NUM_LAYERS} layers) GT vs Pred",
                        os.path.join(out_dir, SUBDIR_ALLLAYER, f"{label}_{var_name}_FirstCol_AllLayers_gt_vs_pred.png")
                    )
            except Exception as e:
                print(f"  Skipped overall plot for {var_name}: {e}")

        # When selection (top-bad) is used: also plot just the bad layers aggregated together.
        if selection is not None and var_name in selection and plot_scatter:
            sel = selection.get(var_name, None)
            if sel and sel.get('layers'):
                try:
                    bad_layer_nums = sorted(sel['layers'])
                    # Layer numbers in report are 1-based; columns are 0-based
                    col_indices = [l - 1 for l in bad_layer_nums if 1 <= l <= NUM_LAYERS]
                    if col_indices:
                        gt_bad = gt_data.iloc[:, col_indices].values.flatten()
                        pred_bad = pred_data.iloc[:, col_indices].values.flatten()
                        valid_mask = ~(np.isnan(gt_bad) | np.isnan(pred_bad))
                        if np.sum(valid_mask) >= 3:
                            layer_str = ",".join(str(l) for l in bad_layer_nums)
                            plot_gt_vs_pred(
                                gt_bad[valid_mask], pred_bad[valid_mask],
                                f"{label} {var_name} BadLayers ({layer_str}) GT vs Pred",
                                os.path.join(out_dir, SUBDIR_BADLAYER, f"{label}_{var_name}_BadLayers_gt_vs_pred.png")
                            )
                except Exception as e:
                    print(f"  Skipped BadLayers plot for {var_name}: {e}")
            
            # Also generate aggregate_all plot for this bad variable (all layers, not just bad ones)
            try:
                gt_all_layers = gt_data.iloc[:, 0:NUM_LAYERS].values.flatten()
                pred_all_layers = pred_data.iloc[:, 0:NUM_LAYERS].values.flatten()
                valid_mask = ~(np.isnan(gt_all_layers) | np.isnan(pred_all_layers))
                if np.sum(valid_mask) >= 3:
                    plot_gt_vs_pred(
                        gt_all_layers[valid_mask], pred_all_layers[valid_mask],
                        f"{label} {var_name} FirstCol({NUM_LAYERS} layers) GT vs Pred",
                        os.path.join(out_dir, SUBDIR_ALLLAYER, f"{label}_{var_name}_FirstCol_AllLayers_gt_vs_pred.png")
                    )
            except Exception as e:
                print(f"  Skipped aggregate_all AllLayers plot for {var_name}: {e}")

def analyze_2d(gt_path, pred_path, label, out_dir, results_dir, stats_data, plot_scatter=True, selection=None):
    """Legacy function for old single-file 2D format - kept for compatibility"""
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    # Read variable names from cnp_config.json
    config_path = os.path.join(results_dir, 'cnp_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        variable_names = config.get('data_info', {}).get('variables_2d_soil', None)
        if variable_names is None:
            print('[ERROR] Could not find variables_2d_soil in cnp_config.json!')
            return
    else:
        print(f'[ERROR] cnp_config.json not found in {results_dir}!')
        return
    num_vars = len(variable_names)
    num_columns = 18  # 2D soil data has 18 columns
    num_layers_per_column = NUM_LAYERS  # All layers = 10 only
    num_samples = gt.shape[0]
    expected_cols = num_vars * num_columns * num_layers_per_column
    print(f"[DEBUG] 2D: num_samples={num_samples}, num_vars={num_vars}, num_columns={num_columns}, layers_per_column={num_layers_per_column}, expected_cols={expected_cols}, actual_cols={gt.shape[1]}")
    if gt.shape[1] != expected_cols:
        print("[ERROR] Unexpected number of columns in 2D data!")
        print("Column names:", list(gt.columns))
        return
    gt_reshaped = gt.values.reshape(num_samples, num_vars, num_columns, num_layers_per_column)
    pred_reshaped = pred.values.reshape(num_samples, num_vars, num_columns, num_layers_per_column)
    # For each variable, analyze all 10 layers of the first column
    for i, var in enumerate(variable_names):
        # Skip if restricting to top-bad variables and this variable is not selected
        if selection is not None and var not in selection:
            continue
        print(f"Analyzing 2D variable: {var}")
        for j in range(NUM_LAYERS):  # All layers = 10 only
            # If selection provided, enforce layer filtering (1-based indexing in report)
            if selection is not None:
                sel = selection.get(var, None)
                if sel is not None and sel['layers'] and (j + 1) not in sel['layers']:
                    continue
            gt_col = gt_reshaped[:, i, 0, j]  # First column (index 0), all 10 layers
            pred_col = pred_reshaped[:, i, 0, j]  # First column (index 0), all 10 layers
            
            # Calculate statistics
            gt_stats = {'min': np.nanmin(gt_col), 'max': np.nanmax(gt_col), 'sum': np.nansum(gt_col)}
            pred_stats = {'min': np.nanmin(pred_col), 'max': np.nanmax(pred_col), 'sum': np.nansum(pred_col)}
            
            rmse = np.sqrt(mean_squared_error(gt_col, pred_col))
            mae = mean_absolute_error(gt_col, pred_col)
            r2 = r2_score(gt_col, pred_col)
            print(f"{label} - {var} (Layer {j+1}): RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            print(f"  GT - min: {gt_stats['min']:.6g}, max: {gt_stats['max']:.6g}, sum: {gt_stats['sum']:.6g}")
            print(f"  Pred - min: {pred_stats['min']:.6g}, max: {pred_stats['max']:.6g}, sum: {pred_stats['sum']:.6g}")
            
            if plot_scatter:
                plot_gt_vs_pred(gt_col, pred_col, f"{label} {var} Layer{j+1} GT vs Pred", os.path.join(out_dir, SUBDIR_INDIVIDUAL, f"{label}_{var}_Layer{j+1}_gt_vs_pred.png"))
            # Collect stats
            stats_data.append({
                'type': '2D',
                'variable': var,
                'layer': j+1,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'gt_min': gt_stats['min'],
                'gt_max': gt_stats['max'],
                'gt_sum': gt_stats['sum'],
                'pred_min': pred_stats['min'],
                'pred_max': pred_stats['max'],
                'pred_sum': pred_stats['sum']
            })

def plot_train_val_accuracy(loss_csv, out_dir):
    df = pd.read_csv(loss_csv)
    plt.figure()
    if 'Train Loss' in df.columns and 'Validation Loss' in df.columns:
        plt.plot(df['Train Loss'], label='Train Loss')
        plt.plot(df['Validation Loss'], label='Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('Train/Validation Loss')
        plt.tight_layout()
        # Ensure the directory exists
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, 'train_val_loss.png'))
        plt.close()
    else:
        print("train_loss or val_loss columns not found in loss CSV.")

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Postprocess CNP model results')
    # MODIFIED: Make results_dir optional with default to current directory
    parser.add_argument('results_dir', nargs='?', default='.', help='Results directory (run_xxxxx). Default is current directory.')
    
    # NEW: Separate flags for scatter plots and loss plot
    parser.add_argument('--no-scatter', action='store_false', dest='plot_scatter', help='Do not generate scatter plots')
    parser.add_argument('--no-plot-loss', action='store_false', dest='plot_loss', help='Do not plot train/val loss curve')
    # NEW: Stats-only mode disables all plots but still computes and saves statistics
    parser.add_argument('--stats-only', action='store_true', help='Only compute and save statistics CSV; do not generate any plots')
    # NEW: Restrict plotting to top-bad or worst variables from summary report
    parser.add_argument('--top-bad-only', action='store_true', help='Plot only variables listed in the quality summary top-bad section')
    parser.add_argument('--worst-only', action='store_true', help='Plot only variables listed under \"Variables with Worst Predictions\"')
    parser.add_argument('--top-bad-report', type=str, default=None, help='Path to quality_summary_report.txt (defaults to results_dir/analysis/quality_summary_report.txt)')
    parser.add_argument('--no-natveg-filter', action='store_false', dest='use_natveg_filter',
                        help='Do not apply PFT/2D filter (plot all gridcells). Default: natveg filter is ON (exclude PCT_NATVEG=0 or PCT_NAT_PFT_0=100).')
    
    parser.set_defaults(plot_scatter=True, plot_loss=True, use_natveg_filter=True)
    args = parser.parse_args()
    
    # If stats-only requested, force-disable all plotting
    if getattr(args, 'stats_only', False):
        args.plot_scatter = False
        args.plot_loss = False
    
    if len(sys.argv) < 2:
        print("Using current directory as results directory")
    
    main_with_flag(args.results_dir, args.plot_scatter, args.plot_loss, args.top_bad_only, args.top_bad_report,
                   worst_only=getattr(args, 'worst_only', False),
                   use_natveg_filter=getattr(args, 'use_natveg_filter', True))