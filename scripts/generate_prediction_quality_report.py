#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import importlib.util
import sys
import json
from typing import Optional, List

def main():
    parser = argparse.ArgumentParser(description='Generate prediction quality report from validation statistics')
    parser.add_argument('input', nargs='?', default="./validation_stats.csv",
                        help='Path to validation statistics CSV file (default: ./validation_stats.csv)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save output files (default: same directory as input + /analysis)')
    parser.add_argument('--training-config', default=None,
                        help='Path to training cnp_config.json (default: auto-detect near input)')
    parser.add_argument('--r2-good', type=float, default=0.9,
                        help='R² threshold for good predictions (default: 0.9)')
    parser.add_argument('--r2-ok', type=float, default=0.7,
                        help='R² threshold for ok predictions (default: 0.7)')
    parser.add_argument('--rmse-good', type=float, default=0.1,
                        help='Relative RMSE threshold for good predictions (default: 0.1)')
    parser.add_argument('--rmse-ok', type=float, default=0.25,
                        help='Relative RMSE threshold for ok predictions (default: 0.25)')
    parser.add_argument('--mae-good', type=float, default=0.1,
                        help='Relative MAE threshold for good predictions (default: 0.1)')
    parser.add_argument('--mae-ok', type=float, default=0.25,
                        help='Relative MAE threshold for ok predictions (default: 0.25)')
    # Diagnostics and display options (default: enabled); provide --no-* to disable
    parser.add_argument('--force-xlim-01', dest='force_xlim_01', action='store_true',
                        help='Force R² x-axis limits to [0, 1] in the scatter plot', default=True)
    parser.add_argument('--no-force-xlim-01', dest='force_xlim_01', action='store_false',
                        help='Do not force R² x-axis limits to [0, 1]')
    parser.add_argument('--print-scatter-stats', dest='print_scatter_stats', action='store_true',
                        help='Print min/max and counts for R² and relative RMSE used in the scatter plot', default=True)
    parser.add_argument('--no-print-scatter-stats', dest='print_scatter_stats', action='store_false',
                        help='Disable printing diagnostics for R² and relative RMSE used in the scatter plot')
    parser.add_argument('--bad-html-limit', type=int, default=100,
                        help='Maximum number of bad prediction rows to show in the HTML report (default: 100)')
    parser.add_argument('--bad-text-limit', type=int, default=200,
                        help='Maximum number of bad prediction rows to print in the text report (default: 200)')
    parser.add_argument('--export-bad', dest='export_bad', action='store_true', default=True,
                        help='Export detailed bad predictions to CSV (default: enabled)')
    parser.add_argument('--no-export-bad', dest='export_bad', action='store_false',
                        help='Disable exporting detailed bad predictions to CSV')
    parser.add_argument('--include-bad-details-text', dest='include_bad_details_text', action='store_true', default=False,
                        help='Include the long detailed list of bad predictions in the text report (default: disabled)')
    parser.add_argument('--no-include-bad-details-text', dest='include_bad_details_text', action='store_false',
                        help='Do not include the long detailed list of bad predictions in the text report')
    parser.add_argument('--top-bad-plots', dest='top_bad_plots', action='store_true', default=True,
                        help='Generate plots for top-bad variables (default: enabled)')
    parser.add_argument('--no-top-bad-plots', dest='top_bad_plots', action='store_false',
                        help='Disable generating top-bad plots')
    # Plot only variables listed in "Variables with Worst Predictions"
    parser.add_argument('--worst-only', dest='worst_only', action='store_true', default=False,
                        help='Plot only variables in the "Variables with Worst Predictions" section')
    parser.add_argument('--worst-filter-bad-only', dest='worst_filter_bad_only', action='store_true', default=True,
                        help='Filter worst variables to only include those with bad predictions (bad_pct > 0). Default: True')
    parser.add_argument('--no-worst-filter-bad-only', dest='worst_filter_bad_only', action='store_false',
                        help='Include all variables in worst list, even if 0%% bad (sorted by good_pct)')
    parser.add_argument('--worst-min-good-pct', type=float, default=None,
                        help='Include variables in worst list with good_pct below this threshold (e.g., 50.0 for <50%% good)')
    parser.add_argument('--worst-vars-list', type=str, default=None,
                        help='Comma-separated list of specific variables to include in worst list (e.g., "cpool,npool,ppool")')
    parser.add_argument('--no-natveg-filter', dest='natveg_filter', action='store_false', default=True,
                        help='Do not apply PFT/2D filter for top-bad plots (plot all gridcells). Default: natveg filter is ON.')
    args = parser.parse_args()
    
    # Set up input and output paths
    # Resolve relative input against the current working directory
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = input_path.resolve()
    
    if args.output_dir is None:
        output_dir = input_path.parent / "analysis"
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            # Resolve relative output against the current working directory
            output_dir = output_dir.resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper: auto-detect training config (cnp_config.json) near the run directory
    def _auto_detect_training_config(start_dir: Path) -> Optional[Path]:
        for parent in [start_dir] + list(start_dir.parents):
            cfg = parent / 'cnp_config.json'
            if cfg.exists():
                return cfg
        return None
    
    # Load expected variables from training configuration to ensure full coverage in reports
    expected_vars: List[str] = []
    training_cfg_path: Optional[Path] = None
    try:
        training_cfg_path = Path(args.training_config) if args.training_config else _auto_detect_training_config(input_path.parent)
        if training_cfg_path and training_cfg_path.exists():
            with open(training_cfg_path, 'r') as f:
                cfg = json.load(f)
            di = cfg.get('data_info', {}) if isinstance(cfg, dict) else {}
            # Prefer scalar target list if available; fallback to input scalar list
            scalar_targets = di.get('y_list_scalar_columns', []) or di.get('x_list_scalar_columns', []) or []
            # Strip any leading Y_ prefixes
            scalar_vars = [str(v)[2:] if isinstance(v, str) and v.startswith('Y_') else str(v) for v in scalar_targets]
            pft1d_vars = [str(v) for v in di.get('variables_1d_pft', []) or []]
            soil2d_vars = [str(v) for v in di.get('x_list_columns_2d', []) or []]
            expected_vars = list(dict.fromkeys(scalar_vars + pft1d_vars + soil2d_vars))
            if expected_vars:
                print(f"Loaded {len(expected_vars)} expected variables from training config: {training_cfg_path}")
        else:
            print("Warning: Could not locate cnp_config.json to derive full variable list. Proceeding with variables present in stats.")
    except Exception as e:
        print(f"Warning: Failed to parse training config for expected variables: {e}")
    
    # Define thresholds for categorization
    thresholds = {
        'good': {
            'r2': args.r2_good,
            'rmse_rel': args.rmse_good,
            'mae_rel': args.mae_good,
        },
        'ok': {
            'r2': args.r2_ok,
            'rmse_rel': args.rmse_ok,
            'mae_rel': args.mae_ok,
        }
    }
    
    print(f"Reading validation statistics from {input_path}")
    df = pd.read_csv(input_path)
    
    # Normalize variable naming: strip leading 'Y_' from variable names (targets)
    if 'variable' in df.columns:
        try:
            df['variable'] = df['variable'].apply(lambda v: v[2:] if isinstance(v, str) and v.startswith('Y_') else v)
        except Exception:
            pass
    
    # Function to categorize prediction quality
    def categorize_prediction(row):
        # Calculate relative metrics (normalized by data range)
        gt_range = row['gt_max'] - row['gt_min']
        
        # Handle zero range (constant values)
        if gt_range == 0:
            if row['rmse'] == 0 and row['mae'] == 0:
                return 'good'  # Perfect prediction for constant values
            else:
                return 'bad'   # Any error on constant values is bad
        
        rmse_rel = row['rmse'] / gt_range
        mae_rel = row['mae'] / gt_range
        
        # Apply thresholds for categorization
        if (row['r2'] >= thresholds['good']['r2'] and 
            rmse_rel <= thresholds['good']['rmse_rel'] and 
            mae_rel <= thresholds['good']['mae_rel']):
            return 'good'
        elif (row['r2'] >= thresholds['ok']['r2'] and 
              rmse_rel <= thresholds['ok']['rmse_rel'] and 
              mae_rel <= thresholds['ok']['mae_rel']):
            return 'ok'
        else:
            return 'bad'
    
    # Add a quality category column
    print("Categorizing predictions...")
    df['prediction_quality'] = df.apply(categorize_prediction, axis=1)
    
    # Filter out rows that are just coordinates (Longitude, Latitude)
    # Be robust to files without a 'pft' column
    if 'pft' not in df.columns:
        df['pft'] = ''
    coord_labels = {'Longitude', 'Latitude'}
    mask_pft = ~df['pft'].isin(coord_labels) if 'pft' in df.columns else True
    mask_var = ~df['variable'].isin(coord_labels) if 'variable' in df.columns else True
    analysis_df = df[mask_pft & mask_var].copy()

    # Pre-compute relative errors for later exports/reports
    analysis_df['gt_range'] = analysis_df['gt_max'] - analysis_df['gt_min']
    analysis_df['rmse_rel'] = np.where(analysis_df['gt_range'] > 0,
                                       analysis_df['rmse'] / analysis_df['gt_range'],
                                       np.nan)
    analysis_df['mae_rel'] = np.where(analysis_df['gt_range'] > 0,
                                      analysis_df['mae'] / analysis_df['gt_range'],
                                      np.nan)
    
    # Create summary by variable
    variable_summary = analysis_df.groupby(['variable', 'prediction_quality']).size().unstack(fill_value=0)
    
    # Ensure all expected variables appear in the summary (even if missing from CSV)
    if expected_vars:
        # Add any missing variables as zero rows
        for v in expected_vars:
            if v not in variable_summary.index:
                variable_summary.loc[v, :] = 0
    
    # Calculate percentages
    variable_summary['total'] = variable_summary.sum(axis=1)
    for category in ['good', 'ok', 'bad']:
        if category in variable_summary.columns:
            variable_summary[f'{category}_pct'] = (variable_summary[category] / variable_summary['total'] * 100).round(1)
    
    # Sort by percentage of good predictions
    if 'good_pct' in variable_summary.columns:
        variable_summary = variable_summary.sort_values(by='good_pct', ascending=False)
    
    # Save the detailed results
    print(f"Saving detailed quality assessment to {output_dir / 'detailed_quality_assessment.csv'}")
    df.to_csv(output_dir / "detailed_quality_assessment.csv", index=False)

    # Save detailed bad predictions
    bad_df = analysis_df[analysis_df['prediction_quality'] == 'bad'].copy()
    if args.export_bad and not bad_df.empty:
        bad_csv_path = output_dir / "bad_predictions_detailed.csv"
        bad_df.to_csv(bad_csv_path, index=False)
        print(f"Saved detailed bad predictions to: {bad_csv_path}")
    
    # Save the variable summary
    print(f"Saving variable quality summary to {output_dir / 'variable_quality_summary.csv'}")
    variable_summary.to_csv(output_dir / "variable_quality_summary.csv")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Stacked bar chart of prediction quality by variable
    plt.figure(figsize=(14, 10))
    pivot_df = analysis_df.pivot_table(
        index='variable', 
        columns='prediction_quality', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Calculate percentages for the chart
    pivot_total = pivot_df.sum(axis=1)
    for col in pivot_df.columns:
        pivot_df[col] = (pivot_df[col] / pivot_total * 100).round(1)
    
    # Ensure all expected variables appear in the chart
    if expected_vars:
        for v in expected_vars:
            if v not in pivot_df.index:
                pivot_df.loc[v, :] = 0
    
    # Sort by 'good' percentage if it exists
    if 'good' in pivot_df.columns:
        pivot_df = pivot_df.sort_values(by='good', ascending=False)
    
    # Set color map
    colors = {'good': '#2ecc71', 'ok': '#f39c12', 'bad': '#e74c3c'}
    color_list = [colors.get(x, 'gray') for x in pivot_df.columns]
    
    # Plot the stacked bar chart
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(14, 10), color=color_list)
    plt.title('Prediction Quality by Variable', fontsize=16)
    plt.xlabel('Variable', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=90)
    plt.legend(title='Quality')
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_quality_by_variable.png", dpi=300)
    
    # 2. Pie chart of overall prediction quality
    plt.figure(figsize=(8, 8))
    quality_counts = analysis_df['prediction_quality'].value_counts()
    plt.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%', 
            colors=[colors.get(x, 'gray') for x in quality_counts.index],
            explode=[0.05 if x == 'bad' else 0 for x in quality_counts.index])
    plt.title('Overall Prediction Quality Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "overall_prediction_quality.png", dpi=300)
    
    # 3. Scatter plot of R² vs Relative RMSE for all predictions
    plt.figure(figsize=(12, 10))
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    scatter_df = analysis_df.copy()
    # Ensure relative RMSE exists (it does from pre-compute; keep guard for safety)
    if 'rmse_rel' not in scatter_df.columns:
        scatter_df['rmse_rel'] = scatter_df.apply(
            lambda row: row['rmse'] / (row['gt_max'] - row['gt_min']) if row['gt_max'] > row['gt_min'] else 0,
            axis=1
        )

    # Optional diagnostics about what will be plotted
    if args.print_scatter_stats:
        r2_vals = scatter_df['r2'].replace([np.inf, -np.inf], np.nan).dropna()
        rmse_rel_vals = scatter_df['rmse_rel'].replace([np.inf, -np.inf], np.nan).dropna()
        total_points = len(scatter_df)
        valid_r2 = len(r2_vals)
        valid_rmse_rel = len(rmse_rel_vals)
        print(f"Scatter diagnostics: total_points={total_points}, valid_r2={valid_r2}, valid_rmse_rel={valid_rmse_rel}")
        if valid_r2 > 0:
            print(f"  R²: min={r2_vals.min():.6f}, max={r2_vals.max():.6f}, count_>0={(r2_vals > 0).sum()}, count_>=0={(r2_vals >= 0).sum()}")
        if valid_rmse_rel > 0:
            print(f"  RMSE_rel: min={rmse_rel_vals.min():.6f}, max={rmse_rel_vals.max():.6f}")
    
    # Create scatter plot: quality colors (Good/OK/Bad); PFT variables with coverage <2% in a different color
    from matplotlib.lines import Line2D
    has_pft_low = 'pft_pct_low' in scatter_df.columns and scatter_df['pft_pct_low'].any()
    if has_pft_low:
        # Plot non-low-PFT points with quality colors
        mask_normal = ~scatter_df['pft_pct_low'].fillna(False)
        if mask_normal.any():
            plt.scatter(
                scatter_df.loc[mask_normal, 'r2'],
                scatter_df.loc[mask_normal, 'rmse_rel'],
                c=scatter_df.loc[mask_normal, 'prediction_quality'].map({'good': 0, 'ok': 1, 'bad': 2}),
                cmap=plt.cm.viridis,
                alpha=0.7,
                s=50,
                label='_nolegend_'
            )
        # Overlay PFT coverage <2% points in distinct color
        mask_low = scatter_df['pft_pct_low'].fillna(False)
        if mask_low.any():
            plt.scatter(
                scatter_df.loc[mask_low, 'r2'],
                scatter_df.loc[mask_low, 'rmse_rel'],
                c='gray',
                alpha=0.8,
                s=60,
                edgecolors='black',
                linewidths=0.5,
                label='PFT coverage <2%',
                zorder=5
            )
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0), markersize=10, label='Good'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0.5), markersize=10, label='OK'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(1.0), markersize=10, label='Bad'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, label='PFT coverage <2%'),
        ]
    else:
        scatter = plt.scatter(
            scatter_df['r2'],
            scatter_df['rmse_rel'],
            c=scatter_df['prediction_quality'].map({'good': 0, 'ok': 1, 'bad': 2}),
            cmap=plt.cm.viridis,
            alpha=0.7,
            s=50
        )
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0), markersize=10, label='Good'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0.5), markersize=10, label='OK'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(1.0), markersize=10, label='Bad'),
        ]
    
    # Add threshold lines
    plt.axhline(y=thresholds['good']['rmse_rel'], color='green', linestyle='--', alpha=0.7)
    plt.axhline(y=thresholds['ok']['rmse_rel'], color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=thresholds['good']['r2'], color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=thresholds['ok']['r2'], color='orange', linestyle='--', alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('R²', fontsize=14)
    plt.ylabel('Relative RMSE (RMSE / Range)', fontsize=14)
    plt.title('R² vs Relative RMSE for All Predictions', fontsize=16)
    
    # Optionally force x-axis limits for clarity
    if args.force_xlim_01:
        plt.xlim(0, 1)
    
    plt.legend(handles=legend_elements)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "r2_vs_rmse.png", dpi=300)
    
    # 4. Optionally generate restricted plots (top-bad or worst-only) into a subfolder using the validation plotting utility
    top_bad_plot_count = 0
    if args.top_bad_plots:
        try:
            results_dir = str(input_path.parent)
            top_bad_out = str((output_dir / 'top_bad_plots').resolve())
            (output_dir / 'top_bad_plots').mkdir(parents=True, exist_ok=True)

            # Pre-write "Top variables by bad-count" to the report so the plot script can read it
            # (the full report is written later; this ensures top_bad selection is available when plotting)
            if not args.worst_only and not bad_df.empty:
                try:
                    report_path = output_dir / 'quality_summary_report.txt'
                    with open(report_path, 'w') as _pref:
                        _pref.write("# Prediction Quality Summary Report\n\n")
                        _pref.write("Top variables by bad-count (with PFT indices or layer numbers):\n")
                        bad_by_var = bad_df.groupby('variable').size().sort_values(ascending=False).head(25)
                        for v, c in bad_by_var.items():
                            sub = bad_df[bad_df['variable'] == v]
                            pft_indices = []
                            for val in sub['pft'].dropna().unique():
                                if isinstance(val, str) and 'pft' in val:
                                    try:
                                        idx = ''.join(ch for ch in val.split('pft')[-1] if ch.isdigit())
                                        if idx:
                                            pft_indices.append(int(idx))
                                    except Exception:
                                        continue
                            pft_indices = sorted(set(pft_indices))
                            layer_numbers = []
                            for lay in sub['layer'].dropna().unique():
                                try:
                                    li = int(lay) if float(lay).is_integer() else float(lay)
                                    layer_numbers.append(li)
                                except Exception:
                                    continue
                            layer_numbers = sorted(set(layer_numbers))
                            details_parts = []
                            if pft_indices:
                                details_parts.append("pfts: " + ", ".join(str(i) for i in pft_indices))
                            if layer_numbers:
                                details_parts.append("layers: " + ", ".join(str(i) for i in layer_numbers))
                            details = ("; " + " ".join(details_parts)) if details_parts else ""
                            _pref.write(f"  {v}: {c}{details}\n")
                        _pref.write("\n")
                    print(f"Wrote top-bad section for plot selection: {report_path}")
                except Exception as _e:
                    print(f"Warning: Failed to pre-write top-bad section: {_e}")

            # If worst-only requested, pre-write a minimal 'Variables with Worst Predictions' section
            if args.worst_only:
                try:
                    tmp_report_path = output_dir / 'quality_summary_report.txt'
                    with open(tmp_report_path, 'w') as _pref:
                        _pref.write("# Prediction Quality Summary Report\n\n")
                        _pref.write("## Variables with Worst Predictions\n")
                        _vw = variable_summary.copy()
                        if 'good_pct' in _vw.columns:
                            _vw['good_pct'] = _vw['good_pct'].fillna(0)
                        _worst = _vw.nsmallest(15, 'good_pct')
                        for _var_name, _row in _worst.iterrows():
                            _pref.write(f"{_var_name}: {_row.get('good_pct', 0):.1f}% good, {_row.get('ok_pct', 0):.1f}% ok, {_row.get('bad_pct', 0):.1f}% bad\n")
                        _pref.write("\n")
                    print(f"Wrote minimal worst-variables section for selection: {tmp_report_path}")
                except Exception as _e:
                    print(f"Warning: Failed to pre-write worst-variables section for plotting selection: {_e}")

            # Protect the input validation_stats.csv from being overwritten by the plotting utility
            original_bytes = None
            try:
                if input_path.exists():
                    original_bytes = input_path.read_bytes()
            except Exception:
                original_bytes = None
            # Dynamically import cnp_result_validationplot without relying on PYTHONPATH
            plot_mod_path = (output_dir.parent.parent / 'scripts' / 'cnp_result_validationplot.py')
            # If running from repo root, construct direct path as fallback
            if not plot_mod_path.exists():
                plot_mod_path = Path(__file__).parent / 'cnp_result_validationplot.py'
            spec = importlib.util.spec_from_file_location('cnp_plot_mod', str(plot_mod_path))
            mod = importlib.util.module_from_spec(spec)
            sys.modules['cnp_plot_mod'] = mod
            assert spec.loader is not None
            try:
                spec.loader.exec_module(mod)
                if hasattr(mod, 'main_with_flag'):
                    mod.main_with_flag(results_dir, plot_scatter=True, plot_loss=False,
                                       top_bad_only=not args.worst_only,
                                       top_bad_report=str(output_dir / 'quality_summary_report.txt'),
                                       plots_dir_override=top_bad_out,
                                       worst_only=args.worst_only,
                                       use_natveg_filter=getattr(args, 'natveg_filter', True))
                    print(f"Top-bad plots saved to: {top_bad_out}")
                    try:
                        # Count the number of PNGs generated for quick reporting
                        top_bad_plot_count = len(list((output_dir / 'top_bad_plots').rglob('*.png')))
                        print(f"Top-bad plot count: {top_bad_plot_count}")
                    except Exception:
                        top_bad_plot_count = 0
                else:
                    print("Warning: cnp_result_validationplot.main_with_flag not found; skipping top-bad plots")
            finally:
                # Restore original validation_stats.csv to prevent any overwrite
                try:
                    if original_bytes is not None:
                        with open(input_path, 'wb') as _f:
                            _f.write(original_bytes)
                        print(f"Restored original validation_stats.csv after generating top-bad plots: {input_path}")
                except Exception as _e:
                    print(f"Warning: Failed to restore original validation_stats.csv: {_e}")
        except Exception as e:
            print(f"Warning: Failed to generate top-bad plots: {e}")
    
    # Generate a comprehensive summary report
    print(f"Generating summary report to {output_dir / 'quality_summary_report.txt'}")
    with open(output_dir / "quality_summary_report.txt", "w") as f:
        f.write("# Prediction Quality Summary Report\n\n")
        
        # Overall statistics
        total_predictions = len(analysis_df)
        # Variables analyzed (unique variable names in stats; if expected list provided, report both)
        analyzed_variables = sorted(set(analysis_df['variable'].unique()))
        num_analyzed_variables = len(analyzed_variables)
        total_expected_variables = len(expected_vars) if expected_vars else None
        good_count = analysis_df[analysis_df['prediction_quality'] == 'good'].shape[0]
        ok_count = analysis_df[analysis_df['prediction_quality'] == 'ok'].shape[0]
        bad_count = analysis_df[analysis_df['prediction_quality'] == 'bad'].shape[0]
        
        f.write(f"## Overall Statistics\n")
        f.write(f"Variables analyzed: {num_analyzed_variables}")
        if total_expected_variables is not None:
            f.write(f" (of {total_expected_variables} expected from training config)")
        f.write("\n")
        if args.top_bad_plots:
            f.write(f"Top-bad plots generated: {top_bad_plot_count}\n")
        f.write(f"Total predictions analyzed: {total_predictions}\n")
        f.write(f"Good predictions: {good_count} ({good_count/total_predictions*100:.1f}%)\n")
        f.write(f"OK predictions: {ok_count} ({ok_count/total_predictions*100:.1f}%)\n")
        f.write(f"Bad predictions: {bad_count} ({bad_count/total_predictions*100:.1f}%)\n\n")
        
        f.write("## Classification Thresholds Used\n")
        f.write(f"Good: R² ≥ {thresholds['good']['r2']}, Relative RMSE ≤ {thresholds['good']['rmse_rel']}, Relative MAE ≤ {thresholds['good']['mae_rel']}\n")
        f.write(f"OK: R² ≥ {thresholds['ok']['r2']}, Relative RMSE ≤ {thresholds['ok']['rmse_rel']}, Relative MAE ≤ {thresholds['ok']['mae_rel']}\n")
        f.write(f"Bad: Below OK thresholds\n\n")

        # Bad predictions summary and details
        f.write("## Bad Predictions Summary\n")
        if bad_df.empty:
            f.write("No bad predictions found.\n\n")
        else:
            # Counts by type
            f.write("Bad predictions by type:\n")
            bad_by_type = bad_df.groupby('type').size().sort_values(ascending=False)
            for t, c in bad_by_type.items():
                f.write(f"  {t}: {c}\n")
            f.write("\nTop variables by bad-count (with PFT indices or layer numbers):\n")
            bad_by_var = bad_df.groupby('variable').size().sort_values(ascending=False).head(20)
            for v, c in bad_by_var.items():
                sub = bad_df[bad_df['variable'] == v]
                # Collect pft indices if present (1D); parse trailing digits after 'pft'
                pft_indices = []
                for val in sub['pft'].dropna().unique():
                    if isinstance(val, str) and 'pft' in val:
                        try:
                            idx = ''.join(ch for ch in val.split('pft')[-1] if ch.isdigit())
                            if idx:
                                pft_indices.append(int(idx))
                        except Exception:
                            continue
                pft_indices = sorted(set(pft_indices))
                # Collect layer numbers if present (2D)
                layer_numbers = []
                for lay in sub['layer'].dropna().unique():
                    try:
                        # cast to int if integral
                        li = int(lay) if float(lay).is_integer() else float(lay)
                        layer_numbers.append(li)
                    except Exception:
                        continue
                layer_numbers = sorted(set(layer_numbers))

                details_parts = []
                if pft_indices:
                    details_parts.append("pfts: " + ", ".join(str(i) for i in pft_indices))
                if layer_numbers:
                    details_parts.append("layers: " + ", ".join(str(i) for i in layer_numbers))
                details = ("; " + " ".join(details_parts)) if details_parts else ""
                f.write(f"  {v}: {c}{details}\n")
            f.write("\n")

            # Optional: long detailed rows (disabled by default)
            if args.include_bad_details_text:
                f.write(f"## Detailed Bad Predictions (first {args.bad_text_limit})\n")
                printable = bad_df.copy()
                # Order by worst first: lowest R², then highest relative RMSE
                printable = printable.sort_values(by=['r2','rmse_rel'], ascending=[True, False])
                if len(printable) > args.bad_text_limit:
                    printable = printable.head(args.bad_text_limit)
                for _, row in printable.iterrows():
                    f.write(
                        f"- {row.get('type','')}, {row.get('variable','')}, {row.get('pft','')}, layer={row.get('layer','')}"
                        f", r2={row.get('r2',np.nan):.6f}, rmse_rel={row.get('rmse_rel',np.nan):.6f}, "
                        f"mae_rel={row.get('mae_rel',np.nan):.6f}, rmse={row.get('rmse',np.nan):.6f}, mae={row.get('mae',np.nan):.6f}\n"
                    )
                f.write("\n")
                if args.export_bad:
                    f.write("Full list saved to bad_predictions_detailed.csv\n\n")
        
        f.write("## Variables with Best Predictions\n")
        if 'good_pct' in variable_summary.columns:
            # Treat NaN as 0 for ranking
            _vs = variable_summary.copy()
            _vs['good_pct'] = _vs['good_pct'].fillna(0)
            _vs['ok_pct'] = _vs.get('ok_pct', 0)
            _vs['bad_pct'] = _vs.get('bad_pct', 0)
            best_vars = _vs.nlargest(15, 'good_pct')
            for var_name, row in best_vars.iterrows():
                f.write(f"{var_name}: {row.get('good_pct', 0):.1f}% good, {row.get('ok_pct', 0):.1f}% ok, {row.get('bad_pct', 0):.1f}% bad\n")
        
        f.write("\n## Variables with Worst Predictions\n")
        if 'good_pct' in variable_summary.columns:
            _vs2 = variable_summary.copy()
            _vs2['good_pct'] = _vs2['good_pct'].fillna(0)
            if 'ok_pct' in _vs2.columns:
                _vs2['ok_pct'] = _vs2['ok_pct'].fillna(0)
            else:
                _vs2['ok_pct'] = 0
            if 'bad_pct' in _vs2.columns:
                _vs2['bad_pct'] = _vs2['bad_pct'].fillna(0)
            else:
                _vs2['bad_pct'] = 0
            
            # Apply filtering based on user options
            _vs2_filtered = _vs2.copy()
            
            # Filter 1: By default, only include variables with bad predictions (bad_pct > 0)
            if args.worst_filter_bad_only:
                if 'bad_pct' in _vs2_filtered.columns:
                    _vs2_filtered = _vs2_filtered[_vs2_filtered['bad_pct'] > 0].copy()
            
            # Filter 2: Include variables with good_pct below threshold if specified
            if args.worst_min_good_pct is not None:
                _vs2_filtered = _vs2_filtered[_vs2_filtered['good_pct'] < args.worst_min_good_pct].copy()
            
            # Filter 3: Include specific variables if list provided
            if args.worst_vars_list:
                var_list = [v.strip() for v in args.worst_vars_list.split(',')]
                # Add specified variables even if they don't meet other filters
                specified_vars = _vs2[_vs2.index.isin(var_list)].copy()
                _vs2_filtered = pd.concat([_vs2_filtered, specified_vars]).drop_duplicates()
            
            # Sort by bad_pct (highest first), then by good_pct (lowest first) for tie-breaking
            if 'bad_pct' in _vs2_filtered.columns and len(_vs2_filtered) > 0:
                worst_vars = _vs2_filtered.nlargest(15, 'bad_pct').nsmallest(15, 'good_pct')
            elif len(_vs2_filtered) > 0:
                worst_vars = _vs2_filtered.nsmallest(15, 'good_pct')
            else:
                worst_vars = pd.DataFrame()
            
            for var_name, row in worst_vars.iterrows():
                f.write(f"{var_name}: {row.get('good_pct', 0):.1f}% good, {row.get('ok_pct', 0):.1f}% ok, {row.get('bad_pct', 0):.1f}% bad\n")
        
        # Group all variables by dominant quality category
        f.write("\n## All Variables Grouped by Quality Category\n")
        if 'good_pct' in variable_summary.columns:
            _vs3 = variable_summary.copy()
            _vs3['good_pct'] = _vs3['good_pct'].fillna(0)
            if 'ok_pct' in _vs3.columns:
                _vs3['ok_pct'] = _vs3['ok_pct'].fillna(0)
            else:
                _vs3['ok_pct'] = 0
            if 'bad_pct' in _vs3.columns:
                _vs3['bad_pct'] = _vs3['bad_pct'].fillna(0)
            else:
                _vs3['bad_pct'] = 0
            
            # Determine dominant category for each variable (highest percentage)
            def get_dominant_category(row):
                good_pct = row['good_pct'] if 'good_pct' in row.index else 0
                ok_pct = row['ok_pct'] if 'ok_pct' in row.index else 0
                bad_pct = row['bad_pct'] if 'bad_pct' in row.index else 0
                if good_pct >= ok_pct and good_pct >= bad_pct:
                    return 'good'
                elif ok_pct >= bad_pct:
                    return 'ok'
                else:
                    return 'bad'
            
            _vs3['dominant_category'] = _vs3.apply(get_dominant_category, axis=1)
            
            # Group variables by category
            good_vars = _vs3[_vs3['dominant_category'] == 'good'].sort_values('good_pct', ascending=False)
            ok_vars = _vs3[_vs3['dominant_category'] == 'ok'].sort_values('ok_pct', ascending=False)
            bad_vars = _vs3[_vs3['dominant_category'] == 'bad'].sort_values('bad_pct', ascending=False)
            
            f.write(f"\n### Good Variables ({len(good_vars)} total)\n")
            if len(good_vars) > 0:
                for var_name, row in good_vars.iterrows():
                    good_val = row['good_pct'] if 'good_pct' in row.index else 0
                    ok_val = row['ok_pct'] if 'ok_pct' in row.index else 0
                    bad_val = row['bad_pct'] if 'bad_pct' in row.index else 0
                    f.write(f"{var_name}: {good_val:.1f}% good, {ok_val:.1f}% ok, {bad_val:.1f}% bad\n")
            else:
                f.write("None\n")
            
            f.write(f"\n### OK Variables ({len(ok_vars)} total)\n")
            if len(ok_vars) > 0:
                for var_name, row in ok_vars.iterrows():
                    good_val = row['good_pct'] if 'good_pct' in row.index else 0
                    ok_val = row['ok_pct'] if 'ok_pct' in row.index else 0
                    bad_val = row['bad_pct'] if 'bad_pct' in row.index else 0
                    f.write(f"{var_name}: {good_val:.1f}% good, {ok_val:.1f}% ok, {bad_val:.1f}% bad\n")
            else:
                f.write("None\n")
            
            f.write(f"\n### Bad Variables ({len(bad_vars)} total)\n")
            if len(bad_vars) > 0:
                for var_name, row in bad_vars.iterrows():
                    good_val = row['good_pct'] if 'good_pct' in row.index else 0
                    ok_val = row['ok_pct'] if 'ok_pct' in row.index else 0
                    bad_val = row['bad_pct'] if 'bad_pct' in row.index else 0
                    f.write(f"{var_name}: {good_val:.1f}% good, {ok_val:.1f}% ok, {bad_val:.1f}% bad\n")
            else:
                f.write("None\n")
        
        # Report variables missing from the stats but present in training
        if expected_vars:
            present_vars = set(analysis_df['variable'].unique())
            missing_vars = [v for v in expected_vars if v not in present_vars]
            f.write("\n## Variables Missing from validation_stats.csv (listed in training config)\n")
            if missing_vars:
                f.write(f"Count: {len(missing_vars)}\n")
                # Limit long lists in text to keep report concise
                preview = missing_vars[:100]
                f.write("" + ", ".join(preview) + (" ..." if len(missing_vars) > 100 else "") + "\n")
            else:
                f.write("None\n")
    
    # Generate an HTML report for better visualization
    print(f"Generating HTML report to {output_dir / 'prediction_quality_report.html'}")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Quality Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .summary-box {{
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
            }}
            .stat-item {{
                text-align: center;
                padding: 15px;
                border-radius: 5px;
            }}
            .good {{
                background-color: rgba(46, 204, 113, 0.2);
                border: 1px solid #2ecc71;
            }}
            .ok {{
                background-color: rgba(243, 156, 18, 0.2);
                border: 1px solid #f39c12;
            }}
            .bad {{
                background-color: rgba(231, 76, 60, 0.2);
                border: 1px solid #e74c3c;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .image-container {{
                margin: 30px 0;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Prediction Quality Report</h1>
            
            <div class="summary-box">
                <h2>Overall Statistics</h2>
                <div class="stats">
                    <div class="stat-item good">
                        <h3>Good Predictions</h3>
                        <p>{good_count} ({good_count/total_predictions*100:.1f}%)</p>
                    </div>
                    <div class="stat-item ok">
                        <h3>OK Predictions</h3>
                        <p>{ok_count} ({ok_count/total_predictions*100:.1f}%)</p>
                    </div>
                    <div class="stat-item bad">
                        <h3>Bad Predictions</h3>
                        <p>{bad_count} ({bad_count/total_predictions*100:.1f}%)</p>
                    </div>
                </div>
                
                <h3>Classification Thresholds</h3>
                <ul>
                    <li><strong>Good:</strong> R² ≥ {thresholds['good']['r2']}, Relative RMSE ≤ {thresholds['good']['rmse_rel']}, Relative MAE ≤ {thresholds['good']['mae_rel']}</li>
                    <li><strong>OK:</strong> R² ≥ {thresholds['ok']['r2']}, Relative RMSE ≤ {thresholds['ok']['rmse_rel']}, Relative MAE ≤ {thresholds['ok']['mae_rel']}</li>
                    <li><strong>Bad:</strong> Below OK thresholds</li>
                </ul>
            </div>
            
            <div class="image-container">
                <h2>Visualization of Overall Results</h2>
                <img src="overall_prediction_quality.png" alt="Overall Prediction Quality Distribution">
            </div>
            
            <div class="image-container">
                <h2>Prediction Quality by Variable</h2>
                <img src="prediction_quality_by_variable.png" alt="Prediction Quality by Variable">
            </div>
            
            <div class="image-container">
                <h2>R² vs Relative RMSE</h2>
                <img src="r2_vs_rmse.png" alt="R² vs Relative RMSE">
            </div>
            
            <h2>Detailed Bad Predictions (first {args.bad_html_limit})</h2>
            <table>
                <tr>
                    <th>Type</th>
                    <th>Variable</th>
                    <th>PFT/Soil</th>
                    <th>Layer</th>
                    <th>R²</th>
                    <th>RMSE_rel</th>
                    <th>MAE_rel</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                </tr>
    """

    # Insert bad predictions table rows (limited)
    if not bad_df.empty:
        bad_html_rows = bad_df.copy().sort_values(by=['r2','rmse_rel'], ascending=[True, False])
        if len(bad_html_rows) > args.bad_html_limit:
            bad_html_rows = bad_html_rows.head(args.bad_html_limit)
        for _, row in bad_html_rows.iterrows():
            html_content += f"""
                <tr>
                    <td>{row.get('type','')}</td>
                    <td>{row.get('variable','')}</td>
                    <td>{row.get('pft','')}</td>
                    <td>{row.get('layer','')}</td>
                    <td>{row.get('r2',float('nan')):.6f}</td>
                    <td>{row.get('rmse_rel',float('nan')):.6f}</td>
                    <td>{row.get('mae_rel',float('nan')):.6f}</td>
                    <td>{row.get('rmse',float('nan')):.6f}</td>
                    <td>{row.get('mae',float('nan')):.6f}</td>
                </tr>
            """
    
    # Close bad predictions table and proceed with the rest of the report
    html_content += """
            </table>
            
            <h2>Best Performing Variables</h2>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Good (%)</th>
                    <th>OK (%)</th>
                    <th>Bad (%)</th>
                </tr>
    """
    
    # Add best variables
    best_vars = variable_summary.nlargest(15, 'good_pct')
    for var_name, row in best_vars.iterrows():
        html_content += f"""
                <tr>
                    <td>{var_name}</td>
                    <td>{row.get('good_pct', 0):.1f}%</td>
                    <td>{row.get('ok_pct', 0):.1f}%</td>
                    <td>{row.get('bad_pct', 0):.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Worst Performing Variables</h2>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Good (%)</th>
                    <th>OK (%)</th>
                    <th>Bad (%)</th>
                </tr>
    """
    
    # Add worst variables
    _vw = variable_summary.copy()
    if 'good_pct' in _vw.columns:
        _vw['good_pct'] = _vw['good_pct'].fillna(0)
    worst_vars = _vw.nsmallest(15, 'good_pct')
    for var_name, row in worst_vars.iterrows():
        html_content += f"""
                <tr>
                    <td>{var_name}</td>
                    <td>{row.get('good_pct', 0):.1f}%</td>
                    <td>{row.get('ok_pct', 0):.1f}%</td>
                    <td>{row.get('bad_pct', 0):.1f}%</td>
                </tr>
        """
    
    # Add all variables grouped by quality category
    html_content += """
            </table>
            
            <h2>All Variables Grouped by Quality Category</h2>
    """
    
    # Prepare grouped variables data
    _vs_html = variable_summary.copy()
    if 'good_pct' in _vs_html.columns:
        _vs_html['good_pct'] = _vs_html['good_pct'].fillna(0)
        if 'ok_pct' in _vs_html.columns:
            _vs_html['ok_pct'] = _vs_html['ok_pct'].fillna(0)
        else:
            _vs_html['ok_pct'] = 0
        if 'bad_pct' in _vs_html.columns:
            _vs_html['bad_pct'] = _vs_html['bad_pct'].fillna(0)
        else:
            _vs_html['bad_pct'] = 0
        
        # Determine dominant category
        def get_dominant_category_html(row):
            good_pct = row['good_pct'] if 'good_pct' in row.index else 0
            ok_pct = row['ok_pct'] if 'ok_pct' in row.index else 0
            bad_pct = row['bad_pct'] if 'bad_pct' in row.index else 0
            if good_pct >= ok_pct and good_pct >= bad_pct:
                return 'good'
            elif ok_pct >= bad_pct:
                return 'ok'
            else:
                return 'bad'
        
        _vs_html['dominant_category'] = _vs_html.apply(get_dominant_category_html, axis=1)
        
        good_vars_html = _vs_html[_vs_html['dominant_category'] == 'good'].sort_values('good_pct', ascending=False)
        ok_vars_html = _vs_html[_vs_html['dominant_category'] == 'ok'].sort_values('ok_pct', ascending=False)
        bad_vars_html = _vs_html[_vs_html['dominant_category'] == 'bad'].sort_values('bad_pct', ascending=False)
        
        # Good variables table
        html_content += f"""
            <h3>Good Variables ({len(good_vars_html)} total)</h3>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Good (%)</th>
                    <th>OK (%)</th>
                    <th>Bad (%)</th>
                </tr>
        """
        if len(good_vars_html) > 0:
            for var_name, row in good_vars_html.iterrows():
                good_val = row['good_pct'] if 'good_pct' in row.index else 0
                ok_val = row['ok_pct'] if 'ok_pct' in row.index else 0
                bad_val = row['bad_pct'] if 'bad_pct' in row.index else 0
                html_content += f"""
                    <tr>
                        <td>{var_name}</td>
                        <td>{good_val:.1f}%</td>
                        <td>{ok_val:.1f}%</td>
                        <td>{bad_val:.1f}%</td>
                    </tr>
                """
        else:
            html_content += "<tr><td colspan='4'>None</td></tr>"
        html_content += "</table>"
        
        # OK variables table
        html_content += f"""
            <h3>OK Variables ({len(ok_vars_html)} total)</h3>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Good (%)</th>
                    <th>OK (%)</th>
                    <th>Bad (%)</th>
                </tr>
        """
        if len(ok_vars_html) > 0:
            for var_name, row in ok_vars_html.iterrows():
                good_val = row['good_pct'] if 'good_pct' in row.index else 0
                ok_val = row['ok_pct'] if 'ok_pct' in row.index else 0
                bad_val = row['bad_pct'] if 'bad_pct' in row.index else 0
                html_content += f"""
                    <tr>
                        <td>{var_name}</td>
                        <td>{good_val:.1f}%</td>
                        <td>{ok_val:.1f}%</td>
                        <td>{bad_val:.1f}%</td>
                    </tr>
                """
        else:
            html_content += "<tr><td colspan='4'>None</td></tr>"
        html_content += "</table>"
        
        # Bad variables table
        html_content += f"""
            <h3>Bad Variables ({len(bad_vars_html)} total)</h3>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Good (%)</th>
                    <th>OK (%)</th>
                    <th>Bad (%)</th>
                </tr>
        """
        if len(bad_vars_html) > 0:
            for var_name, row in bad_vars_html.iterrows():
                good_val = row['good_pct'] if 'good_pct' in row.index else 0
                ok_val = row['ok_pct'] if 'ok_pct' in row.index else 0
                bad_val = row['bad_pct'] if 'bad_pct' in row.index else 0
                html_content += f"""
                    <tr>
                        <td>{var_name}</td>
                        <td>{good_val:.1f}%</td>
                        <td>{ok_val:.1f}%</td>
                        <td>{bad_val:.1f}%</td>
                    </tr>
                """
        else:
            html_content += "<tr><td colspan='4'>None</td></tr>"
        html_content += "</table>"
    
    # Add missing variables section if available
    if expected_vars:
        present_vars = set(analysis_df['variable'].unique())
        missing_vars = [v for v in expected_vars if v not in present_vars]
        html_content += """
            </table>
            <h2>Variables Missing from validation_stats.csv (in training config)</h2>
            <div class="summary-box">
        """
        if missing_vars:
            # Show as a comma-separated list (trim if very long)
            preview = missing_vars[:300]
            remainder = len(missing_vars) - len(preview)
            html_content += f"<p>Count: {len(missing_vars)}</p>"
            html_content += f"<p>{', '.join(preview)}{' ...' if remainder > 0 else ''}</p>"
        else:
            html_content += "<p>None</p>"
        html_content += """
            </div>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_dir / "prediction_quality_report.html", "w") as f:
        f.write(html_content)
    
    print("\nAnalysis complete. Results saved to", output_dir)
    print("\nOverall Prediction Quality Summary:")
    quality_counts = analysis_df['prediction_quality'].value_counts()
    for quality, count in quality_counts.items():
        print(f"{quality}: {count} ({count/len(analysis_df)*100:.1f}%)")
    
    print("\nTop 5 Best Predicted Variables:")
    _vb = variable_summary.copy()
    if 'good_pct' in _vb.columns:
        _vb['good_pct'] = _vb['good_pct'].fillna(0)
    best_vars = _vb.nlargest(5, 'good_pct')
    for var_name, row in best_vars.iterrows():
        print(f"{var_name}: {row.get('good_pct', 0):.1f}% good")
    
    print("\nTop 5 Worst Predicted Variables:")
    _vw5 = variable_summary.copy()
    if 'good_pct' in _vw5.columns:
        _vw5['good_pct'] = _vw5['good_pct'].fillna(0)
    worst_vars = _vw5.nsmallest(5, 'good_pct')
    for var_name, row in worst_vars.iterrows():
        print(f"{var_name}: {row.get('good_pct', 0):.1f}% good, {row.get('bad_pct', 0):.1f}% bad")

if __name__ == "__main__":
    main()
