#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate prediction quality report from validation statistics')
    parser.add_argument('--input', default="./validation_stats.csv",
                        help='Path to validation statistics CSV file')
    parser.add_argument('--output-dir', default=None,
                        help='Directory to save output files (default: same directory as input + /analysis)')
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
    analysis_df = df[~df['pft'].isin(['Longitude', 'Latitude'])].copy()

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
    
    # Create scatter plot
    scatter = plt.scatter(
        scatter_df['r2'], 
        scatter_df['rmse_rel'], 
        c=scatter_df['prediction_quality'].map({'good': 0, 'ok': 1, 'bad': 2}),
        cmap=plt.cm.viridis,
        alpha=0.7,
        s=50
    )
    
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
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0), markersize=10, label='Good'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(0.5), markersize=10, label='OK'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(1.0), markersize=10, label='Bad'),
    ]
    plt.legend(handles=legend_elements)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "r2_vs_rmse.png", dpi=300)
    
    # Generate a comprehensive summary report
    print(f"Generating summary report to {output_dir / 'quality_summary_report.txt'}")
    with open(output_dir / "quality_summary_report.txt", "w") as f:
        f.write("# Prediction Quality Summary Report\n\n")
        
        # Overall statistics
        total_predictions = len(analysis_df)
        good_count = analysis_df[analysis_df['prediction_quality'] == 'good'].shape[0]
        ok_count = analysis_df[analysis_df['prediction_quality'] == 'ok'].shape[0]
        bad_count = analysis_df[analysis_df['prediction_quality'] == 'bad'].shape[0]
        
        f.write(f"## Overall Statistics\n")
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
            best_vars = variable_summary.nlargest(15, 'good_pct')
            for var_name, row in best_vars.iterrows():
                f.write(f"{var_name}: {row.get('good_pct', 0):.1f}% good, {row.get('ok_pct', 0):.1f}% ok, {row.get('bad_pct', 0):.1f}% bad\n")
        
        f.write("\n## Variables with Worst Predictions\n")
        if 'good_pct' in variable_summary.columns:
            worst_vars = variable_summary.nsmallest(15, 'good_pct')
            for var_name, row in worst_vars.iterrows():
                f.write(f"{var_name}: {row.get('good_pct', 0):.1f}% good, {row.get('ok_pct', 0):.1f}% ok, {row.get('bad_pct', 0):.1f}% bad\n")
    
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
    worst_vars = variable_summary.nsmallest(15, 'good_pct')
    for var_name, row in worst_vars.iterrows():
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
    best_vars = variable_summary.nlargest(5, 'good_pct')
    for var_name, row in best_vars.iterrows():
        print(f"{var_name}: {row.get('good_pct', 0):.1f}% good")
    
    print("\nTop 5 Worst Predicted Variables:")
    worst_vars = variable_summary.nsmallest(5, 'good_pct')
    for var_name, row in worst_vars.iterrows():
        print(f"{var_name}: {row.get('good_pct', 0):.1f}% good, {row.get('bad_pct', 0):.1f}% bad")

if __name__ == "__main__":
    main()
