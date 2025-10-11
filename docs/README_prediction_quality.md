# Prediction Quality Analysis Tools

This repository contains tools for analyzing the quality of AI model predictions based on validation statistics. The tools categorize predictions as "good", "ok", or "bad" based on statistical metrics like R², RMSE, and MAE.

## Available Tool

**generate_prediction_quality_report.py**: Comprehensive script that analyzes validation statistics, categorizes predictions, and generates detailed reports with visualizations.

## Quick Start

To analyze prediction quality using default settings:

```bash
python scripts/generate_prediction_quality_report.py
```

This will:
1. Read the validation statistics from the default location (`cnp_results/run_20250911_080250/validation_stats.csv`)
2. Categorize each prediction as good, ok, or bad
3. Generate summary reports and visualizations in an `analysis` subdirectory

**Note**: The script can be run from any directory within the project, as it automatically resolves relative paths based on the project root.

## Command Line Arguments

The `generate_prediction_quality_report.py` script accepts the following command line arguments:

```
--input PATH          Path to validation statistics CSV file
                      Default: cnp_results/run_20250911_080250/validation_stats.csv

--output-dir PATH     Directory to save output files
                      Default: Same directory as input + /analysis

--r2-good FLOAT       R² threshold for good predictions
                      Default: 0.9

--r2-ok FLOAT         R² threshold for ok predictions
                      Default: 0.7

--rmse-good FLOAT     Relative RMSE threshold for good predictions
                      Default: 0.1

--rmse-ok FLOAT       Relative RMSE threshold for ok predictions
                      Default: 0.25

--mae-good FLOAT      Relative MAE threshold for good predictions
                      Default: 0.1

--mae-ok FLOAT        Relative MAE threshold for ok predictions
                      Default: 0.25

--force-xlim-01       Force R² x-axis limits to [0, 1] in the scatter plot
                      Default: enabled (use --no-force-xlim-01 to disable)

--print-scatter-stats Print min/max and counts for R² and relative RMSE used
                      in the scatter plot
                      Default: enabled (use --no-print-scatter-stats to disable)

--bad-html-limit N    Max number of bad rows shown in HTML (default 100)
--bad-text-limit N    Max number of bad rows printed in text report (default 200)
--export-bad          Export bad predictions to CSV (default enabled)
--no-export-bad       Do not export bad predictions CSV
```

## Example Usage

Analyze a specific validation statistics file with custom thresholds:

```bash
python scripts/generate_prediction_quality_report.py \
  --input path/to/validation_stats.csv \
  --output-dir path/to/output \
  --r2-good 0.85 \
  --rmse-good 0.15
```

Run from any subdirectory (e.g., from within a results directory):

```bash
python ../../scripts/generate_prediction_quality_report.py
```

## Output Files

The analysis generates the following output files:

1. **detailed_quality_assessment.csv**: Full dataset with quality categorization for each prediction
2. **variable_quality_summary.csv**: Summary statistics for each variable
3. **quality_summary_report.txt**: Text report with overall statistics and best/worst variables
4. **prediction_quality_report.html**: Interactive HTML report
5. **prediction_quality_by_variable.png**: Bar chart showing quality distribution by variable
6. **overall_prediction_quality.png**: Pie chart showing overall quality distribution
7. **r2_vs_rmse.png**: Scatter plot of R² vs Relative RMSE
8. **bad_predictions_detailed.csv**: Full list of predictions classified as "bad" (export can be disabled)

In addition, the text report now includes a "Bad Predictions Summary" and a limited detailed list (controlled by `--bad-text-limit`), and the HTML report includes a table of the first N bad predictions (controlled by `--bad-html-limit`).

## Classification Criteria

Predictions are classified based on the following criteria:

- **Good**: R² ≥ 0.9, Relative RMSE ≤ 0.1, Relative MAE ≤ 0.1
- **OK**: R² ≥ 0.7, Relative RMSE ≤ 0.25, Relative MAE ≤ 0.25
- **Bad**: Below OK thresholds

Relative RMSE and MAE are calculated by dividing the absolute value by the range of the ground truth data:
- Relative RMSE = RMSE / (gt_max - gt_min)
- Relative MAE = MAE / (gt_max - gt_min)

## Interpreting Results

The analysis provides several ways to interpret the prediction quality:

1. **Overall Statistics**: Percentage of predictions in each quality category
2. **Variable-level Analysis**: Quality breakdown for each variable
3. **Visualizations**: Charts showing the distribution of prediction quality

Focus on variables with high percentages of "bad" predictions for model improvement.

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
