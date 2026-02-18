# Worst Variables Filtering Options

## Overview

The `generate_prediction_quality_report.py` script now provides flexible options to control which variables appear in the "Variables with Worst Predictions" section and are plotted in `top_bad_plots`.

## Default Behavior

**By default**, only variables with **bad predictions** (`bad_pct > 0`) are included in the worst list. This means:
- Variables that are 100% OK (0% bad) are **excluded**, even if they have 0% good
- Variables are sorted by `bad_pct` (highest first), then by `good_pct` (lowest first)

## Command-Line Options

### `--worst-filter-bad-only` (Default: Enabled)

Filter worst variables to only include those with bad predictions (`bad_pct > 0`).

```bash
# Default behavior (only variables with bad predictions)
python scripts/generate_prediction_quality_report.py --input validation_stats.csv

# Explicitly enable (same as default)
python scripts/generate_prediction_quality_report.py --input validation_stats.csv --worst-filter-bad-only

# Disable to include all variables sorted by good_pct
python scripts/generate_prediction_quality_report.py --input validation_stats.csv --no-worst-filter-bad-only
```

**Example**: `solutionp_vr` with 0% good, 100% ok, 0% bad will be excluded by default.

### `--worst-min-good-pct THRESHOLD`

Include variables in worst list with `good_pct` below the specified threshold.

```bash
# Include variables with <50% good predictions
python scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --worst-min-good-pct 50.0

# Include variables with <30% good predictions
python scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --worst-min-good-pct 30.0
```

**Note**: This filter is applied **in addition** to the bad-only filter (if enabled).

### `--worst-vars-list VARIABLES`

Include specific variables in the worst list, regardless of their quality metrics.

```bash
# Include specific variables
python scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --worst-vars-list "cpool,npool,ppool"

# Combine with other filters
python scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --worst-vars-list "solutionp_vr,primp_vr"
```

**Note**: Specified variables are added even if they don't meet other filter criteria.

## Filter Combination Logic

Filters are applied in this order:

1. **Bad-only filter** (if `--worst-filter-bad-only`): `bad_pct > 0`
2. **Good percentage threshold** (if `--worst-min-good-pct`): `good_pct < threshold`
3. **Specific variable list** (if `--worst-vars-list`): Add specified variables

The final list is sorted by:
- `bad_pct` (highest first)
- `good_pct` (lowest first) for tie-breaking

## Examples

### Example 1: Default (Only Bad Predictions)

```bash
python scripts/generate_prediction_quality_report.py --input validation_stats.csv
```

**Result**: Only variables with `bad_pct > 0` appear in worst list. Variables like `solutionp_vr` (0% good, 100% ok, 0% bad) are excluded.

### Example 2: Include Low Good-Percentage Variables

```bash
python scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --no-worst-filter-bad-only \
  --worst-min-good-pct 30.0
```

**Result**: Includes variables with `good_pct < 30.0`, even if they have 0% bad (e.g., 100% OK variables with low good percentage).

### Example 3: Include Specific Variables

```bash
python scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --worst-vars-list "solutionp_vr,primp_vr,occlp_vr"
```

**Result**: Includes the specified variables in the worst list, regardless of their metrics. Useful for investigating specific variables even if they're not in the worst-performing group.

### Example 4: Combine Filters

```bash
python scripts/generate_prediction_quality_report.py \
  --input validation_stats.csv \
  --worst-min-good-pct 50.0 \
  --worst-vars-list "cpool"
```

**Result**: 
- Includes variables with `bad_pct > 0` AND `good_pct < 50.0`
- Also includes `cpool` even if it doesn't meet the above criteria

## Impact on Plotting

The `top_bad_plots` directory will only contain plots for variables listed in the "Variables with Worst Predictions" section. By default, this means:
- Only variables with bad predictions are plotted
- Variables that are 100% OK are excluded from plots

Use `--worst-vars-list` to include specific variables in plots even if they're not in the worst-performing group.

## Migration Notes

**Previous behavior**: Variables were sorted by `good_pct` only, which included variables with 0% good even if they were 100% OK.

**New default behavior**: Only variables with `bad_pct > 0` are included, sorted by `bad_pct` first.

**To restore previous behavior**: Use `--no-worst-filter-bad-only` flag.
