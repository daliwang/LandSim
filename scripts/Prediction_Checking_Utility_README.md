## Scripts Overview

This README documents three utility scripts used to validate and clean model outputs:

- `scripts/check_pft1d_predictions.py`
- `scripts/check_soil2d_predictions.py`
- `scripts/fix_pft1d_nans.py`

All scripts support CLI flags so you can point them at different result folders without editing code.

---

### check_pft1d_predictions.py

**Purpose**: Quality checks for PFT1D prediction CSVs (`predictions_Y_*.csv`). Performs:

- **Negative/invalid values check**
- **NaN detection**
- **Constant prediction detection** with tolerance:
  - Variables where **all PFT columns** are constant
  - Variables where **at least one PFT column** is constant (reports `pftN=value`)
- Ignores rows where **all PFT columns are NaN** when checking constancy
- Assumes the first two columns are `Longitude` and `Latitude` and excludes them from PFT analysis

**Key assumptions**:

- Files are shaped like: `Longitude,Latitude,Y_<var>_pft1,...,Y_<var>_pft14`
- PFT columns begin at column index 2

**CLI**:

- `--pred-dir`: Directory containing `predictions_Y_*.csv`
- `--eps`: Numerical tolerance for “effectively constant” checks (default suggested: `1e-9`)

**Example**:

```bash
python /mnt/proj-shared/AI4BGC_7xw/AI4BGC/scripts/check_pft1d_predictions.py \
  --pred-dir /mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20250904_180409/cnp_predictions/pft_1d_predictions \
  --eps 1e-9
```

You can filter for the “all PFTs constant” section:

```bash
python /mnt/proj-shared/AI4BGC_7xw/AI4BGC/scripts/check_pft1d_predictions.py \
  --pred-dir /mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20250904_180409/cnp_predictions/pft_1d_predictions \
  --eps 1e-9 | sed -n '/Variables where all PFT columns are constant:/,$p' | cat
```

---

### check_soil2d_predictions.py

**Purpose**: Quality checks for Soil-2D prediction CSVs (e.g., `predictions_Y_soil*_*.csv`). Similar checks as the PFT1D script:

- Negative/invalid values
- NaNs
- Constant per-column detection with tolerance
- Optional summary of variables where all columns are constant

**CLI**:

- `--pred-dir`: Directory containing Soil-2D prediction CSVs
- `--eps`: Numerical tolerance for constant checks

**Example**:

```bash
python /mnt/proj-shared/AI4BGC_7xw/AI4BGC/scripts/check_soil2d_predictions.py \
  --pred-dir /mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20250904_180409/cnp_predictions/soil_2d_predictions \
  --eps 1e-9
```

---

### fix_pft1d_nans.py

**Purpose**: Clean PFT1D prediction CSVs by replacing `NaN` values with zeros.

Note: This currently replaces NaNs with zeros across all columns. Use before running the checkers to prevent NaNs from interfering with constant detection.

**CLI**:

- `--pred-dir`: Directory containing `predictions_Y_*.csv`

**Example**:

```bash
python /mnt/proj-shared/AI4BGC_7xw/AI4BGC/scripts/fix_pft1d_nans.py \
  --pred-dir /mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20250904_180409/cnp_inference_entire_dataset/cnp_predictions/pft_1d_predictions
```

---

### Tips

- Prefer using CLI flags to control behavior and paths; avoid editing script internals.
- Use `--eps` when you want to treat nearly constant floating values as constant.
- Large CSVs: piping through tools like `sed`, `grep`, or `head` can help focus on the sections you need.


