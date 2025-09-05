import os
import pandas as pd
import numpy as np
import argparse

# Directory containing all soil2d predictions_Y_*.csv files
PREDICTIONS_DIR = './cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions'

neg_violations = []
nan_violations = []
nan_locations = {}
col_const_violations = {}  # var -> list of tuples (col_name, const_value)
all_cols_const_violations = {}  # var -> list of tuples (col_name, const_value)

# CLI args
parser = argparse.ArgumentParser(description='Check Soil2D prediction CSVs for sign, NaNs, and constant columns')
parser.add_argument('--pred-dir', type=str, default=None, help='Directory containing predictions_Y_*.csv files')
parser.add_argument('--eps', type=float, default=1e-9, help='Tolerance for constant detection (max-min <= eps)')
args, unknown = parser.parse_known_args()
if args.pred_dir:
    PREDICTIONS_DIR = args.pred_dir
EPS = float(args.eps)

def fmt(v: float) -> str:
    try:
        return f"{float(v):.8g}"
    except Exception:
        return str(v)

# Helper to check sign constraints
def check_sign(var, arr):
    neg_count = (arr < 0).sum()
    if neg_count > 0:
        print(f'  [WARNING] {neg_count} negative values found in {var} (should be non-negative)')
        neg_violations.append(var)

def main():
    files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith('predictions_Y_') and f.endswith('.csv')]
    files.sort()
    print(f'Found {len(files)} soil2d prediction files.')
    for fname in files:
        var = fname.replace('predictions_Y_', '').replace('.csv', '')
        fpath = os.path.join(PREDICTIONS_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            # Always ignore the first two columns (Longitude, Latitude) if present
            if df.shape[1] > 2:
                df_numeric = df.iloc[:, 2:]
            else:
                df_numeric = df
            # Replace all NaNs with zeros for analysis
            nan_count = df_numeric.isna().sum().sum()
            if nan_count > 0:
                nan_violations.append(var)
                nan_pos = np.argwhere(df_numeric.isna().values)
                nan_locations[var] = nan_pos[:5]
                print(f'  [NaN] First NaN locations (row, col): {nan_pos[:5]}')
            df_numeric = df_numeric.fillna(0)
            arr = df_numeric.values.flatten()
            print(f'Variable: {var}')
            print(f'  Shape: {df_numeric.shape}, Total values: {arr.size}, NaNs (before replace): {nan_count}')
            if arr.size:
                print(f'  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}')
            else:
                print('  Min: N/A, Max: N/A, Mean: N/A')
            check_sign(var, arr)
            print(f'  Sample values: {arr[:8] if arr.size else "N/A"}')

            # Per-column constant detection (tolerance)
            const_cols = []
            for col in df_numeric.columns:
                col_vals = df_numeric[col].values
                if col_vals.size == 0:
                    continue
                if (np.nanmax(col_vals) - np.nanmin(col_vals)) <= EPS:
                    const_cols.append((col, float(np.nanmean(col_vals))))
            if const_cols:
                # sort by column name for readability
                const_cols_sorted = sorted(const_cols, key=lambda t: t[0])
                col_const_violations[var] = const_cols_sorted
                pretty = ', '.join([f'{c}:{fmt(v)}' for c, v in const_cols_sorted])
                print(f'  [CONST_COLS] Constant columns: {pretty}')

            # All columns constant
            if df_numeric.shape[1] > 0:
                is_const_mask = [(np.nanmax(df_numeric[c].values) - np.nanmin(df_numeric[c].values)) <= EPS for c in df_numeric.columns]
                if all(is_const_mask):
                    all_cols = [(c, float(np.nanmean(df_numeric[c].values))) for c in df_numeric.columns]
                    all_cols_sorted = sorted(all_cols, key=lambda t: t[0])
                    all_cols_const_violations[var] = all_cols_sorted
                    pretty_all = ', '.join([f'{c}:{fmt(v)}' for c, v in all_cols_sorted])
                    print(f'  [CONST_ALL_COLS] All columns constant: {pretty_all}')
        except Exception as e:
            print(f'[ERROR] Could not process {fname}: {e}')
        print('-' * 60)

    # Summary Table
    print('\nSUMMARY TABLE')
    print('Soil2D variables with negative values (should be non-negative):')
    if neg_violations:
        for v in neg_violations:
            print(f'  - {v}')
    else:
        print('  None')
    print('\nSoil2D variables with NaNs (before replace):')
    if nan_violations:
        for v in nan_violations:
            print(f'  - {v} (first NaN locations: {nan_locations[v]})')
    else:
        print('  None')
    print('\nSoil2D variables with constant predictions in specific columns:')
    if col_const_violations:
        for v, items in col_const_violations.items():
            items_str = ', '.join([f'{c}={fmt(val)}' for c, val in items])
            print(f'  - {v}: {items_str}')
    else:
        print('  None')
    print('\nSoil2D variables where all columns are constant:')
    if all_cols_const_violations:
        for v, items in all_cols_const_violations.items():
            items_str = ', '.join([f'{c}={fmt(val)}' for c, val in items])
            print(f'  - {v}: {items_str}')
    else:
        print('  None')
    print('\nSoil2D variables (names only) with at least one constant column:')
    if col_const_violations:
        for v in sorted(col_const_violations.keys()):
            print(f'  - {v}')
    else:
        print('  None')

if __name__ == '__main__':
    main()
