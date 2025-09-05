import os
import pandas as pd
import numpy as np
import argparse

# Directory containing all predictions_Y_*.csv files
PREDICTIONS_DIR = './cnp_inference_entire_dataset/cnp_predictions/pft_1d_predictions'

neg_violations = []
nan_violations = []
nan_locations = {}
const_violations = []  # list of tuples (var, const_value)
pft_const_violations = {}  # var -> list of tuples (pft_index, const_value)
all_pft_const_violations = {}  # var -> list of tuples (pft_index, const_value)

# Parse CLI args to allow overriding predictions directory
parser = argparse.ArgumentParser(description='Check PFT1D prediction CSVs for sign and NaN issues')
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
    if var == 'xsmrpool':
        pos_count = (arr > 0).sum()
        if pos_count > 0:
            print(f'  [WARNING] {pos_count} positive values found in xsmrpool (should be non-positive)')
    else:
        neg_count = (arr < 0).sum()
        if neg_count > 0:
            print(f'  [WARNING] {neg_count} negative values found in {var} (should be non-negative)')
            neg_violations.append(var)

def parse_pft_index(col_name: str) -> int:
    # Expecting columns like 'Y_var_pft7'
    try:
        return int(col_name.rsplit('pft', 1)[-1])
    except Exception:
        return -1


def main():
    files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith('predictions_Y_') and f.endswith('.csv')]
    files.sort()
    print(f'Found {len(files)} prediction files.')
    for fname in files:
        var = fname.replace('predictions_Y_', '').replace('.csv', '')
        fpath = os.path.join(PREDICTIONS_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            # Always ignore the first two columns (Longitude, Latitude)
            df_numeric = df.iloc[:, 2:]
            nan_count = df_numeric.isna().sum().sum()
            print(f'Variable: {var}')
            print(f'  Shape: {df_numeric.shape}, Total values: {df_numeric.size}, NaNs: {nan_count}')
            print(f'  Min: {df_numeric.min().min() if df_numeric.size else "N/A"}, Max: {df_numeric.max().max() if df_numeric.size else "N/A"}, Mean: {df_numeric.mean().mean() if df_numeric.size else "N/A"}')
            check_sign(var, df_numeric.values.flatten())
            # Print a few sample values
            print(f'  Sample values: {df_numeric.values.flatten()[:8] if df_numeric.size else "N/A"}')
            if nan_count > 0:
                nan_violations.append(var)
                # Find first few NaN locations (row, col)
                nan_pos = np.argwhere(df_numeric.isna().values)
                nan_locations[var] = nan_pos[:5]  # Show up to 5 locations
                print(f'  [NaN] First NaN locations (row, col): {nan_pos[:5]}')

            # Create a NaN-robust view by dropping rows where all PFT columns are NaN
            df_valid = df_numeric[~df_numeric.isna().all(axis=1)]

            # Constant prediction detection across all PFTs (tolerance) using NaN-robust view
            if df_valid.size > 0:
                arr_valid = df_valid.values.flatten()
                # If all values are NaN after drop (unlikely), skip
                if not np.all(np.isnan(arr_valid)):
                    if (np.nanmax(arr_valid) - np.nanmin(arr_valid)) <= EPS:
                        const_val = float(np.nanmean(arr_valid))
                        const_violations.append((var, const_val))
                        print(f'  [CONST] All predictions are constant (±{EPS}): {fmt(const_val)}')

            # Per-PFT constant detection (by column) using NaN-robust view
            const_pfts = []
            for col in df_valid.columns:
                col_vals = df_valid[col].values
                if col_vals.size == 0 or np.all(np.isnan(col_vals)):
                    continue
                if (np.nanmax(col_vals) - np.nanmin(col_vals)) <= EPS:
                    pft_idx = parse_pft_index(col)
                    const_pfts.append((pft_idx, float(np.nanmean(col_vals))))
            if const_pfts:
                # sort by pft index where possible
                const_pfts_sorted = sorted(const_pfts, key=lambda t: (t[0] if t[0] != -1 else 10**9))
                pft_const_violations[var] = const_pfts_sorted
                pretty = ', '.join([f'pft{p}:{fmt(v)}' for p, v in const_pfts_sorted if p != -1])
                fallback = ', '.join([f'{c}:{fmt(v)}' for (p, v), c in zip(const_pfts_sorted, df_valid.columns)])
                print(f'  [CONST_PFT] Constant columns: {pretty if pretty else fallback}')

            # All-PFT constant (every PFT column constant) detection using NaN-robust view
            if df_valid.shape[1] > 0 and df_valid.shape[0] > 0:
                is_const_mask = []
                all_const = []
                for c in df_valid.columns:
                    vals = df_valid[c].values
                    if vals.size == 0 or np.all(np.isnan(vals)):
                        is_const_mask.append(False)
                        continue
                    is_const = (np.nanmax(vals) - np.nanmin(vals)) <= EPS
                    is_const_mask.append(is_const)
                    if is_const:
                        all_const.append((parse_pft_index(c), float(np.nanmean(vals))))
                if all(is_const_mask) and len(is_const_mask) == df_valid.shape[1]:
                    all_const_sorted = sorted(all_const, key=lambda t: (t[0] if t[0] != -1 else 10**9))
                    all_pft_const_violations[var] = all_const_sorted
                    pretty_all = ', '.join([f'pft{p}:{fmt(v)}' for p, v in all_const_sorted if p != -1])
                    print(f'  [CONST_ALL_PFTS] All PFT columns constant: {pretty_all}')
        except Exception as e:
            print(f'[ERROR] Could not process {fname}: {e}')
        print('-' * 60)

    # Summary Table
    print('\nSUMMARY TABLE')
    print('Variables with negative values (should be non-negative):')
    if neg_violations:
        for v in neg_violations:
            print(f'  - {v}')
    else:
        print('  None')
    print('\nVariables with NaNs:')
    if nan_violations:
        for v in nan_violations:
            print(f'  - {v} (first NaN locations: {nan_locations[v]})')
    else:
        print('  None')
    print('\nVariables with constant predictions (all PFTs):')
    if const_violations:
        for v, val in const_violations:
            print(f'  - {v}: constant value {fmt(val)}')
    else:
        print('  None')
    print('\nVariables with constant predictions in specific PFTs:')
    if pft_const_violations:
        for v, items in pft_const_violations.items():
            items_sorted = sorted(items, key=lambda t: (t[0] if t[0] != -1 else 10**9))
            items_str = ', '.join([f'pft{p}={fmt(val)}' if p != -1 else f'col={fmt(val)}' for p, val in items_sorted])
            print(f'  - {v}: {items_str}')
    else:
        print('  None')
    print('\nVariables where all PFT columns are constant:')
    if all_pft_const_violations:
        for v, items in all_pft_const_violations.items():
            items_sorted = sorted(items, key=lambda t: (t[0] if t[0] != -1 else 10**9))
            items_str = ', '.join([f'pft{p}={fmt(val)}' if p != -1 else f'col={fmt(val)}' for p, val in items_sorted])
            print(f'  - {v}: {items_str}')
    else:
        print('  None')
    print('\nVariables (names only) with at least one constant PFT column:')
    if pft_const_violations:
        for v in sorted(pft_const_violations.keys()):
            print(f'  - {v}')
    else:
        print('  None')

if __name__ == '__main__':
    main()
