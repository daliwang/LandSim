import os
import pandas as pd
import numpy as np
import argparse

PREDICTIONS_DIR = './cnp_inference_entire_dataset/cnp_predictions/pft_1d_predictions'

parser = argparse.ArgumentParser(description='Replace NaNs in PFT1D predictions with zeros')
parser.add_argument('--pred-dir', type=str, default=None, help='Directory containing predictions_Y_*.csv files')
args, _ = parser.parse_known_args()
if args.pred_dir:
    PREDICTIONS_DIR = args.pred_dir

def main():
    files = [f for f in os.listdir(PREDICTIONS_DIR) if f.startswith('predictions_Y_') and f.endswith('.csv')]
    files.sort()
    print(f'Found {len(files)} prediction files.')
    for fname in files:
        fpath = os.path.join(PREDICTIONS_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            total_nans_before = int(df.isna().sum().sum())
            if total_nans_before == 0:
                print(f'{fname}: No NaNs found.')
                continue
            df_filled = df.fillna(0)
            df_filled.to_csv(fpath, index=False)
            print(f'{fname}: {total_nans_before} NaNs replaced with zeros.')
        except Exception as e:
            print(f'[ERROR] Could not process {fname}: {e}')

if __name__ == '__main__':
    main()
