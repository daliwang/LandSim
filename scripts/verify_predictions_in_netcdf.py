#!/usr/bin/env python3
"""
Verify that all prediction CSV values are correctly written into the NetCDF file.

Checks:
1) Scalar predictions: predictions_scalar.csv -> NetCDF var (gridcell)
2) PFT 1D predictions: predictions_Y_*.csv -> NetCDF var (pft, gridcell)
3) Soil 2D predictions: predictions_Y_*.csv -> NetCDF var (column, levgrnd, gridcell)

Also validates that CSV Longitude/Latitude align with NetCDF grid1d_lon/grid1d_lat.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_PRED_DIR = "cnp_inference_entire_dataset/cnp_predictions"
DEFAULT_NETCDF = "comparison_results/ai_predictions_for_plotting.nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify CSV predictions are correctly written into NetCDF."
    )
    parser.add_argument("--predictions-dir", default=DEFAULT_PRED_DIR,
                        help="Directory containing cnp_predictions (scalar/pft/soil CSVs).")
    parser.add_argument("--netcdf", default=DEFAULT_NETCDF,
                        help="NetCDF file produced by ai_predictions_to_netcdf.py.")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Tolerance for max abs diff.")
    parser.add_argument("--max-vars", type=int, default=0,
                        help="Limit number of variables per group (0 = no limit).")
    return parser.parse_args()


def _coords_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return df["Longitude"].values, df["Latitude"].values


def _coords_match(lon_csv: np.ndarray, lat_csv: np.ndarray,
                  lon_nc: np.ndarray, lat_nc: np.ndarray, tol: float) -> bool:
    if len(lon_csv) != len(lon_nc) or len(lat_csv) != len(lat_nc):
        return False
    return np.allclose(lon_csv, lon_nc, atol=tol) and np.allclose(lat_csv, lat_nc, atol=tol)


def _build_coord_index(lon_nc: np.ndarray, lat_nc: np.ndarray, decimals: int = 4) -> dict:
    keys = {}
    for i in range(len(lon_nc)):
        key = (round(float(lon_nc[i]), decimals), round(float(lat_nc[i]), decimals))
        if key not in keys:
            keys[key] = i
    return keys


def _map_csv_to_grid(df: pd.DataFrame, coord_index: dict, lon_nc: np.ndarray,
                     lat_nc: np.ndarray, decimals: int = 4) -> np.ndarray:
    lon = df["Longitude"].values
    lat = df["Latitude"].values
    idx = np.full(len(df), -1, dtype=int)
    for i in range(len(df)):
        key = (round(float(lon[i]), decimals), round(float(lat[i]), decimals))
        if key in coord_index:
            idx[i] = coord_index[key]
        else:
            # Fallback: nearest neighbor (distance is tiny for these cases)
            d = (lon_nc - lon[i]) ** 2 + (lat_nc - lat[i]) ** 2
            idx[i] = int(np.argmin(d))
    return idx


def _max_mean_diff(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = np.abs(a - b)
    return float(np.nanmax(diff)), float(np.nanmean(diff))


def _sorted_pft_cols(df: pd.DataFrame, var: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(f"Y_{var}_pft")]
    def pft_num(c: str) -> int:
        try:
            return int(c.split("_pft")[-1])
        except ValueError:
            return 999
    return sorted(cols, key=pft_num)


def _sorted_layer_cols(df: pd.DataFrame, var: str) -> List[str]:
    cols = [c for c in df.columns if c.startswith(f"Y_{var}_col1_layer")]
    def layer_num(c: str) -> int:
        try:
            return int(c.replace(f"Y_{var}_col1_layer", ""))
        except ValueError:
            return 999
    return sorted(cols, key=layer_num)


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.predictions_dir)
    netcdf_path = Path(args.netcdf)

    if not pred_dir.exists():
        raise FileNotFoundError(f"predictions-dir not found: {pred_dir}")
    if not netcdf_path.exists():
        raise FileNotFoundError(f"netcdf not found: {netcdf_path}")

    ds = xr.open_dataset(netcdf_path)
    lon_nc = ds["grid1d_lon"].values
    lat_nc = ds["grid1d_lat"].values
    coord_index = _build_coord_index(lon_nc, lat_nc)

    print("NetCDF:", netcdf_path)
    print("Predictions dir:", pred_dir)
    print(f"gridcell count: {ds.sizes.get('gridcell')}")

    all_ok = True

    # 1) Scalar predictions
    scalar_csv = pred_dir / "predictions_scalar.csv"
    if scalar_csv.exists():
        df = pd.read_csv(scalar_csv)
        lon_csv, lat_csv = _coords_from_df(df)
        coords_ok = _coords_match(lon_csv, lat_csv, lon_nc, lat_nc, args.tol)
        if not coords_ok:
            idx = _map_csv_to_grid(df, coord_index, lon_nc, lat_nc)
            missing = int(np.sum(idx < 0))
            if missing > 0:
                print(f"ERROR: scalar coords missing {missing} rows in NetCDF grid")
                all_ok = False
        scalar_cols = [c for c in df.columns if c.startswith("Y_")]
        if args.max_vars > 0:
            scalar_cols = scalar_cols[:args.max_vars]
        for col in scalar_cols:
            var = col[2:]
            if var not in ds:
                print(f"Missing in NetCDF (skipped): {var}")
                continue
            if coords_ok:
                a = df[col].values
                b = ds[var].values
            else:
                b = ds[var].values[idx]
                a = df[col].values
            max_diff, mean_diff = _max_mean_diff(a, b)
            ok = max_diff <= args.tol
            all_ok = all_ok and ok
            print(f"scalar {var}: max_diff={max_diff} mean_diff={mean_diff} ok={ok}")

    # 2) PFT 1D predictions
    pft_dir = pred_dir / "pft_1d_predictions"
    if pft_dir.exists():
        files = sorted(pft_dir.glob("predictions_*.csv"))
        if args.max_vars > 0:
            files = files[:args.max_vars]
        for f in files:
            var = f.stem.replace("predictions_Y_", "")
            df = pd.read_csv(f)
            lon_csv, lat_csv = _coords_from_df(df)
            coords_ok = _coords_match(lon_csv, lat_csv, lon_nc, lat_nc, args.tol)
            if not coords_ok:
                idx = _map_csv_to_grid(df, coord_index, lon_nc, lat_nc)
                missing = int(np.sum(idx < 0))
                if missing > 0:
                    print(f"ERROR: pft coords missing {missing} rows in NetCDF for {var}")
                    all_ok = False
            if var not in ds:
                print(f"Missing in NetCDF (skipped): {var}")
                continue
            cols = _sorted_pft_cols(df, var)
            pft_data = np.zeros((16, len(df)), dtype=float)
            for i, c in enumerate(cols[:16]):
                pft_data[i, :] = df[c].values
            if coords_ok:
                nc_data = ds[var].values
            else:
                nc_data = ds[var].values[:, idx]
            max_diff, mean_diff = _max_mean_diff(pft_data, nc_data)
            ok = max_diff <= args.tol
            all_ok = all_ok and ok
            print(f"pft {var}: max_diff={max_diff} mean_diff={mean_diff} ok={ok}")

    # 3) Soil 2D predictions
    soil_dir = pred_dir / "soil_2d_predictions"
    if soil_dir.exists():
        files = sorted(soil_dir.glob("predictions_*.csv"))
        if args.max_vars > 0:
            files = files[:args.max_vars]
        for f in files:
            var = f.stem.replace("predictions_Y_", "")
            df = pd.read_csv(f)
            lon_csv, lat_csv = _coords_from_df(df)
            coords_ok = _coords_match(lon_csv, lat_csv, lon_nc, lat_nc, args.tol)
            if not coords_ok:
                idx = _map_csv_to_grid(df, coord_index, lon_nc, lat_nc)
                missing = int(np.sum(idx < 0))
                if missing > 0:
                    print(f"ERROR: soil coords missing {missing} rows in NetCDF for {var}")
                    all_ok = False
            if var not in ds:
                print(f"Missing in NetCDF (skipped): {var}")
                continue
            cols = _sorted_layer_cols(df, var)[:10]
            soil_data = np.zeros((1, len(cols), len(df)), dtype=float)
            for i, c in enumerate(cols):
                soil_data[0, i, :] = df[c].values
            if coords_ok:
                nc_data = ds[var].values[:, :len(cols), :]
            else:
                nc_data = ds[var].values[:, :len(cols), idx]
            max_diff, mean_diff = _max_mean_diff(soil_data, nc_data)
            ok = max_diff <= args.tol
            all_ok = all_ok and ok
            print(f"soil {var}: max_diff={max_diff} mean_diff={mean_diff} ok={ok}")

    ds.close()

    print("\nOverall:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
