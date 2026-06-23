#!/usr/bin/env python3
"""Export saved CNP inference predictions.pkl to restart-converter CSVs.

This is an export-only helper for long full-domain inference runs where the
model forward pass has already completed and written predictions.pkl.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from scripts.derive_np_from_c import derive_np_from_c_predictions
except Exception:
    derive_np_from_c_predictions = None


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _load_scaler(scalers_dir: Path, key: str):
    path = scalers_dir / f"individual_{key}_scaler.pkl"
    if not path.exists():
        path = scalers_dir / f"{key}_scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing scaler for {key}: {path}")
    return _load_pickle(path)


def _coords(static_inverse_path: Path, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    static = pd.read_csv(static_inverse_path, usecols=["Longitude", "Latitude"])
    if len(static) != n_samples:
        raise ValueError(
            f"Coordinate row count mismatch: {len(static)} coords vs {n_samples} predictions"
        )
    return static["Longitude"].to_numpy(), static["Latitude"].to_numpy()


def _with_coords(df: pd.DataFrame, lon: np.ndarray, lat: np.ndarray) -> pd.DataFrame:
    df.insert(0, "Longitude", lon)
    df.insert(1, "Latitude", lat)
    return df


def export_scalar(predictions, data_info, scalers_dir, out_dir, lon, lat):
    if "scalar" not in predictions:
        return
    pred = _to_numpy(predictions["scalar"])
    cols = data_info.get("y_list_scalar_columns", [])
    if not cols or len(cols) < pred.shape[1]:
        cols = [f"Y_scalar_{i}" for i in range(pred.shape[1])]
    scaler = _load_scaler(scalers_dir, "y_scalar")
    if hasattr(scaler, "inverse_transform_scalar"):
        pred = scaler.inverse_transform_scalar(pred, cols)
    else:
        pred = scaler.inverse_transform(pred)
    df = _with_coords(pd.DataFrame(pred, columns=cols), lon, lat)
    df.to_csv(out_dir / "predictions_scalar.csv", index=False)


def export_pft(predictions, data_info, scalers_dir, out_dir, lon, lat):
    if "pft_1d" not in predictions:
        return
    pred = _to_numpy(predictions["pft_1d"])
    n_samples = pred.shape[0]
    n_pfts = 16
    var_names = data_info.get("y_list_columns_1d", [])
    if pred.ndim == 2:
        n_vars = len(var_names) if var_names else pred.shape[1] // n_pfts
        pred = pred.reshape(n_samples, n_vars, n_pfts)
    elif pred.ndim != 3:
        raise ValueError(f"Unsupported pft_1d prediction shape: {pred.shape}")
    n_vars = pred.shape[1]
    if not var_names or len(var_names) < n_vars:
        var_names = [f"Y_pft_1d_var_{i}" for i in range(n_vars)]
    var_names = var_names[:n_vars]

    scaler = _load_scaler(scalers_dir, "y_pft_1d")
    pft_names = [f"PFT{i}" for i in range(1, n_pfts + 1)]
    pft_dir = out_dir / "pft_1d_predictions"
    pft_dir.mkdir(parents=True, exist_ok=True)

    for idx, var_name in enumerate(var_names):
        per_var = pred[:, idx, :].reshape(n_samples, n_pfts, 1)
        if hasattr(scaler, "inverse_transform_pft_1d"):
            denorm = scaler.inverse_transform_pft_1d(per_var, pft_names, [var_name])[:, :, 0]
        else:
            denorm = per_var[:, :, 0]
        cols = [f"{var_name}_pft{p}" for p in range(1, n_pfts + 1)]
        df = _with_coords(pd.DataFrame(denorm, columns=cols), lon, lat)
        df.to_csv(pft_dir / f"predictions_{var_name}.csv", index=False)


def export_soil(predictions, data_info, scalers_dir, out_dir, lon, lat):
    if "soil_2d" not in predictions:
        return
    pred = _to_numpy(predictions["soil_2d"])
    n_samples = pred.shape[0]
    n_columns = 1
    n_layers = 10
    var_names = data_info.get("y_list_columns_2d", [])
    if pred.ndim == 2:
        per_var = n_columns * n_layers
        if pred.shape[1] % per_var != 0:
            raise ValueError(
                f"Unexpected soil_2d width {pred.shape[1]} not divisible by {per_var}"
            )
        n_vars = pred.shape[1] // per_var
        pred = pred.reshape(n_samples, n_vars, n_columns, n_layers)
    elif pred.ndim != 4:
        raise ValueError(f"Unsupported soil_2d prediction shape: {pred.shape}")
    n_vars = pred.shape[1]
    if not var_names or len(var_names) < n_vars:
        var_names = [f"Y_soil_2d_var_{i}" for i in range(n_vars)]
    var_names = var_names[:n_vars]

    scaler = _load_scaler(scalers_dir, "y_soil_2d")
    soil_dir = out_dir / "soil_2d_predictions"
    soil_dir.mkdir(parents=True, exist_ok=True)

    for idx, var_name in enumerate(var_names):
        per_var = pred[:, idx : idx + 1, :, :]
        if hasattr(scaler, "inverse_transform_soil_2d"):
            denorm = scaler.inverse_transform_soil_2d(per_var, [var_name], n_layers)[:, 0, :, :]
        else:
            denorm = per_var[:, 0, :, :]
        flat = denorm.reshape(n_samples, n_columns * n_layers)
        cols = [
            f"{var_name}_col{col}_layer{layer}"
            for col in range(1, n_columns + 1)
            for layer in range(1, n_layers + 1)
        ]
        df = _with_coords(pd.DataFrame(flat, columns=cols), lon, lat)
        df.to_csv(soil_dir / f"predictions_{var_name}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True, help="Directory containing predictions.pkl")
    parser.add_argument("--config", required=True, help="Training cnp_config.json")
    parser.add_argument("--scalers-dir", required=True, help="Training scalers directory")
    parser.add_argument("--derive-np-from-c", action="store_true", default=False)
    args = parser.parse_args()

    inference_dir = Path(args.inference_dir)
    predictions_dir = inference_dir / "cnp_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = json.load(f)
    data_info = config["data_info"]

    predictions = _load_pickle(inference_dir / "predictions.pkl")
    n_samples = next(_to_numpy(v).shape[0] for v in predictions.values())
    lon, lat = _coords(predictions_dir / "test_static_inverse.csv", n_samples)

    scalers_dir = Path(args.scalers_dir)
    export_scalar(predictions, data_info, scalers_dir, predictions_dir, lon, lat)
    export_pft(predictions, data_info, scalers_dir, predictions_dir, lon, lat)
    export_soil(predictions, data_info, scalers_dir, predictions_dir, lon, lat)

    if args.derive_np_from_c:
        if derive_np_from_c_predictions is None:
            raise RuntimeError("derive_np_from_c_predictions is not available")
        derive_np_from_c_predictions(
            predictions_dir=predictions_dir,
            config_path=Path(args.config),
            output_dir=predictions_dir,
        )

    print(f"Exported CSV predictions for {n_samples} samples to {predictions_dir}")


if __name__ == "__main__":
    main()
