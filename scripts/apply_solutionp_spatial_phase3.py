#!/usr/bin/env python3
"""Apply fitted spatial Phase 3 solutionp calibration (no GT required)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from calibrate_solutionp_spatial_phase3 import (  # noqa: E402
    FIVE_P,
    VAR,
    apply_k,
    apply_spatial_models,
    build_features,
    layer_cols,
    load_models_from_dict,
    load_var_matrix,
)
from region_box_utils import region_mask_arrays
from calibrate_solutionp_spatial_phase3 import REGION_BOXES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply spatial solutionp Phase 3 params to predictions.")
    parser.add_argument("--inference-dir", required=True, help="Dir with cnp_predictions/")
    parser.add_argument("--params-json", required=True, help="Fitted params from calibrate_solutionp_spatial_phase3.py")
    parser.add_argument("--output-subdir", default=None, help="Override output subdir name")
    args = parser.parse_args()

    inf_dir = Path(args.inference_dir).resolve()
    pred_root = inf_dir / "cnp_predictions"
    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    models = load_models_from_dict(params["regions"])

    _, pred_all, cols = load_var_matrix(pred_root, VAR, "pred")
    pred_df, _, _ = load_var_matrix(pred_root, VAR, "pred")
    features_all, feature_names = build_features(pred_root, pred_all)

    lat = pred_df["Latitude"].astype(float).to_numpy()
    lon = pred_df["Longitude"].astype(float).to_numpy()
    pred_corr = pred_all.copy()

    for rname, model in models.items():
        box = REGION_BOXES[rname]
        mask = region_mask_arrays(lat, lon, box)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        pred_corr = apply_spatial_models(pred_corr, features_all, feature_names, idx, model)

    subdir = args.output_subdir or Path(params.get("output_predictions", "out.csv")).parent.name
    if args.output_subdir is None and "output_predictions" in params:
        subdir = Path(params["output_predictions"]).parent.name

    out_dir = pred_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pred_df[["Longitude", "Latitude"]].copy()
    for i, c in enumerate(cols):
        out_df[c] = pred_corr[:, i]
    out_path = out_dir / f"predictions_Y_{VAR}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
