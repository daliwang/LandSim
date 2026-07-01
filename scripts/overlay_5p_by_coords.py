#!/usr/bin/env python3
"""
Overlay Phase-2 (or other) 5P soil predictions onto a base inference grid by (lon, lat).

Use after symlinking Phase-1 ``cnp_inference_entire_dataset`` into a Phase-3 run dir:
keeps Phase-1 global coordinates and ground truth, replaces ``soil_2d_predictions``
for the five P variables where source rows share the same rounded (lon, lat) key.

Example (E3SMv3_h0 Phase 3 seed):

  ln -sfn $PHASE1_RUN_DIR/cnp_inference_entire_dataset $RUN3_DIR/cnp_inference_entire_dataset
  python scripts/overlay_5p_by_coords.py \\
    --base-run-dir "$RUN3_DIR" \\
    --source-predictions "$PHASE2_RUN_DIR/cnp_inference_tropical_only/cnp_predictions"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

FIVE_P: List[str] = [
    "labilep_vr",
    "occlp_vr",
    "solutionp_vr",
    "secondp_vr",
    "primp_vr",
]


def _layer_cols(df: pd.DataFrame, var: str) -> List[str]:
    prefix = f"Y_{var}_col1_layer"
    return sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.split("layer")[-1]),
    )


def overlay_5p(
    base_pred_dir: Path,
    source_pred_dir: Path,
    *,
    source_subdir: str = "soil_2d_predictions",
    source_suffix: str = "",
    ndigits: int = 5,
) -> None:
    src_dir = source_pred_dir / source_subdir
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source predictions dir not found: {src_dir}")
    if not base_pred_dir.is_dir():
        raise FileNotFoundError(f"Base predictions dir not found: {base_pred_dir}")

    for var in FIVE_P:
        base_path = base_pred_dir / f"predictions_Y_{var}.csv"
        src_path = src_dir / f"predictions_Y_{var}{source_suffix}.csv"
        if not base_path.is_file():
            raise FileNotFoundError(base_path)
        if not src_path.is_file():
            raise FileNotFoundError(src_path)

        base = pd.read_csv(base_path)
        src = pd.read_csv(src_path)
        layers = _layer_cols(base, var)
        if _layer_cols(src, var) != layers:
            raise ValueError(f"{var}: layer column mismatch between base and source")

        base = base.assign(
            lon_k=np.round(base["Longitude"].to_numpy(dtype=np.float64), ndigits),
            lat_k=np.round(base["Latitude"].to_numpy(dtype=np.float64), ndigits),
        )
        src_keyed = src.assign(
            lon_k=np.round(src["Longitude"].to_numpy(dtype=np.float64), ndigits),
            lat_k=np.round(src["Latitude"].to_numpy(dtype=np.float64), ndigits),
        ).drop_duplicates(subset=["lon_k", "lat_k"], keep="first")

        lookup = src_keyed.set_index(["lon_k", "lat_k"])[layers]
        keys = list(zip(base["lon_k"].to_numpy(), base["lat_k"].to_numpy()))
        matched = 0
        for i, key in enumerate(keys):
            if key in lookup.index:
                base.loc[i, layers] = lookup.loc[key].to_numpy(dtype=np.float64)
                matched += 1

        base.drop(columns=["lon_k", "lat_k"], inplace=True)
        base.to_csv(base_path, index=False)
        print(f"{var}: overlaid {matched} / {len(base)} base rows from {src_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay 5P predictions from source inference onto base run by (lon, lat)."
    )
    parser.add_argument(
        "--base-run-dir",
        required=True,
        help="Run dir containing cnp_inference_entire_dataset/cnp_predictions (Phase-1 seed).",
    )
    parser.add_argument(
        "--source-predictions",
        required=True,
        help="Source cnp_predictions dir (e.g. Phase-2 tropical inference).",
    )
    parser.add_argument(
        "--source-subdir",
        default="soil_2d_predictions",
        help="Subdir under source with prediction CSVs (default: soil_2d_predictions).",
    )
    parser.add_argument(
        "--source-suffix",
        default="",
        help="Filename suffix after variable name (default: none).",
    )
    parser.add_argument(
        "--coord-round-digits",
        type=int,
        default=5,
        help="Decimal places for (lon, lat) merge keys (default: 5).",
    )
    args = parser.parse_args()

    base_pred_dir = (
        Path(args.base_run_dir).resolve()
        / "cnp_inference_entire_dataset"
        / "cnp_predictions"
        / "soil_2d_predictions"
    )
    source_pred_dir = Path(args.source_predictions).resolve()

    print(f"Base soil_2d_predictions: {base_pred_dir}")
    print(f"Source: {source_pred_dir / args.source_subdir}")
    overlay_5p(
        base_pred_dir,
        source_pred_dir,
        source_subdir=args.source_subdir,
        source_suffix=args.source_suffix,
        ndigits=args.coord_round_digits,
    )


if __name__ == "__main__":
    main()
