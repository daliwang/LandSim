#!/usr/bin/env python3
"""
Compare ground truth of the five soil P variables between the Amazon and Africa
boxes used in the Phase3 / 5P bias workflow.

Region bounds are read from the same JSON configs as apply_5p_bias_scale_correction
(default: config/training_config_amazon_5p_box.json and
config/training_config_africa_5p_box.json), i.e. the same definitions as
scripts/merge_5p_bias_corrected_amazon_africa.py:

  Amazon: lat [-30, 10], lon [270, 330]   (degrees, lon 0–360)
  Africa: lat [-15, 15], lon [0, 30]

Ground truth CSVs are read from:

  <run-dir>/<inference-subdir>/cnp_predictions/soil_2d_ground_truth/ground_truth_Y_<var>.csv

Each file must have Longitude, Latitude and layer columns Y_<var>_col1_layer{1..10}.

Usage (from repo root):

  python scripts/compare_5p_gt_two_regions_inference.py \\
    --run-dir cnp_results/run_20260315_202304_phase3_tworegions

  python scripts/compare_5p_gt_two_regions_inference.py \\
    --run-dir cnp_results/run_20260315_113900_phase1_global \\
    --output-csv cnp_results/run_20260315_113900_phase1_global/analysis/5p_gt_amazon_vs_africa.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

FIVE_P_VARS: List[str] = [
    "labilep_vr",
    "occlp_vr",
    "solutionp_vr",
    "secondp_vr",
    "primp_vr",
]


@dataclass
class RegionBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    name: str

    def mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        lon360 = lon.copy()
        lon360[~np.isfinite(lon360)] = np.nan
        lon360[lon360 < 0] += 360.0
        return (
            np.isfinite(lat)
            & np.isfinite(lon360)
            & (lat >= self.lat_min)
            & (lat <= self.lat_max)
            & (lon360 >= self.lon_min)
            & (lon360 <= self.lon_max)
        )


def _first_region_box_from_config(config_path: Path) -> Tuple[float, float, float, float]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    boxes = cfg.get("data_filtering_config", {}).get("region_boxes")
    if not boxes or len(boxes[0]) != 4:
        raise ValueError(f"No valid data_filtering_config.region_boxes in {config_path}")
    lat_min, lat_max, lon_min, lon_max = (float(x) for x in boxes[0])
    return lat_min, lat_max, lon_min, lon_max


def _layer_columns(df: pd.DataFrame, var: str) -> List[str]:
    prefix = f"Y_{var}_col1_layer"
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort(key=lambda c: int(c.replace(prefix, "")))
    return cols


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "p05": np.nan, "p50": np.nan, "p95": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _optional_natveg_mask(gt: pd.DataFrame, static_path: Path) -> Optional[np.ndarray]:
    if not static_path.is_file():
        return None
    st = pd.read_csv(static_path)
    if "Latitude" not in st.columns or "Longitude" not in st.columns:
        return None
    pct_nv = None
    for c in ("PCT_NATVEG", "pct_natveg"):
        if c in st.columns:
            pct_nv = pd.to_numeric(st[c], errors="coerce").to_numpy()
            break
    pct0 = None
    for c in ("PCT_NAT_PFT_0",):
        if c in st.columns:
            pct0 = pd.to_numeric(st[c], errors="coerce").to_numpy()
            break
    if pct_nv is None:
        return None
    keep = pct_nv > 0
    if pct0 is not None:
        keep &= pct0 < 100.0
    # Align rows: merge on rounded lon/lat (grid snap)
    gt_k = gt.assign(
        _lon=np.round(gt["Longitude"].to_numpy(dtype=np.float64), 5),
        _lat=np.round(gt["Latitude"].to_numpy(dtype=np.float64), 5),
    )
    st_k = st.assign(
        _lon=np.round(st["Longitude"].to_numpy(dtype=np.float64), 5),
        _lat=np.round(st["Latitude"].to_numpy(dtype=np.float64), 5),
    )
    flags = pd.DataFrame({"_lon": gt_k["_lon"], "_lat": gt_k["_lat"]})
    meta = st_k[["_lon", "_lat"]].copy()
    meta["_natveg_ok"] = keep.astype(bool)
    meta = meta.drop_duplicates(subset=["_lon", "_lat"], keep="first")
    merged = flags.merge(meta, on=["_lon", "_lat"], how="left")
    if merged["_natveg_ok"].isna().all():
        return None
    filled = merged["_natveg_ok"].fillna(True).to_numpy(dtype=bool)
    return filled


def main() -> None:
    p = argparse.ArgumentParser(description="Compare 5P ground truth: Amazon vs Africa (inference CSVs).")
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing cnp_inference_entire_dataset (or set --inference-dir).",
    )
    p.add_argument(
        "--inference-subdir",
        type=str,
        default="cnp_inference_entire_dataset",
        help="Subdirectory under run-dir with cnp_predictions/soil_2d_ground_truth (default: cnp_inference_entire_dataset).",
    )
    p.add_argument(
        "--inference-dir",
        type=str,
        default=None,
        help="If set, use this path directly as the inference folder (overrides run-dir + inference-subdir).",
    )
    p.add_argument(
        "--amazon-region-config",
        type=str,
        default=str(REPO_ROOT / "config/training_config_amazon_5p_box.json"),
        help="JSON with data_filtering_config.region_boxes[0] = Amazon box.",
    )
    p.add_argument(
        "--africa-region-config",
        type=str,
        default=str(REPO_ROOT / "config/training_config_africa_5p_box.json"),
        help="JSON with data_filtering_config.region_boxes[0] = Africa box.",
    )
    p.add_argument(
        "--regions-json",
        type=str,
        default=None,
        help="Optional JSON: {\"amazon\": [lat_min,lat_max,lon_min,lon_max], \"africa\": [...]} overrides config paths.",
    )
    p.add_argument(
        "--natveg-filter",
        action="store_true",
        help="Restrict to natveg cells (PCT_NATVEG>0, PCT_NAT_PFT_0<100) via merge with test_static_inverse.csv.",
    )
    p.add_argument("--output", type=str, default=None, help="Write text report to this path.")
    p.add_argument("--output-csv", type=str, default=None, help="Write long-form stats CSV.")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if args.inference_dir:
        inf_dir = Path(args.inference_dir).resolve()
    else:
        inf_dir = (run_dir / args.inference_subdir).resolve()
    gt_dir = inf_dir / "cnp_predictions" / "soil_2d_ground_truth"
    static_path = inf_dir / "cnp_predictions" / "test_static_inverse.csv"

    if args.regions_json:
        with open(args.regions_json, "r", encoding="utf-8") as f:
            rj = json.load(f)
        am = rj["amazon"]
        af = rj["africa"]
        amazon = RegionBox(float(am[0]), float(am[1]), float(am[2]), float(am[3]), "amazon")
        africa = RegionBox(float(af[0]), float(af[1]), float(af[2]), float(af[3]), "africa")
    else:
        am = _first_region_box_from_config(Path(args.amazon_region_config))
        af = _first_region_box_from_config(Path(args.africa_region_config))
        amazon = RegionBox(am[0], am[1], am[2], am[3], "amazon")
        africa = RegionBox(af[0], af[1], af[2], af[3], "africa")

    lines: List[str] = []
    lines.append("5P ground truth: Amazon vs Africa (from inference soil_2d_ground_truth CSVs)")
    lines.append(f"Inference dir: {inf_dir}")
    lines.append(f"Amazon box [lat_min, lat_max, lon_min, lon_max]: [{amazon.lat_min}, {amazon.lat_max}, {amazon.lon_min}, {amazon.lon_max}]")
    lines.append(f"Africa box: [{africa.lat_min}, {africa.lat_max}, {africa.lon_min}, {africa.lon_max}]")
    lines.append(f"Natveg filter: {args.natveg_filter}")
    lines.append("")

    rows_csv: List[dict] = []

    for var in FIVE_P_VARS:
        path = gt_dir / f"ground_truth_Y_{var}.csv"
        if not path.is_file():
            lines.append(f"MISSING {path.name}")
            continue
        df = pd.read_csv(path)
        if "Longitude" not in df.columns or "Latitude" not in df.columns:
            print(f"ERROR: {path} missing Longitude/Latitude", file=sys.stderr)
            sys.exit(1)
        layer_cols = _layer_columns(df, var)
        if len(layer_cols) != 10:
            print(f"WARNING: {var}: expected 10 layer columns, found {len(layer_cols)}", file=sys.stderr)

        lat = df["Latitude"].to_numpy(dtype=np.float64)
        lon = df["Longitude"].to_numpy(dtype=np.float64)
        m_am = amazon.mask(lat, lon)
        m_af = africa.mask(lat, lon)
        base_mask = np.ones(len(df), dtype=bool)
        if args.natveg_filter:
            nv = _optional_natveg_mask(df, static_path)
            if nv is None:
                print("WARNING: --natveg-filter set but could not build mask from test_static_inverse; ignoring.", file=sys.stderr)
            else:
                base_mask = nv
        m_am &= base_mask
        m_af &= base_mask

        mat = df[layer_cols].to_numpy(dtype=np.float64) if layer_cols else np.zeros((len(df), 0))
        row_sum = np.nansum(mat, axis=1) if mat.size else np.zeros(len(df))

        lines.append(f"## {var}")
        lines.append(f"  Cells in Amazon box (after filters): {int(np.sum(m_am))}")
        lines.append(f"  Cells in Africa box (after filters): {int(np.sum(m_af))}")

        for region_name, mask in (("amazon", m_am), ("africa", m_af)):
            pooled = mat[mask].ravel() if mat.size else np.array([])
            st_p = _stats(pooled)
            st_sum = _stats(row_sum[mask])
            lines.append(f"  [{region_name}] all layer values — n={st_p['n']} mean={st_p['mean']:.6g} std={st_p['std']:.6g} p50={st_p['p50']:.6g} min={st_p['min']:.6g} max={st_p['max']:.6g}")
            lines.append(f"  [{region_name}] per-cell sum(10 layers) — n={st_sum['n']} mean={st_sum['mean']:.6g} p50={st_sum['p50']:.6g}")

            for kind, st in (("pooled_layers", st_p), ("row_sum_10L", st_sum)):
                rows_csv.append({"variable": var, "region": region_name, "stat_kind": kind, **st})

        # Per-layer mean contrast (Amazon mean - Africa mean) at each layer
        if mat.size and np.any(m_am) and np.any(m_af):
            prof_am = np.nanmean(mat[m_am], axis=0)
            prof_af = np.nanmean(mat[m_af], axis=0)
            lines.append(f"  Per-layer mean (Amazon - Africa): {np.round(prof_am - prof_af, 6).tolist()}")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out_txt = Path(args.output) if args.output else run_dir / "analysis" / "5p_gt_amazon_vs_africa_report.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(report, encoding="utf-8")
    print(f"Wrote {out_txt}")

    if args.output_csv:
        out_c = Path(args.output_csv)
    else:
        out_c = run_dir / "analysis" / "5p_gt_amazon_vs_africa_stats.csv"
    if rows_csv:
        out_c.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_csv).to_csv(out_c, index=False)
        print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
