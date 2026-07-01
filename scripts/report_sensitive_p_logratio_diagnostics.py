#!/usr/bin/env python3
"""
Report site log-ratio error and regional p90 relative error for ELM-sensitive P pools.

Metrics (solutionp_vr, occlp_vr, labilep_vr):
  - site_median_abs_log_ratio: median over layers of |log((pred+eps)/(gt+eps))|
  - site_max_abs_log_ratio: max over layers
  - region_p90_rel_error: 90th percentile of |pred-gt|/(|gt|+eps) over cells×layers in box

Example:
  python scripts/report_sensitive_p_logratio_diagnostics.py \\
    --run phase2_baseline=cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical/cnp_inference_tropical_only \\
    --run p3moderate=cnp_results/run_20260630_145817_e3smv3_h0_phase2_tropical_p3moderate \\
    --output-csv analysis/sensitive_p_logratio_diagnostics.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from region_box_utils import region_mask_arrays

SENSITIVE_P = ["solutionp_vr", "occlp_vr", "labilep_vr"]
EPS = 1e-12

REFERENCE_SITES: List[Tuple[str, float, float]] = [
    ("Amazon", 303.75, -17.434553),
    ("Africa", 28.0, 0.0),
]

REGION_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "amazon": (-30.0, 10.0, 270.0, 330.0),
    "africa": (-15.0, 15.0, 0.0, 30.0),
}


@dataclass
class RunSpec:
    label: str
    inference_root: Path
    prediction_subdir: str = "soil_2d_predictions"


def find_pred_root(run_or_inf_path: Path) -> Path:
    """Return directory containing cnp_predictions/ (inference dir or run dir)."""
    p = run_or_inf_path.resolve()
    if (p / "cnp_predictions" / "soil_2d_ground_truth").is_dir():
        return p
    for sub in (
        "cnp_inference_tropical_only",
        "cnp_inference_entire_dataset",
        "",
    ):
        cand = p / sub if sub else p
        if (cand / "cnp_predictions" / "soil_2d_ground_truth").is_dir():
            return cand
    if (p / "cnp_predictions" / "soil_2d_ground_truth").is_dir():
        return p
    raise FileNotFoundError(f"No cnp_predictions with soil_2d_ground_truth under {p}")


def layer_cols(df: pd.DataFrame, var: str) -> List[str]:
    pref = f"Y_{var}_col1_layer"
    cols = [c for c in df.columns if c.startswith(pref)]
    cols.sort(key=lambda c: int(c.replace(pref, "")))
    return cols[:10]


def nearest_row_index(df: pd.DataFrame, slon: float, slat: float) -> Tuple[int, float, float, float]:
    lat = df["Latitude"].astype(float).to_numpy()
    lon = df["Longitude"].astype(float).to_numpy()
    lon360 = lon.copy()
    lon360[lon360 < 0] += 360.0
    slon360 = slon if slon >= 0 else slon + 360.0
    d = (lat - slat) ** 2 + (lon360 - slon360) ** 2
    i = int(np.argmin(d))
    return i, float(lat[i]), float(lon[i]), float(np.sqrt(d[i]))


def site_log_ratio_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(gt) & np.isfinite(pred) & ((np.abs(gt) + np.abs(pred)) > EPS)
    if not np.any(m):
        return {"median_abs_log_ratio": np.nan, "max_abs_log_ratio": np.nan, "mean_log_ratio": np.nan}
    log_ratio = np.log((pred[m] + EPS) / (gt[m] + EPS))
    abs_lr = np.abs(log_ratio)
    return {
        "median_abs_log_ratio": float(np.median(abs_lr)),
        "max_abs_log_ratio": float(np.max(abs_lr)),
        "mean_log_ratio": float(np.mean(log_ratio)),
    }


def region_p90_rel_error(gt_mat: np.ndarray, pred_mat: np.ndarray) -> float:
    g = np.asarray(gt_mat, dtype=np.float64).ravel()
    p = np.asarray(pred_mat, dtype=np.float64).ravel()
    m = np.isfinite(g) & np.isfinite(p)
    g, p = g[m], p[m]
    if g.size == 0:
        return np.nan
    rel = np.abs(p - g) / (np.abs(g) + EPS)
    rel = rel[np.isfinite(rel)]
    return float(np.percentile(rel, 90)) if rel.size else np.nan


def load_gt_pred(
    inf_root: Path, var: str, prediction_subdir: str = "soil_2d_predictions"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    pred_root = inf_root / "cnp_predictions"
    gt_path = pred_root / "soil_2d_ground_truth" / f"ground_truth_Y_{var}.csv"
    pr_path = pred_root / prediction_subdir / f"predictions_Y_{var}.csv"
    if not pr_path.is_file() and prediction_subdir != "soil_2d_predictions":
        pr_path = pred_root / "soil_2d_predictions" / f"predictions_Y_{var}.csv"
    if not gt_path.is_file() or not pr_path.is_file():
        raise FileNotFoundError(f"Missing GT or pred for {var} in {pred_root}")
    gt_df = pd.read_csv(gt_path)
    pr_df = pd.read_csv(pr_path)
    cols = layer_cols(gt_df, var)
    return gt_df, pr_df, cols


def evaluate_run(spec: RunSpec) -> List[dict]:
    rows: List[dict] = []
    inf_root = spec.inference_root

    for var in SENSITIVE_P:
        gt_df, pr_df, cols = load_gt_pred(inf_root, var, spec.prediction_subdir)

        for site_name, slon, slat in REFERENCE_SITES:
            i_gt, nlat, nlon, dist = nearest_row_index(gt_df, slon, slat)
            i_pr, _, _, _ = nearest_row_index(pr_df, slon, slat)
            gt_prof = gt_df.loc[i_gt, cols].astype(float).to_numpy()
            pr_prof = pr_df.loc[i_pr, cols].astype(float).to_numpy()
            lr = site_log_ratio_metrics(gt_prof, pr_prof)
            rows.append(
                {
                    "run_label": spec.label,
                    "variable": var,
                    "metric_kind": "site",
                    "site": site_name,
                    "region": "",
                    "nearest_lat": nlat,
                    "nearest_lon": nlon,
                    "nearest_dist_deg": dist,
                    **lr,
                    "p90_rel_error": np.nan,
                }
            )

        lat = gt_df["Latitude"].astype(float).to_numpy()
        lon = gt_df["Longitude"].astype(float).to_numpy()
        gmat = gt_df[cols].astype(float).to_numpy()
        pmat = pr_df[cols].astype(float).to_numpy()

        for region_name, box in REGION_BOXES.items():
            mask = region_mask_arrays(lat, lon, box)
            if not np.any(mask):
                continue
            p90 = region_p90_rel_error(gmat[mask], pmat[mask])
            rows.append(
                {
                    "run_label": spec.label,
                    "variable": var,
                    "metric_kind": "region_p90_rel",
                    "site": "",
                    "region": region_name,
                    "nearest_lat": np.nan,
                    "nearest_lon": np.nan,
                    "nearest_dist_deg": np.nan,
                    "median_abs_log_ratio": np.nan,
                    "max_abs_log_ratio": np.nan,
                    "mean_log_ratio": np.nan,
                    "p90_rel_error": p90,
                    "n_cells": int(np.sum(mask)),
                }
            )
    return rows


def parse_run_args(specs: Sequence[str]) -> List[RunSpec]:
    out: List[RunSpec] = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--run must be LABEL=PATH or LABEL=PATH:SUBDIR, got: {s}")
        label, rest = s.split("=", 1)
        if ":" in rest:
            path_str, subdir = rest.rsplit(":", 1)
            pred_subdir = subdir.strip()
        else:
            path_str, pred_subdir = rest, "soil_2d_predictions"
        inf = find_pred_root(Path(path_str.strip()))
        out.append(RunSpec(label.strip(), inf, pred_subdir))
    return out


def format_summary_table(df: pd.DataFrame) -> str:
    lines = ["Sensitive P diagnostics (log-ratio at sites, p90 rel error in region)\n"]
    site_df = df[df.metric_kind == "site"].copy()
    reg_df = df[df.metric_kind == "region_p90_rel"].copy()

    lines.append("=== Site median |log(pred/GT)| (layers) ===")
    for site in site_df["site"].unique():
        lines.append(f"\n{site}:")
        sub = site_df[site_df.site == site]
        for var in SENSITIVE_P:
            lines.append(f"  {var}:")
            for _, r in sub[sub.variable == var].iterrows():
                lines.append(
                    f"    {r['run_label']:18s}  median|log ratio|={r['median_abs_log_ratio']:.4f}  "
                    f"max={r['max_abs_log_ratio']:.4f}  mean_log={r['mean_log_ratio']:+.4f}"
                )

    lines.append("\n=== Region p90 relative error ===")
    for region in reg_df["region"].unique():
        lines.append(f"\n{region}:")
        sub = reg_df[reg_df.region == region]
        for var in SENSITIVE_P:
            lines.append(f"  {var}:")
            for _, r in sub[sub.variable == var].iterrows():
                lines.append(f"    {r['run_label']:18s}  p90_rel={r['p90_rel_error']:.4f}  n_cells={int(r.get('n_cells', 0))}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Site log-ratio and p90 rel error for sensitive P variables.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeatable: label=run_dir or inference_dir",
    )
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-txt", type=str, default=None, help="Optional human-readable summary")
    args = parser.parse_args()

    if not args.run:
        print("ERROR: provide at least one --run LABEL=PATH", file=sys.stderr)
        sys.exit(1)

    specs = parse_run_args(args.run)
    all_rows: List[dict] = []
    for spec in specs:
        print(f"Evaluating {spec.label} @ {spec.inference_root}")
        all_rows.extend(evaluate_run(spec))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    summary = format_summary_table(df)
    print("\n" + summary)

    out_txt = Path(args.output_txt) if args.output_txt else out_csv.with_suffix(".txt")
    out_txt.write_text(summary + "\n", encoding="utf-8")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
