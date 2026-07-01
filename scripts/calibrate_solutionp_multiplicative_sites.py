#!/usr/bin/env python3
"""
Multiplicative calibration for solutionp_vr only (Amazon + Africa boxes).

Methods:
  site               - per-layer k = GT/PRED at reference site (can overcorrect region)
  regional_quantile  - per-layer k from regional GT/PRED ratio quantile, clipped to
                       [p_low, p_high] of the same regional distribution
  site_capped        - min(site_k, regional p_high) per layer (cap anchor by regional tail)
  regional_optimal   - per-layer k chosen to minimize regional p90 relative error

Writes corrected predictions, params JSON, and evaluation metrics JSON.

Example:
  python scripts/calibrate_solutionp_multiplicative_sites.py \\
    --inference-dir cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical/cnp_inference_tropical_only \\
    --method regional_quantile \\
    --fit-quantile 0.5 --cap-low-quantile 0.1 --cap-high-quantile 0.9 \\
    --output-subdir soil_2d_predictions_solutionp_mult_calibrated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from region_box_utils import region_mask_arrays

VAR = "solutionp_vr"
EPS = 1e-12

REFERENCE_SITES: Dict[str, Tuple[str, float, float]] = {
    "amazon": ("Amazon", 303.75, -17.434553),
    "africa": ("Africa", 28.0, 0.0),
}

REGION_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "amazon": (-30.0, 10.0, 270.0, 330.0),
    "africa": (-15.0, 15.0, 0.0, 30.0),
}


def layer_cols(df: pd.DataFrame, var: str) -> List[str]:
    pref = f"Y_{var}_col1_layer"
    cols = [c for c in df.columns if c.startswith(pref)]
    cols.sort(key=lambda c: int(c.replace(pref, "")))
    return cols[:10]


def nearest_row_index(df: pd.DataFrame, slon: float, slat: float) -> Tuple[int, float, float]:
    lat = df["Latitude"].astype(float).to_numpy()
    lon = df["Longitude"].astype(float).to_numpy()
    lon360 = lon.copy()
    lon360[lon360 < 0] += 360.0
    slon360 = slon if slon >= 0 else slon + 360.0
    d = (lat - slat) ** 2 + (lon360 - slon360) ** 2
    i = int(np.argmin(d))
    return i, float(lat[i]), float(lon[i])


def fit_layer_scales_site(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    k = np.ones_like(gt)
    detail: Dict[str, List[float]] = {"gt": [], "pred": [], "k": []}
    for li in range(len(gt)):
        g, p = gt[li], pred[li]
        detail["gt"].append(float(g))
        detail["pred"].append(float(p))
        k[li] = float(g / p) if np.isfinite(g) and np.isfinite(p) and p > EPS else 1.0
        detail["k"].append(float(k[li]))
    return k, detail


def fit_layer_scales_regional_quantile(
    gt_mat: np.ndarray,
    pred_mat: np.ndarray,
    fit_quantile: float,
    cap_low_quantile: float,
    cap_high_quantile: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Per-layer k from regional ratio distribution with quantile caps."""
    n_layers = gt_mat.shape[1]
    k = np.ones(n_layers, dtype=np.float64)
    layer_detail: Dict[str, object] = {"k": [], "n_valid": [], "quantiles": []}

    for li in range(n_layers):
        g = gt_mat[:, li]
        p = pred_mat[:, li]
        valid = np.isfinite(g) & np.isfinite(p) & (p > EPS) & (g > EPS)
        ratios = (g[valid] / p[valid]).astype(np.float64)
        if ratios.size == 0:
            k[li] = 1.0
            layer_detail["k"].append(1.0)
            layer_detail["n_valid"].append(0)
            layer_detail["quantiles"].append({})
            continue

        q_fit = float(np.percentile(ratios, fit_quantile * 100.0))
        q_lo = float(np.percentile(ratios, cap_low_quantile * 100.0))
        q_hi = float(np.percentile(ratios, cap_high_quantile * 100.0))
        k[li] = float(np.clip(q_fit, q_lo, q_hi))

        layer_detail["k"].append(float(k[li]))
        layer_detail["n_valid"].append(int(ratios.size))
        layer_detail["quantiles"].append(
            {
                f"p{int(cap_low_quantile * 100)}": q_lo,
                f"p{int(fit_quantile * 100)}": q_fit,
                f"p{int(cap_high_quantile * 100)}": q_hi,
                "median": float(np.median(ratios)),
                "mean": float(np.mean(ratios)),
            }
        )
    return k, layer_detail


def fit_layer_scales_site_capped(
    gt_prof: np.ndarray,
    pred_prof: np.ndarray,
    gt_mat: np.ndarray,
    pred_mat: np.ndarray,
    cap_high_quantile: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    k_site, site_detail = fit_layer_scales_site(gt_prof, pred_prof)
    k_cap, cap_detail = fit_layer_scales_regional_quantile(
        gt_mat, pred_mat, cap_high_quantile, 0.0, cap_high_quantile
    )
    k = np.minimum(k_site, k_cap)
    return k, {"k_site": site_detail["k"], "k_cap_p_high": cap_detail["k"], "k_applied": [float(x) for x in k]}


def fit_layer_scales_regional_optimal(
    gt_mat: np.ndarray,
    pred_mat: np.ndarray,
    cap_low_quantile: float,
    cap_high_quantile: float,
    n_grid: int = 81,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Per-layer scalar k minimizing regional p90 relative error within quantile bounds."""
    n_layers = gt_mat.shape[1]
    k = np.ones(n_layers, dtype=np.float64)
    layer_detail: Dict[str, object] = {"k": [], "p90_at_k": [], "search_range": []}

    for li in range(n_layers):
        g = gt_mat[:, li]
        p = pred_mat[:, li]
        valid = np.isfinite(g) & np.isfinite(p) & (p > EPS) & (g > EPS)
        gv, pv = g[valid], p[valid]
        if gv.size == 0:
            k[li] = 1.0
            layer_detail["k"].append(1.0)
            layer_detail["p90_at_k"].append(np.nan)
            layer_detail["search_range"].append([1.0, 1.0])
            continue

        ratios = gv / pv
        k_lo = float(np.percentile(ratios, cap_low_quantile * 100.0))
        k_hi = float(np.percentile(ratios, cap_high_quantile * 100.0))
        candidates = np.linspace(k_lo, k_hi, n_grid)
        best_k, best_p90 = 1.0, np.inf
        for cand in candidates:
            rel = np.abs(cand * pv - gv) / (np.abs(gv) + EPS)
            p90 = float(np.percentile(rel, 90))
            if p90 < best_p90:
                best_p90, best_k = p90, float(cand)
        k[li] = best_k
        layer_detail["k"].append(best_k)
        layer_detail["p90_at_k"].append(best_p90)
        layer_detail["search_range"].append([k_lo, k_hi])
    return k, layer_detail


def apply_calibration(
    pred_df: pd.DataFrame,
    cols: List[str],
    region_k: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = pred_df.copy()
    lat = out["Latitude"].astype(float).to_numpy()
    lon = out["Longitude"].astype(float).to_numpy()
    for region_name, box in REGION_BOXES.items():
        k = region_k[region_name]
        mask = region_mask_arrays(lat, lon, box)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        for li, col in enumerate(cols):
            vals = out.loc[idx, col].astype(float).to_numpy()
            out.loc[out.index[idx], col] = np.maximum(vals * k[li], 0.0)
    return out


def profile_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(gt) & np.isfinite(pred) & (pred > EPS) & (gt > EPS)
    if not np.any(m):
        return {
            "median_abs_log_ratio": np.nan,
            "max_abs_log_ratio": np.nan,
            "median_rel_error": np.nan,
            "mean_rel_error": np.nan,
        }
    g, p = gt[m], pred[m]
    lr = np.abs(np.log(p / g))
    rel = np.abs(p - g) / (np.abs(g) + EPS)
    return {
        "median_abs_log_ratio": float(np.median(lr)),
        "max_abs_log_ratio": float(np.max(lr)),
        "median_rel_error": float(np.median(rel)),
        "mean_rel_error": float(np.mean(rel)),
    }


def region_error_metrics(gt_mat: np.ndarray, pred_mat: np.ndarray) -> Dict[str, float]:
    g = np.asarray(gt_mat, dtype=np.float64).ravel()
    p = np.asarray(pred_mat, dtype=np.float64).ravel()
    m = np.isfinite(g) & np.isfinite(p) & (g > EPS)
    g, p = g[m], p[m]
    if g.size == 0:
        return {
            "p50_rel_error": np.nan,
            "p90_rel_error": np.nan,
            "median_abs_log_ratio": np.nan,
            "mean_rel_error": np.nan,
            "frac_rel_lt_0.5": np.nan,
            "n_pairs": 0,
        }
    rel = np.abs(p - g) / (np.abs(g) + EPS)
    lr = np.abs(np.log((p + EPS) / (g + EPS)))
    return {
        "p50_rel_error": float(np.percentile(rel, 50)),
        "p90_rel_error": float(np.percentile(rel, 90)),
        "median_abs_log_ratio": float(np.median(lr)),
        "mean_rel_error": float(np.mean(rel)),
        "frac_rel_lt_0.5": float(np.mean(rel < 0.5)),
        "n_pairs": int(g.size),
    }


def evaluate_solutionp(
    gt_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    cols: List[str],
) -> Dict[str, object]:
    lat = gt_df["Latitude"].astype(float).to_numpy()
    lon = gt_df["Longitude"].astype(float).to_numpy()
    gmat = gt_df[cols].astype(float).to_numpy()
    pmat = pr_df[cols].astype(float).to_numpy()

    out: Dict[str, object] = {"regions": {}, "reference_sites": {}}
    for region_name, box in REGION_BOXES.items():
        mask = region_mask_arrays(lat, lon, box)
        if np.any(mask):
            out["regions"][region_name] = region_error_metrics(gmat[mask], pmat[mask])

    for region_name, (site_label, slon, slat) in REFERENCE_SITES.items():
        i_gt, nlat, nlon = nearest_row_index(gt_df, slon, slat)
        i_pr, _, _ = nearest_row_index(pr_df, slon, slat)
        gt_prof = gt_df.loc[i_gt, cols].astype(float).to_numpy()
        pr_prof = pr_df.loc[i_pr, cols].astype(float).to_numpy()
        out["reference_sites"][region_name] = {
            "label": site_label,
            "nearest_lon": nlon,
            "nearest_lat": nlat,
            **profile_metrics(gt_prof, pr_prof),
        }
    return out


def print_eval_summary(label: str, metrics: Dict[str, object]) -> None:
    print(f"\n=== {label} ===")
    for region_name in REGION_BOXES:
        r = metrics["regions"].get(region_name, {})
        print(
            f"  {region_name} region: p50_rel={r.get('p50_rel_error', np.nan):.4f}  "
            f"p90_rel={r.get('p90_rel_error', np.nan):.4f}  "
            f"median|log|={r.get('median_abs_log_ratio', np.nan):.4f}  "
            f"frac_rel<0.5={r.get('frac_rel_lt_0.5', np.nan):.3f}"
        )
    for region_name in REFERENCE_SITES:
        s = metrics["reference_sites"].get(region_name, {})
        print(
            f"  {s.get('label', region_name)} site: median|log|={s.get('median_abs_log_ratio', np.nan):.4f}  "
            f"median_rel={s.get('median_rel_error', np.nan):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicative solutionp_vr calibration.")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument(
        "--method",
        choices=["site", "regional_quantile", "site_capped", "regional_optimal"],
        default="regional_quantile",
        help="Calibration method (see module docstring)",
    )
    parser.add_argument(
        "--fit-quantile",
        type=float,
        default=0.5,
        help="Regional ratio quantile used as k (regional_quantile method)",
    )
    parser.add_argument(
        "--cap-low-quantile",
        type=float,
        default=0.1,
        help="Lower cap on k from regional ratio distribution",
    )
    parser.add_argument(
        "--cap-high-quantile",
        type=float,
        default=0.9,
        help="Upper cap on k from regional ratio distribution",
    )
    parser.add_argument(
        "--output-subdir",
        default="soil_2d_predictions_solutionp_mult_calibrated",
    )
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--eval-json", default=None)
    args = parser.parse_args()

    if not (0.0 <= args.fit_quantile <= 1.0):
        raise ValueError("--fit-quantile must be in [0, 1]")
    if not (0.0 <= args.cap_low_quantile < args.cap_high_quantile <= 1.0):
        raise ValueError("Require 0 <= cap-low-quantile < cap-high-quantile <= 1")

    inf_dir = Path(args.inference_dir).resolve()
    pred_root = inf_dir / "cnp_predictions"
    gt_path = pred_root / "soil_2d_ground_truth" / f"ground_truth_Y_{VAR}.csv"
    pr_path = pred_root / "soil_2d_predictions" / f"predictions_Y_{VAR}.csv"
    if not gt_path.is_file() or not pr_path.is_file():
        raise FileNotFoundError(f"Missing solutionp GT/pred under {pred_root}")

    gt_df = pd.read_csv(gt_path)
    pr_df = pd.read_csv(pr_path)
    cols = layer_cols(gt_df, VAR)
    lat = gt_df["Latitude"].astype(float).to_numpy()
    lon = gt_df["Longitude"].astype(float).to_numpy()
    gmat_all = gt_df[cols].astype(float).to_numpy()
    pmat_all = pr_df[cols].astype(float).to_numpy()

    region_k: Dict[str, np.ndarray] = {}
    params: Dict[str, object] = {
        "variable": VAR,
        "method": args.method,
        "inference_dir": str(inf_dir),
        "fit_quantile": args.fit_quantile,
        "cap_low_quantile": args.cap_low_quantile,
        "cap_high_quantile": args.cap_high_quantile,
        "reference_sites": {},
        "regions": {},
    }

    for region_name, box in REGION_BOXES.items():
        site_label, slon, slat = REFERENCE_SITES[region_name]
        i_gt, nlat, nlon = nearest_row_index(gt_df, slon, slat)
        i_pr, _, _ = nearest_row_index(pr_df, slon, slat)
        gt_prof = gt_df.loc[i_gt, cols].astype(float).to_numpy()
        pr_prof = pr_df.loc[i_pr, cols].astype(float).to_numpy()

        mask = region_mask_arrays(lat, lon, box)
        gt_reg = gmat_all[mask]
        pr_reg = pmat_all[mask]

        if args.method == "site":
            k, layer_detail = fit_layer_scales_site(gt_prof, pr_prof)
            fit_note = "site GT/PRED at anchor"
        elif args.method == "site_capped":
            k, layer_detail = fit_layer_scales_site_capped(
                gt_prof, pr_prof, gt_reg, pr_reg, args.cap_high_quantile
            )
            fit_note = f"min(site_k, regional p{int(args.cap_high_quantile * 100)})"
        elif args.method == "regional_optimal":
            k, layer_detail = fit_layer_scales_regional_optimal(
                gt_reg, pr_reg, args.cap_low_quantile, args.cap_high_quantile
            )
            fit_note = (
                f"minimize regional p90 rel error, k in "
                f"[p{int(args.cap_low_quantile * 100)}, p{int(args.cap_high_quantile * 100)}]"
            )
        else:
            k, layer_detail = fit_layer_scales_regional_quantile(
                gt_reg,
                pr_reg,
                args.fit_quantile,
                args.cap_low_quantile,
                args.cap_high_quantile,
            )
            fit_note = (
                f"clip(p{int(args.fit_quantile * 100)}, "
                f"p{int(args.cap_low_quantile * 100)}, p{int(args.cap_high_quantile * 100)})"
            )

        region_k[region_name] = k
        k_site, _ = fit_layer_scales_site(gt_prof, pr_prof)

        params["reference_sites"][region_name] = {
            "label": site_label,
            "requested_lon": slon,
            "requested_lat": slat,
            "nearest_lon": nlon,
            "nearest_lat": nlat,
            "k_site_per_layer": [float(x) for x in k_site],
        }
        params["regions"][region_name] = {
            "box": list(box),
            "n_cells": int(np.sum(mask)),
            "fit_note": fit_note,
            "k_applied_per_layer": [float(x) for x in k],
            "layer_detail": layer_detail,
        }

        print(f"\n{region_name} ({site_label} @ {nlon:.3f}, {nlat:.3f}) [{args.method}]:")
        print(f"  k applied: {[round(x, 4) for x in k]}")
        if args.method == "regional_quantile":
            print(f"  k at anchor (uncapped site): {[round(x, 4) for x in k_site]}")
        print(f"  anchor median|log ratio| before: {profile_metrics(gt_prof, pr_prof)['median_abs_log_ratio']:.4f}")
        print(f"  anchor median|log ratio| after:  {profile_metrics(gt_prof, pr_prof * k)['median_abs_log_ratio']:.4f}")

    before_eval = evaluate_solutionp(gt_df, pr_df, cols)
    corrected = apply_calibration(pr_df, cols, region_k)
    after_eval = evaluate_solutionp(gt_df, corrected, cols)

    print_eval_summary("Before calibration", before_eval)
    print_eval_summary("After calibration", after_eval)

    out_dir = pred_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_Y_{VAR}.csv"
    corrected.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    run_dir = inf_dir.parent if inf_dir.name.startswith("cnp_inference") else inf_dir
    params_path = (
        Path(args.params_json)
        if args.params_json
        else run_dir / "analysis" / "solutionp_multiplicative_calibration_params.json"
    )
    eval_path = (
        Path(args.eval_json)
        if args.eval_json
        else run_dir / "analysis" / "solutionp_multiplicative_calibration_eval.json"
    )
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params["output_predictions"] = str(out_path)
    params["evaluation"] = {"before": before_eval, "after": after_eval}
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    eval_path.write_text(
        json.dumps({"before": before_eval, "after": after_eval, "method": args.method}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {params_path}")
    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
