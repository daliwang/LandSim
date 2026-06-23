#!/usr/bin/env python3
"""
Two-region 5P soil tools (Amazon / Africa boxes used in the 5P bias workflow).

**CLI change:** you must pass a subcommand: ``gt-summary`` or ``pred-eval``.

Subcommands
-----------

1) **gt-summary** — Ground-truth only: descriptive stats of GT in each region (no pred vs GT).

2) **pred-eval** — For each inference run (phase1, phase2, phase3, …), compare predictions
   to ground truth **at each gridcell** in the **Amazon** box and, separately, in the
   **Africa** box.    Metrics are **within-region only** (no Amazon-vs-Africa comparison).
   The summary CSV includes per-cell aggregates (median/mean/p90 of RMSE, MAE, bias on
   column sums, mean relative error per cell) plus **pooled** metrics over all cells×layers
   in the region: `pooled_rmse`, `pooled_mae`, `pooled_bias_mean`, `pooled_rel_mae`,
   `pooled_r2`, `pooled_pearson_r`, `pooled_n`.

Region bounds (default, same as ``merge_5p_bias_corrected_amazon_africa.py``):

  Amazon: lat [-30, 10], lon [270, 330]   (lon 0–360 °)
  Africa: lat [-15, 15], lon [0, 30]

Paths (per inference folder, e.g. ``…/cnp_inference_entire_dataset``):

  Ground truth: ``cnp_predictions/soil_2d_ground_truth/ground_truth_Y_<var>.csv``
  Predictions:  ``cnp_predictions/<predictions_subdir>/predictions_Y_<var><suffix>.csv``
  (for merged 5P bias output, ``suffix`` is ``_bias_corrected`` — see ``--prediction-filename-suffix`` or ``runs-json``).

Examples
--------

  python scripts/compare_5p_gt_two_regions_inference.py gt-summary \\
    --run-dir cnp_results/run_20260315_202304_phase3_tworegions

  python scripts/compare_5p_gt_two_regions_inference.py pred-eval \\
    --inference-run phase1=/path/.../run_.../cnp_inference_entire_dataset \\
    --inference-run phase2=/path/.../run_.../cnp_inference_entire_dataset \\
    --inference-run phase3_raw=/path/.../run_.../cnp_inference_entire_dataset \\
    --output-summary analysis/5p_pred_eval_summary.csv \\
    --output-per-cell-long analysis/5p_pred_eval_per_cell_long.csv \\
    --output-per-cell-wide analysis/5p_pred_eval_per_cell_wide_rmse.csv

  python scripts/compare_5p_gt_two_regions_inference.py pred-eval \\
    --inference-run phase3_bc=/path/.../cnp_inference_entire_dataset:soil_2d_predictions_5P_bias_corrected_phase2 \\
    --prediction-filename-suffix _bias_corrected \\
    --output-summary analysis/5p_phase3_bias_corrected_summary.csv
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


def _load_regions(
    regions_json: Optional[str],
    amazon_config: str,
    africa_config: str,
) -> Tuple[RegionBox, RegionBox]:
    if regions_json:
        with open(regions_json, "r", encoding="utf-8") as f:
            rj = json.load(f)
        am = rj["amazon"]
        af = rj["africa"]
        return (
            RegionBox(float(am[0]), float(am[1]), float(am[2]), float(am[3]), "amazon"),
            RegionBox(float(af[0]), float(af[1]), float(af[2]), float(af[3]), "africa"),
        )
    am = _first_region_box_from_config(Path(amazon_config))
    af = _first_region_box_from_config(Path(africa_config))
    return RegionBox(am[0], am[1], am[2], am[3], "amazon"), RegionBox(af[0], af[1], af[2], af[3], "africa")


def _parse_inference_runs(
    runs_json: Optional[str],
    inference_run_args: List[str],
    default_pred_suffix: str,
) -> List[Tuple[str, Path, str, str]]:
    """Return list of (label, inference_dir, predictions_subdir, predictions_filename_suffix)."""
    out: List[Tuple[str, Path, str, str]] = []
    if runs_json:
        with open(runs_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            label = entry["label"]
            inf = Path(entry["inference_dir"]).resolve()
            sub = entry.get("predictions_subdir", "soil_2d_predictions")
            # Per-run suffix in JSON; omit key for default "" (bias-corrected runs set "_bias_corrected").
            sfx = entry.get("predictions_filename_suffix", "")
            out.append((label, inf, sub, sfx))
        return out
    for s in inference_run_args:
        if "=" not in s:
            raise ValueError(f"--inference-run must be LABEL=PATH or LABEL=PATH:SUBDIR, got: {s}")
        label, rest = s.split("=", 1)
        label = label.strip()
        rest = rest.strip()
        if ":" in rest:
            # Last colon separates subdir only if rest looks like path:subdir (heuristic: subdir has no /)
            parts = rest.rsplit(":", 1)
            if len(parts) == 2 and "/" not in parts[1] and "\\" not in parts[1]:
                pth, sub = parts[0], parts[1]
            else:
                pth, sub = rest, "soil_2d_predictions"
        else:
            pth, sub = rest, "soil_2d_predictions"
        out.append((label, Path(pth).resolve(), sub, default_pred_suffix))
    return out


def _add_cell_keys(df: pd.DataFrame, ndigits: int = 5) -> pd.DataFrame:
    return df.assign(
        lon_k=np.round(df["Longitude"].to_numpy(dtype=np.float64), ndigits),
        lat_k=np.round(df["Latitude"].to_numpy(dtype=np.float64), ndigits),
    )


def _load_gt_pred_layers(
    inf_dir: Path,
    var: str,
    predictions_subdir: str,
    predictions_filename_suffix: str = "",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    gt_path = inf_dir / "cnp_predictions" / "soil_2d_ground_truth" / f"ground_truth_Y_{var}.csv"
    pr_path = (
        inf_dir / "cnp_predictions" / predictions_subdir / f"predictions_Y_{var}{predictions_filename_suffix}.csv"
    )
    if not gt_path.is_file():
        raise FileNotFoundError(f"Missing ground truth: {gt_path}")
    if not pr_path.is_file():
        raise FileNotFoundError(f"Missing predictions: {pr_path}")
    gt_df = pd.read_csv(gt_path)
    pr_df = pd.read_csv(pr_path)
    gt_cols = _layer_columns(gt_df, var)
    pr_cols = _layer_columns(pr_df, var)
    if len(gt_cols) != len(pr_cols):
        raise ValueError(f"{var}: GT has {len(gt_cols)} layer cols, pred has {len(pr_cols)}")
    return gt_df, gt_cols, pr_cols


def _align_gt_pred(
    gt_df: pd.DataFrame,
    pr_df: pd.DataFrame,
    gt_cols: List[str],
    pr_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lon, lat, gt_mat (n, L), pred_mat (n, L) aligned on rounded (lon_k, lat_k)."""
    g = _add_cell_keys(gt_df[["Longitude", "Latitude"] + gt_cols].copy())
    p = _add_cell_keys(pr_df[["Longitude", "Latitude"] + pr_cols].copy())
    g_idx = g.set_index(["lon_k", "lat_k"])
    p_idx = p.set_index(["lon_k", "lat_k"])
    common = g_idx.index.intersection(p_idx.index)
    if len(common) == 0:
        raise RuntimeError("GT/Pred merge on (lon_k, lat_k) produced zero rows — check coordinates.")
    lon = g_idx.loc[common, "Longitude"].to_numpy(dtype=np.float64)
    lat = g_idx.loc[common, "Latitude"].to_numpy(dtype=np.float64)
    gmat = g_idx.loc[common, gt_cols].to_numpy(dtype=np.float64)
    pmat = p_idx.loc[common, pr_cols].to_numpy(dtype=np.float64)
    return lon, lat, gmat, pmat


def _pooled_vector_metrics(gt_mat: np.ndarray, pred_mat: np.ndarray, eps: float = 1e-20) -> Dict[str, float]:
    """Flatten (n, L) pairs inside region; pooled RMSE, MAE, bias, R², Pearson r (ignoring nan pairs)."""
    g = np.asarray(gt_mat, dtype=np.float64).ravel()
    p = np.asarray(pred_mat, dtype=np.float64).ravel()
    m = np.isfinite(g) & np.isfinite(p)
    g, p = g[m], p[m]
    if g.size < 4:
        return {
            "pooled_n": float(g.size),
            "pooled_rmse": np.nan,
            "pooled_mae": np.nan,
            "pooled_bias_mean": np.nan,
            "pooled_rel_mae": np.nan,
            "pooled_r2": np.nan,
            "pooled_pearson_r": np.nan,
        }
    diff = p - g
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    rel_mae = float(np.clip(np.mean(np.abs(diff) / (np.abs(g) + eps)), 0.0, 1e6))
    ss_res = float(np.sum(diff**2))
    g_mean = float(np.mean(g))
    ss_tot = float(np.sum((g - g_mean) ** 2))
    pooled_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else np.nan
    if np.std(g) > eps and np.std(p) > eps:
        pooled_r = float(np.corrcoef(g, p)[0, 1])
    else:
        pooled_r = np.nan
    return {
        "pooled_n": float(g.size),
        "pooled_rmse": rmse,
        "pooled_mae": mae,
        "pooled_bias_mean": bias,
        "pooled_rel_mae": rel_mae,
        "pooled_r2": pooled_r2,
        "pooled_pearson_r": pooled_r,
    }


def _per_cell_vector_metrics(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-20) -> Dict[str, np.ndarray]:
    """gt, pred shape (n, L). Return dict of length-n arrays."""
    diff = pred - gt
    rmse_l = np.sqrt(np.nanmean(diff**2, axis=1))
    mae_l = np.nanmean(np.abs(diff), axis=1)
    sum_gt = np.nansum(gt, axis=1)
    sum_pr = np.nansum(pred, axis=1)
    bias_sum = sum_pr - sum_gt
    gt_range = np.nanmax(gt, axis=1) - np.nanmin(gt, axis=1)
    # Per-layer relative error; clip extreme ratios when |gt|≈0 so regional means stay interpretable
    ratio = np.abs(diff) / (np.abs(gt) + eps)
    ratio = np.clip(ratio, 0.0, 1e6)
    rel_mae_l = np.nanmean(ratio, axis=1)
    nrmse_sum = np.clip(np.abs(bias_sum) / (gt_range + eps), 0.0, 1e6)
    return {
        "rmse_10L": rmse_l,
        "mae_10L": mae_l,
        "bias_sum": bias_sum,
        "rel_mae_mean_layer": rel_mae_l,
        "nrmse_sum_vs_gt_range": nrmse_sum,
    }


def cmd_gt_summary(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    if args.inference_dir:
        inf_dir = Path(args.inference_dir).resolve()
    else:
        inf_dir = (run_dir / args.inference_subdir).resolve()
    gt_dir = inf_dir / "cnp_predictions" / "soil_2d_ground_truth"
    static_path = inf_dir / "cnp_predictions" / "test_static_inverse.csv"

    amazon, africa = _load_regions(args.regions_json, args.amazon_region_config, args.africa_region_config)

    lines: List[str] = []
    lines.append("5P ground truth: Amazon vs Africa (distribution summary only)")
    lines.append(f"Inference dir: {inf_dir}")
    lines.append(f"Amazon box: [{amazon.lat_min}, {amazon.lat_max}, {amazon.lon_min}, {amazon.lon_max}]")
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
        lat = df["Latitude"].to_numpy(dtype=np.float64)
        lon = df["Longitude"].to_numpy(dtype=np.float64)
        m_am = amazon.mask(lat, lon)
        m_af = africa.mask(lat, lon)
        base_mask = np.ones(len(df), dtype=bool)
        if args.natveg_filter:
            nv = _optional_natveg_mask(df, static_path)
            if nv is None:
                print("WARNING: --natveg-filter set but could not build mask; ignoring.", file=sys.stderr)
            else:
                base_mask = nv
        m_am &= base_mask
        m_af &= base_mask

        mat = df[layer_cols].to_numpy(dtype=np.float64) if layer_cols else np.zeros((len(df), 0))
        row_sum = np.nansum(mat, axis=1) if mat.size else np.zeros(len(df))

        lines.append(f"## {var}")
        lines.append(f"  Cells in Amazon box: {int(np.sum(m_am))}")
        lines.append(f"  Cells in Africa box: {int(np.sum(m_af))}")

        for region_name, mask in (("amazon", m_am), ("africa", m_af)):
            pooled = mat[mask].ravel() if mat.size else np.array([])
            st_p = _stats(pooled)
            st_sum = _stats(row_sum[mask])
            lines.append(f"  [{region_name}] pooled layers — n={st_p['n']} mean={st_p['mean']:.6g} p50={st_p['p50']:.6g}")
            lines.append(f"  [{region_name}] per-cell sum(10L) — n={st_sum['n']} mean={st_sum['mean']:.6g} p50={st_sum['p50']:.6g}")
            for kind, st in (("pooled_layers", st_p), ("row_sum_10L", st_sum)):
                rows_csv.append({"variable": var, "region": region_name, "stat_kind": kind, **st})
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out_txt = Path(args.output) if args.output else run_dir / "analysis" / "5p_gt_amazon_vs_africa_report.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(report, encoding="utf-8")
    print(f"Wrote {out_txt}")

    out_c = Path(args.output_csv) if args.output_csv else run_dir / "analysis" / "5p_gt_amazon_vs_africa_stats.csv"
    if rows_csv:
        out_c.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_csv).to_csv(out_c, index=False)
        print(f"Wrote {out_c}")


def cmd_pred_eval(args: argparse.Namespace) -> None:
    from functools import reduce

    runs = _parse_inference_runs(args.runs_json, args.inference_run, args.prediction_filename_suffix or "")
    if len(runs) < 1:
        print("ERROR: provide --runs-json or at least one --inference-run LABEL=DIR", file=sys.stderr)
        sys.exit(1)

    amazon, africa = _load_regions(args.regions_json, args.amazon_region_config, args.africa_region_config)

    summary_rows: List[dict] = []
    long_rows: List[dict] = []
    wide_fragments_by_label: Dict[str, List[pd.DataFrame]] = {label: [] for label, _, _, _ in runs}

    print("pred-eval: per-cell prediction vs ground truth (Amazon and Africa separately; not cross-region)\n")
    print(f"Runs: {[r[0] for r in runs]}")
    print("Prediction CSV stem: predictions_Y_<var><suffix>.csv (suffix empty unless --prediction-filename-suffix or runs-json).\n")
    print(f"Amazon box: [{amazon.lat_min}, {amazon.lat_max}, {amazon.lon_min}, {amazon.lon_max}]")
    print(f"Africa box: [{africa.lat_min}, {africa.lat_max}, {africa.lon_min}, {africa.lon_max}]\n")

    for label, inf_dir, pred_subdir, pred_sfx in runs:
        static_path = inf_dir / "cnp_predictions" / "test_static_inverse.csv"
        for var in FIVE_P_VARS:
            gt_df, gt_cols, pr_cols = _load_gt_pred_layers(inf_dir, var, pred_subdir, pred_sfx)
            pr_df = pd.read_csv(
                inf_dir / "cnp_predictions" / pred_subdir / f"predictions_Y_{var}{pred_sfx}.csv"
            )
            lon, lat, gmat, pmat = _align_gt_pred(gt_df, pr_df, gt_cols, pr_cols)

            base_mask = np.ones(len(lon), dtype=bool)
            if args.natveg_filter:
                coord_df = pd.DataFrame({"Longitude": lon, "Latitude": lat})
                nv = _optional_natveg_mask(coord_df, static_path)
                if nv is not None and len(nv) == len(lon):
                    base_mask = nv
                else:
                    print(f"WARNING [{label} {var}]: natveg mask not applied.", file=sys.stderr)

            mets = _per_cell_vector_metrics(gmat, pmat)

            for region_box, region_name in ((amazon, "amazon"), (africa, "africa")):
                rmask = region_box.mask(lat, lon) & base_mask
                if not np.any(rmask):
                    print(f"WARNING [{label} {var} {region_name}]: zero cells", file=sys.stderr)
                    continue

                row = {
                    "run_label": label,
                    "predictions_subdir": pred_subdir,
                    "predictions_filename_suffix": pred_sfx,
                    "region": region_name,
                    "variable": var,
                    "n_cells": int(np.sum(rmask)),
                }
                for mk, arr in mets.items():
                    sub = arr[rmask]
                    row["median_" + mk] = float(np.nanmedian(sub))
                    row["mean_" + mk] = float(np.nanmean(sub))
                    row["p90_" + mk] = float(np.nanpercentile(sub, 90))
                pooled = _pooled_vector_metrics(gmat[rmask], pmat[rmask])
                row.update(pooled)
                summary_rows.append(row)

                lon_r = lon[rmask]
                lat_r = lat[rmask]
                lk = np.round(lon_r, 5)
                lak = np.round(lat_r, 5)
                for i in range(len(lon_r)):
                    long_rows.append(
                        {
                            "run_label": label,
                            "predictions_subdir": pred_subdir,
                            "predictions_filename_suffix": pred_sfx,
                            "region": region_name,
                            "variable": var,
                            "Longitude": float(lon_r[i]),
                            "Latitude": float(lat_r[i]),
                            "lon_k": float(lk[i]),
                            "lat_k": float(lak[i]),
                            "rmse_10L": float(mets["rmse_10L"][rmask][i]),
                            "mae_10L": float(mets["mae_10L"][rmask][i]),
                            "bias_sum": float(mets["bias_sum"][rmask][i]),
                            "rel_mae_mean_layer": float(mets["rel_mae_mean_layer"][rmask][i]),
                            "nrmse_sum_vs_gt_range": float(mets["nrmse_sum_vs_gt_range"][rmask][i]),
                        }
                    )

                wide_fragments_by_label[label].append(
                    pd.DataFrame(
                        {
                            "region": region_name,
                            "variable": var,
                            "lon_k": lk,
                            "lat_k": lak,
                            "rmse_10L": mets["rmse_10L"][rmask],
                        }
                    )
                )

    sum_df = pd.DataFrame(summary_rows)

    if args.output_summary:
        outp = Path(args.output_summary)
        outp.parent.mkdir(parents=True, exist_ok=True)
        sum_df.to_csv(outp, index=False)
        print(f"Wrote summary ({len(sum_df)} rows): {outp}")
    else:
        with pd.option_context("display.max_rows", 200, "display.width", 200):
            print(sum_df.to_string(index=False))

    if args.output_per_cell_long and long_rows:
        outp = Path(args.output_per_cell_long)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(long_rows).to_csv(outp, index=False)
        print(f"Wrote per-cell long ({len(long_rows)} rows): {outp}")

    if args.output_per_cell_wide:
        on = ["region", "variable", "lon_k", "lat_k"]
        wide_dfs: List[pd.DataFrame] = []
        for label, _, _, _ in runs:
            frags = wide_fragments_by_label.get(label, [])
            if not frags:
                continue
            w = pd.concat(frags, ignore_index=True)
            w = w.rename(columns={"rmse_10L": f"rmse_10L__{label}"})
            wide_dfs.append(w)
        if wide_dfs:
            wide_df = reduce(lambda a, b: a.merge(b, on=on, how="outer"), wide_dfs)
            outp = Path(args.output_per_cell_wide)
            outp.parent.mkdir(parents=True, exist_ok=True)
            wide_df.to_csv(outp, index=False)
            print(f"Wrote per-cell wide rmse_10L ({len(wide_df)} rows): {outp}")
        else:
            print("No wide table written (no data).", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-region 5P tools: GT summary or pred vs GT per cell.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gt = sub.add_parser("gt-summary", help="GT-only distribution stats per region (legacy).")
    p_gt.add_argument("--run-dir", type=str, required=True)
    p_gt.add_argument("--inference-subdir", type=str, default="cnp_inference_entire_dataset")
    p_gt.add_argument("--inference-dir", type=str, default=None)
    p_gt.add_argument("--amazon-region-config", type=str, default=str(REPO_ROOT / "config/training_config_amazon_5p_box.json"))
    p_gt.add_argument("--africa-region-config", type=str, default=str(REPO_ROOT / "config/training_config_africa_5p_box.json"))
    p_gt.add_argument("--regions-json", type=str, default=None)
    p_gt.add_argument("--natveg-filter", action="store_true")
    p_gt.add_argument("--output", type=str, default=None)
    p_gt.add_argument("--output-csv", type=str, default=None)
    p_gt.set_defaults(func=cmd_gt_summary)

    p_ev = sub.add_parser(
        "pred-eval",
        help="Per-cell pred vs GT in Amazon and Africa; repeat runs for phase1/2/3 comparison.",
    )
    p_ev.add_argument(
        "--runs-json",
        type=str,
        default=None,
        help='JSON list of {"label","inference_dir","predictions_subdir?","predictions_filename_suffix?"}.',
    )
    p_ev.add_argument(
        "--prediction-filename-suffix",
        type=str,
        default="",
        help='Appended to predictions_Y_<var><suffix>.csv for every --inference-run (e.g. _bias_corrected). Per-run: use runs-json.',
    )
    p_ev.add_argument(
        "--inference-run",
        action="append",
        default=[],
        metavar="LABEL=PATH[:SUBDIR]",
        help="Repeat. PATH = cnp_inference_entire_dataset. Optional :predictions_subdir after last path segment.",
    )
    p_ev.add_argument("--amazon-region-config", type=str, default=str(REPO_ROOT / "config/training_config_amazon_5p_box.json"))
    p_ev.add_argument("--africa-region-config", type=str, default=str(REPO_ROOT / "config/training_config_africa_5p_box.json"))
    p_ev.add_argument("--regions-json", type=str, default=None)
    p_ev.add_argument("--natveg-filter", action="store_true")
    p_ev.add_argument("--output-summary", type=str, default=None)
    p_ev.add_argument("--output-per-cell-long", type=str, default=None, help="Long CSV: one row per cell per run per variable.")
    p_ev.add_argument(
        "--output-per-cell-wide",
        type=str,
        default=None,
        help="Wide CSV: keys region,variable,lon_k,lat_k and rmse_10L__<label> per run for easy deltas.",
    )
    p_ev.set_defaults(func=cmd_pred_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
