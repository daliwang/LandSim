#!/usr/bin/env python3
"""
Generate paper-oriented GT vs Pred scatter plots for strong (good) examples.

Default selection covers:
  - scalar fluxes (GPP/NPP/AR/HR)
  - representative 1D PFT pools (All-PFTs + a few high-R2 PFTs)
  - representative 2D soil pools (All-layers + a few high-R2 layers)

Usage:
  python scripts/plot_good_prediction_examples.py \\
    cnp_results/run_20260319_134216_tesnorth10pc
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COORD_COLS = {"long", "lat", "Long", "Lat", "Longitude", "Latitude"}
DEFAULT_SCALAR = ["Y_GPP", "Y_NPP", "Y_AR", "Y_HR"]
DEFAULT_1D = ["leafc", "deadstemc", "totvegc", "livecrootc", "cpool"]
DEFAULT_2D = ["soil4c_vr", "litr3c_vr", "secondp_vr", "occlp_vr"]


def _drop_coords(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in COORD_COLS]
    return df[cols]


def _metrics(gt, pred):
    valid = ~(np.isnan(gt) | np.isnan(pred))
    gt_v, pred_v = np.asarray(gt)[valid], np.asarray(pred)[valid]
    if len(gt_v) < 3:
        return None
    rmse = float(np.sqrt(mean_squared_error(gt_v, pred_v)))
    mae = float(mean_absolute_error(gt_v, pred_v))
    r2 = float(r2_score(gt_v, pred_v))
    return gt_v, pred_v, rmse, mae, r2


def plot_scatter(gt, pred, title, save_path, dpi=300):
    m = _metrics(gt, pred)
    if m is None:
        print(f"  skip (too few points): {title}")
        return False
    gt_v, pred_v, rmse, mae, r2 = m
    lo = float(min(gt_v.min(), pred_v.min()))
    hi = float(max(gt_v.max(), pred_v.max()))
    # avoid zero-span axis for near-constant series
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        pad = abs(hi) * 0.05 + 1e-12
        lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(gt_v, pred_v, s=8, alpha=0.35, c="#1f4e79", edgecolors="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "--", color="#b22222", lw=1.4, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title, fontsize=11)
    txt = f"$R^2$={r2:.3f}\nRMSE={rmse:.4g}\nMAE={mae:.4g}\nN={len(gt_v)}"
    ax.text(
        0.04,
        0.96,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {save_path}  (R2={r2:.3f})")
    return True


def _load_natveg_mask(pred_dir: Path):
    path = pred_dir / "test_static_inverse.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    pct_natveg = df.get("PCT_NATVEG", df.get("pct_natveg"))
    pct_pft0 = df.get("PCT_NAT_PFT_0", df.get("pct_nat_pft_0"))
    if pct_natveg is None or pct_pft0 is None:
        return None
    return (pct_natveg.astype(float).values > 0) & (pct_pft0.astype(float).values < 100)


def _pick_good_pfts(quality_csv: Path, var: str, n: int = 2):
    """Pick up to n good PFTs with largest GT range (visually useful)."""
    if not quality_csv.exists():
        return []
    df = pd.read_csv(quality_csv)
    sub = df[(df["variable"] == var) & (df["prediction_quality"] == "good")].copy()
    if sub.empty:
        return []
    sub["gt_range"] = sub["gt_max"] - sub["gt_min"]
    # Prefer non-trivial range and higher R2
    sub = sub.sort_values(["gt_range", "r2"], ascending=False)
    picks = []
    for _, row in sub.iterrows():
        pft = str(row.get("pft", ""))
        m = re.search(r"pft(\d+)$", pft)
        if not m:
            continue
        pft_num = int(m.group(1))
        # skip near-constant / empty series
        if row["gt_range"] <= 0:
            continue
        picks.append(pft_num)
        if len(picks) >= n:
            break
    return picks


def _pick_good_layers(quality_csv: Path, var: str, n: int = 2):
    if not quality_csv.exists():
        return []
    df = pd.read_csv(quality_csv)
    sub = df[(df["variable"] == var) & (df["prediction_quality"] == "good")].copy()
    if sub.empty or "layer" not in sub.columns:
        return []
    sub["gt_range"] = sub["gt_max"] - sub["gt_min"]
    sub = sub.sort_values(["gt_range", "r2"], ascending=False)
    layers = []
    for _, row in sub.iterrows():
        if pd.isna(row["layer"]):
            continue
        layer = int(row["layer"])
        if row["gt_range"] <= 0:
            continue
        layers.append(layer)
        if len(layers) >= n:
            break
    return layers


def _resolve_pair(gt_dir: Path, pr_dir: Path, var: str):
    candidates = [
        (gt_dir / f"predictions_Y_{var}.csv", pr_dir / f"predictions_Y_{var}.csv"),
        (gt_dir / f"ground_truth_Y_{var}.csv", pr_dir / f"predictions_Y_{var}.csv"),
        (gt_dir / f"Y_{var}.csv", pr_dir / f"Y_{var}.csv"),
    ]
    for gt_path, pr_path in candidates:
        if gt_path.exists() and pr_path.exists():
            return gt_path, pr_path
    return None, None


def plot_scalars(pred_dir: Path, out_dir: Path, vars_: list[str], dpi: int):
    gt = _drop_coords(pd.read_csv(pred_dir / "ground_truth_scalar.csv"))
    pred = _drop_coords(pd.read_csv(pred_dir / "predictions_scalar.csv"))
    for col in vars_:
        if col not in gt.columns or col not in pred.columns:
            print(f"  missing scalar {col}")
            continue
        plot_scatter(
            gt[col].values,
            pred[col].values,
            f"{col} (scalar)",
            out_dir / "scalar" / f"{col}_gt_vs_pred.png",
            dpi=dpi,
        )


def plot_1d(pred_dir: Path, out_dir: Path, vars_: list[str], quality_csv: Path, include_mask, dpi: int):
    gt_dir = pred_dir / "pft_1d_ground_truth"
    pr_dir = pred_dir / "pft_1d_predictions"
    for var in vars_:
        gt_path, pr_path = _resolve_pair(gt_dir, pr_dir, var)
        if gt_path is None:
            print(f"  missing 1D files for {var}")
            continue
        gt = _drop_coords(pd.read_csv(gt_path))
        pred = _drop_coords(pd.read_csv(pr_path))
        common = [c for c in gt.columns if c in pred.columns]
        if not common:
            continue

        gt_all, pred_all = [], []
        for c in common:
            g, p = gt[c].values, pred[c].values
            if include_mask is not None and len(include_mask) == len(g):
                g, p = g[include_mask], p[include_mask]
            gt_all.append(g)
            pred_all.append(p)
        plot_scatter(
            np.concatenate(gt_all),
            np.concatenate(pred_all),
            f"{var} (all PFTs)",
            out_dir / "aggregate_1d" / f"1D_{var}_AllPFTs_gt_vs_pred.png",
            dpi=dpi,
        )

        for pft in _pick_good_pfts(quality_csv, var, n=2):
            col = f"Y_{var}_pft{pft}"
            if col not in common:
                matches = [c for c in common if c.endswith(f"_pft{pft}")]
                if not matches:
                    continue
                col = matches[0]
            g, p = gt[col].values, pred[col].values
            if include_mask is not None and len(include_mask) == len(g):
                g, p = g[include_mask], p[include_mask]
            plot_scatter(
                g,
                p,
                f"{var} PFT{pft}",
                out_dir / "by_pft" / f"1D_{col}_gt_vs_pred.png",
                dpi=dpi,
            )


def plot_2d(pred_dir: Path, out_dir: Path, vars_: list[str], quality_csv: Path, include_mask, dpi: int):
    gt_dir = pred_dir / "soil_2d_ground_truth"
    pr_dir = pred_dir / "soil_2d_predictions"
    for var in vars_:
        gt_path, pr_path = _resolve_pair(gt_dir, pr_dir, var)
        if gt_path is None:
            print(f"  missing 2D files for {var}")
            continue
        gt = _drop_coords(pd.read_csv(gt_path))
        pred = _drop_coords(pd.read_csv(pr_path))
        layer_cols = [c for c in gt.columns if c in pred.columns]
        if not layer_cols:
            continue

        def layer_index(name: str):
            m = re.search(r"(?:layer|lev|_l)(\d+)$", str(name), flags=re.I)
            return int(m.group(1)) if m else None

        indexed = [(layer_index(c), c) for c in layer_cols]
        if all(i is not None for i, _ in indexed):
            by_layer = {}
            for i, c in indexed:
                by_layer.setdefault(i, c)
            ordered = [by_layer[k] for k in sorted(by_layer)[:10]]
        else:
            ordered = layer_cols[:10]

        gt_all, pred_all = [], []
        for c in ordered:
            g, p = gt[c].values, pred[c].values
            if include_mask is not None and len(include_mask) == len(g):
                g, p = g[include_mask], p[include_mask]
            gt_all.append(g)
            pred_all.append(p)
        plot_scatter(
            np.concatenate(gt_all),
            np.concatenate(pred_all),
            f"{var} (all layers)",
            out_dir / "aggregate_2d" / f"2D_{var}_AllLayers_gt_vs_pred.png",
            dpi=dpi,
        )

        for layer in _pick_good_layers(quality_csv, var, n=2):
            if 1 <= layer <= len(ordered):
                c = ordered[layer - 1]
            else:
                matches = [x for x in ordered if str(layer) in str(x)]
                if not matches:
                    continue
                c = matches[0]
            g, p = gt[c].values, pred[c].values
            if include_mask is not None and len(include_mask) == len(g):
                g, p = g[include_mask], p[include_mask]
            plot_scatter(
                g,
                p,
                f"{var} layer {layer}",
                out_dir / "by_layer" / f"2D_{var}_Layer{layer}_gt_vs_pred.png",
                dpi=dpi,
            )


def main():
    p = argparse.ArgumentParser(description="Plot good GT vs Pred examples for papers")
    p.add_argument("results_dir", nargs="?", default=".", help="CNP run directory")
    p.add_argument("--output-dir", default=None, help="Default: <results_dir>/analysis/good_examples")
    p.add_argument("--scalar", default=",".join(DEFAULT_SCALAR))
    p.add_argument("--vars-1d", default=",".join(DEFAULT_1D))
    p.add_argument("--vars-2d", default=",".join(DEFAULT_2D))
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    pred_dir = results_dir / "cnp_predictions"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else results_dir / "analysis" / "good_examples"
    quality_csv = results_dir / "analysis" / "detailed_quality_assessment.csv"
    include_mask = _load_natveg_mask(pred_dir)

    print(f"Results: {results_dir}")
    print(f"Output:  {out_dir}")
    if include_mask is not None:
        print(f"Natveg filter: {int(include_mask.sum())}/{len(include_mask)} gridcells kept")

    print("\n=== Scalar fluxes ===")
    plot_scalars(pred_dir, out_dir, [x.strip() for x in args.scalar.split(",") if x.strip()], args.dpi)
    print("\n=== 1D PFT pools ===")
    plot_1d(
        pred_dir,
        out_dir,
        [x.strip() for x in args.vars_1d.split(",") if x.strip()],
        quality_csv,
        include_mask,
        args.dpi,
    )
    print("\n=== 2D soil pools ===")
    plot_2d(
        pred_dir,
        out_dir,
        [x.strip() for x in args.vars_2d.split(",") if x.strip()],
        quality_csv,
        include_mask,
        args.dpi,
    )
    print(f"\nDone. Open plots under: {out_dir}")


if __name__ == "__main__":
    main()
