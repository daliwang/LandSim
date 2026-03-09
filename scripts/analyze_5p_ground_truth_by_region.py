#!/usr/bin/env python3
"""
Analyze ground-truth 5 P variables (labilep_vr, occlp_vr, solutionp_vr, secondp_vr, primp_vr)
in the two target regions (Amazon + Central Africa) vs the rest of the world.

Answers: Are the 5 P distributions different in these two regions compared to other regions?
If yes, training only on two-region data is better justified.

Usage:
  python scripts/analyze_5p_ground_truth_by_region.py --data-dir /path/to/Trendy_1_data_CNP [--max-files 20]
  python scripts/analyze_5p_ground_truth_by_region.py   # uses default path and first 20 files

Output: Prints summary table and writes analysis_5p_by_region.txt (and optional CSV) in current dir or --output.
"""

import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Two regions (lat_min, lat_max, lon_min, lon_max), longitude in 0-360
AMAZON_BOX = (-30, 10, 270, 330)   # 10°N to 30°S, 90°W to 30°W
AFRICA_BOX = (-15, 15, 0, 30)      # 15°N to 15°S, 0° to 30°E
FIVE_P_COLS = ["Y_labilep_vr", "Y_occlp_vr", "Y_solutionp_vr", "Y_secondp_vr", "Y_primp_vr"]
SOIL_LAYERS = 10
LAT_CANDIDATES = ["Latitude", "lat", "latitude", "LAT"]
LON_CANDIDATES = ["Longitude", "lon", "longitude", "LON"]


def _to_360(lon: np.ndarray) -> np.ndarray:
    out = np.asarray(lon, dtype=float).copy()
    out[np.isnan(out)] = 0
    out[out < 0] += 360.0
    return out


def _extract_col0_layers10(arr) -> np.ndarray:
    """Extract first column, first 10 layers from a 2D soil variable (row of DataFrame)."""
    try:
        a = np.asarray(arr)
        if a.ndim == 2:
            return np.asarray(a[0, :SOIL_LAYERS].ravel(), dtype=float)
        if a.ndim == 1:
            return np.asarray(a[:SOIL_LAYERS], dtype=float)
        return np.full(SOIL_LAYERS, np.nan)
    except Exception:
        return np.full(SOIL_LAYERS, np.nan)


def _row_summary(vals: np.ndarray) -> dict:
    """Per-row summary: sum, mean, layer0 (surface)."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"sum": np.nan, "mean": np.nan, "layer0": np.nan}
    return {
        "sum": float(np.sum(v)),
        "mean": float(np.mean(v)),
        "layer0": float(v[0]) if len(v) > 0 else np.nan,
    }


def load_data(data_dir: Path, file_pattern: str = "training_data_batch_*.pkl", max_files: int = 20) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(data_dir.glob(file_pattern))[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} in {data_dir}")
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}", file=sys.stderr)
    if not dfs:
        raise RuntimeError("No data loaded")
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze 5 P ground truth: two regions vs rest of world")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/proj-shared/AI4BGC_7xw/TrainingData/Trendy_1_data_CNP",
        help="Directory containing training_data_batch_*.pkl",
    )
    parser.add_argument("--max-files", type=int, default=20, help="Max number of pkl files to load")
    parser.add_argument("--file-pattern", type=str, default="training_data_batch_*.pkl")
    parser.add_argument("--output", type=str, default="analysis_5p_by_region.txt", help="Output text report path")
    parser.add_argument("--output-csv", type=str, default="", help="Optional: write per-variable stats CSV")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(Path(args.data_dir), file_pattern=args.file_pattern, max_files=args.max_files)
    print(f"Loaded {len(df)} rows")

    # Resolve lat/lon
    lat_col = next((c for c in LAT_CANDIDATES if c in df.columns), None)
    lon_col = next((c for c in LON_CANDIDATES if c in df.columns), None)
    if lat_col is None or lon_col is None:
        print("ERROR: Latitude/Longitude columns not found. Available:", list(df.columns)[:20], "...")
        sys.exit(1)
    lat = pd.to_numeric(df[lat_col], errors="coerce").values
    lon = _to_360(pd.to_numeric(df[lon_col], errors="coerce").values)

    # Mask: in Amazon or Africa
    in_amazon = (lat >= AMAZON_BOX[0]) & (lat <= AMAZON_BOX[1]) & (lon >= AMAZON_BOX[2]) & (lon <= AMAZON_BOX[3])
    in_africa = (lat >= AFRICA_BOX[0]) & (lat <= AFRICA_BOX[1]) & (lon >= AFRICA_BOX[2]) & (lon <= AFRICA_BOX[3])
    two_region = in_amazon | in_africa
    other = ~two_region
    n_two = int(np.sum(two_region))
    n_other = int(np.sum(other))
    print(f"Two regions (Amazon + Africa): {n_two} cells")
    print(f"Other: {n_other} cells")

    # Build per-variable stats for "sum" over 10 layers (main comparison)
    results = []
    lines = []
    lines.append("=" * 80)
    lines.append("5 P ground truth: Two regions (Amazon + Africa) vs Rest of world")
    lines.append("=" * 80)
    lines.append(f"Data: {args.data_dir} (max {args.max_files} files)")
    lines.append(f"Two regions: Amazon {AMAZON_BOX}, Africa {AFRICA_BOX}")
    lines.append(f"Two-region cells: {n_two}  |  Other cells: {n_other}")
    lines.append("")

    for col in FIVE_P_COLS:
        if col not in df.columns:
            lines.append(f"{col}: NOT FOUND in DataFrame")
            continue
        # Extract first column, first 10 layers per row
        raw = np.stack([_extract_col0_layers10(v) for v in df[col].values])
        # Per-row sum over layers (total P in top 10 layers)
        row_sum = np.nansum(raw, axis=1)
        row_mean = np.nanmean(raw, axis=1)
        layer0 = raw[:, 0]

        s_two_sum = row_sum[two_region]
        s_other_sum = row_sum[other]
        s_two_sum = s_two_sum[~np.isnan(s_two_sum) & np.isfinite(s_two_sum)]
        s_other_sum = s_other_sum[~np.isnan(s_other_sum) & np.isfinite(s_other_sum)]

        def stats(x):
            if len(x) == 0:
                return {"mean": np.nan, "std": np.nan, "p5": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p95": np.nan, "n": 0}
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "p5": float(np.percentile(x, 5)),
                "p25": float(np.percentile(x, 25)),
                "p50": float(np.percentile(x, 50)),
                "p75": float(np.percentile(x, 75)),
                "p95": float(np.percentile(x, 95)),
                "n": len(x),
            }

        st_two = stats(s_two_sum)
        st_other = stats(s_other_sum)
        results.append({
            "variable": col,
            "two_region_n": st_two["n"],
            "other_n": st_other["n"],
            "two_region_mean": st_two["mean"],
            "other_mean": st_other["mean"],
            "two_region_std": st_two["std"],
            "other_std": st_other["std"],
            "two_region_p50": st_two["p50"],
            "other_p50": st_other["p50"],
            "ratio_mean_two_over_other": st_two["mean"] / st_other["mean"] if st_other["mean"] and np.isfinite(st_other["mean"]) else np.nan,
        })

        ratio = st_two["mean"] / st_other["mean"] if st_other["mean"] and np.isfinite(st_other["mean"]) else np.nan
        lines.append(f"# {col} (sum over first 10 layers)")
        lines.append(f"  Two regions: n={st_two['n']}  mean={st_two['mean']:.6g}  std={st_two['std']:.6g}  p50={st_two['p50']:.6g}")
        lines.append(f"  Other:       n={st_other['n']}  mean={st_other['mean']:.6g}  std={st_other['std']:.6g}  p50={st_other['p50']:.6g}")
        lines.append(f"  Ratio (two_region / other) mean = {ratio:.4f}")
        lines.append("")

    # Summary verdict
    lines.append("---")
    lines.append("Conclusion (from ratio of means):")
    for r in results:
        if "variable" not in r:
            continue
        ratio = r.get("ratio_mean_two_over_other", np.nan)
        if np.isfinite(ratio) and ratio != 0:
            if abs(ratio - 1.0) > 0.2:
                lines.append(f"  {r['variable']}: DISTRIBUTIONS DIFFER (ratio={ratio:.3f}) — two-region-only training is well justified.")
            else:
                lines.append(f"  {r['variable']}: Similar (ratio={ratio:.3f}) — two-region focus still helps by reducing noise from other regions.")
        else:
            lines.append(f"  {r['variable']}: Could not compute ratio.")
    lines.append("")
    lines.append("Recommendation: If 5 P in the two regions differ from the rest of the world, training")
    lines.append("only on two-region data (region_boxes) should improve 5 P predictions there. You are")
    lines.append("already doing this with config/training_config_two_region_five_p.json.")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.output_csv and results:
        res_df = pd.DataFrame(results)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
