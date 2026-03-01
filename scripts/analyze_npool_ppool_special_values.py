#!/usr/bin/env python3
"""
Analyze training dataset (ground truth) for npool and ppool special values.

Checks whether the special values 10 (npool) and 1 (ppool) appear in the
global arrays and counts how many grid cells have these values.

Usage:
  python scripts/analyze_npool_ppool_special_values.py --data-paths /path/to/training_data --file-pattern "training_data_batch_*.pkl"
  python scripts/analyze_npool_ppool_special_values.py --config cnp_results/run_YYYYMMDD_HHMMSS/cnp_config.json

Output: printed summary and optional --output report file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Special values (from ELM/TRENDY observations)
NPOOL_SPECIAL = 10.0
PPOOL_SPECIAL = 1.0
TOL = 1e-5  # tolerance for float comparison


def _to_array(x, length=17):
    """Convert to 1D float array and pad/truncate to length (17 PFTs incl. PFT0, or 16 after drop)."""
    if isinstance(x, np.ndarray):
        arr = np.asarray(x, dtype=np.float64).ravel()
    elif isinstance(x, (list, tuple)):
        arr = np.array(x, dtype=np.float64).ravel()
    else:
        arr = np.array([float(x)], dtype=np.float64)
    if arr.size >= length:
        return arr[:length]
    return np.pad(arr, (0, length - arr.size), mode="constant", constant_values=np.nan)


def _eq_special(arr, value):
    arr = np.asarray(arr, dtype=np.float64)
    return np.abs(arr - value) < TOL


def load_data_from_paths(data_paths, file_pattern, max_files=None):
    """Load all PKL files from data_paths into one DataFrame."""
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    all_dfs = []
    for data_path in data_paths:
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"Warning: path does not exist: {data_path}", file=sys.stderr)
            continue
        files = sorted(data_path.glob(file_pattern))
        if max_files is not None:
            files = files[:max_files]
        for f in files:
            try:
                df = pd.read_pickle(f)
                all_dfs.append(df)
            except Exception as e:
                print(f"Warning: failed to load {f}: {e}", file=sys.stderr)
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def analyze_npool_ppool(df, col_npool="Y_npool", col_ppool="Y_ppool"):
    """Analyze special values in Y_npool and Y_ppool columns."""
    if col_npool not in df.columns:
        raise KeyError(f"Column {col_npool} not in DataFrame. Columns: {list(df.columns)[:20]}...")
    if col_ppool not in df.columns:
        raise KeyError(f"Column {col_ppool} not in DataFrame.")

    n_cells = len(df)
    # Fixed length 17 (PFT0..PFT16); pad/truncate so all rows have same shape
    npool_arrs = np.stack([_to_array(df[col_npool].iloc[i], length=17) for i in range(n_cells)])
    ppool_arrs = np.stack([_to_array(df[col_ppool].iloc[i], length=17) for i in range(n_cells)])
    # Drop PFT0 -> 16 PFTs (PFT1..PFT16)
    npool_arrs = npool_arrs[:, 1:]
    ppool_arrs = ppool_arrs[:, 1:]
    n_pfts = 16

    # Masks: (n_cells, n_pfts)
    npool_is_10 = _eq_special(npool_arrs, NPOOL_SPECIAL)
    ppool_is_1 = _eq_special(ppool_arrs, PPOOL_SPECIAL)

    # Counts
    # Grid cells where ALL PFTs have npool == 10
    cells_all_npool_10 = npool_is_10.all(axis=1).sum()
    # Grid cells where ALL PFTs have ppool == 1
    cells_all_ppool_1 = ppool_is_1.all(axis=1).sum()
    # Grid cells where ALL PFTs have both (npool==10 and ppool==1)
    cells_both = (npool_is_10 & ppool_is_1).all(axis=1).sum()

    # Grid cells where AT LEAST ONE PFT has the special value
    cells_any_npool_10 = npool_is_10.any(axis=1).sum()
    cells_any_ppool_1 = ppool_is_1.any(axis=1).sum()

    # Total (cell, PFT) pairs with special value
    pairs_npool_10 = int(npool_is_10.sum())
    pairs_ppool_1 = int(ppool_is_1.sum())
    total_pairs = n_cells * n_pfts

    # Per-PFT: how many grid cells have that PFT == special value
    cells_per_pft_npool_10 = npool_is_10.sum(axis=0)
    cells_per_pft_ppool_1 = ppool_is_1.sum(axis=0)

    return {
        "n_cells": n_cells,
        "n_pfts": n_pfts,
        "total_pairs": total_pairs,
        "cells_all_npool_10": int(cells_all_npool_10),
        "cells_all_ppool_1": int(cells_all_ppool_1),
        "cells_both_all": int(cells_both),
        "cells_any_npool_10": int(cells_any_npool_10),
        "cells_any_ppool_1": int(cells_any_ppool_1),
        "pairs_npool_10": pairs_npool_10,
        "pairs_ppool_1": pairs_ppool_1,
        "pct_pairs_npool_10": 100.0 * pairs_npool_10 / total_pairs if total_pairs else 0,
        "pct_pairs_ppool_1": 100.0 * pairs_ppool_1 / total_pairs if total_pairs else 0,
        "cells_per_pft_npool_10": cells_per_pft_npool_10.tolist(),
        "cells_per_pft_ppool_1": cells_per_pft_ppool_1.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze npool/ppool special values (10 and 1) in training data.")
    ap.add_argument("--data-paths", type=str, nargs="+", help="Paths to directories containing training PKL files.")
    ap.add_argument("--file-pattern", type=str, default="training_data_batch_*.pkl", help="Glob pattern for PKL files.")
    ap.add_argument("--config", type=str, help="Path to cnp_config.json; overrides --data-paths and --file-pattern from data_config.")
    ap.add_argument("--max-files", type=int, default=None, help="Limit number of PKL files to load (default: all).")
    ap.add_argument("--output", type=str, help="Write JSON report to this file.")
    args = ap.parse_args()

    data_paths = args.data_paths
    file_pattern = args.file_pattern
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        dc = cfg.get("data_config") or {}
        data_paths = dc.get("data_paths") or data_paths
        file_pattern = dc.get("file_pattern") or file_pattern
        if not data_paths:
            print("No data_paths in config.", file=sys.stderr)
            sys.exit(1)

    if not data_paths:
        print("Provide --data-paths or --config.", file=sys.stderr)
        sys.exit(1)

    df = load_data_from_paths(data_paths, file_pattern, max_files=args.max_files)
    if df is None or len(df) == 0:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} grid cells from {data_paths} ({file_pattern}).")
    try:
        out = analyze_npool_ppool(df)
    except KeyError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # Report
    print()
    print("=== NPOOL / PPOOL special values in training dataset (ground truth) ===")
    print()
    print(f"Total grid cells: {out['n_cells']}")
    print(f"PFTs per cell:    {out['n_pfts']} (PFT1..PFT{out['n_pfts']})")
    print(f"Total (cell, PFT) pairs: {out['total_pairs']}")
    print()
    print("Special value: npool == 10")
    print(f"  Grid cells where ALL PFTs have npool == 10:  {out['cells_all_npool_10']}  ({100*out['cells_all_npool_10']/out['n_cells']:.1f}% of cells)")
    print(f"  Grid cells where ANY PFT has npool == 10:    {out['cells_any_npool_10']}  ({100*out['cells_any_npool_10']/out['n_cells']:.1f}% of cells)")
    print(f"  Total (cell, PFT) pairs with npool == 10:    {out['pairs_npool_10']}  ({out['pct_pairs_npool_10']:.1f}% of pairs)")
    print()
    print("Special value: ppool == 1")
    print(f"  Grid cells where ALL PFTs have ppool == 1:   {out['cells_all_ppool_1']}  ({100*out['cells_all_ppool_1']/out['n_cells']:.1f}% of cells)")
    print(f"  Grid cells where ANY PFT has ppool == 1:     {out['cells_any_ppool_1']}  ({100*out['cells_any_ppool_1']/out['n_cells']:.1f}% of cells)")
    print(f"  Total (cell, PFT) pairs with ppool == 1:     {out['pairs_ppool_1']}  ({out['pct_pairs_ppool_1']:.1f}% of pairs)")
    print()
    print(f"Grid cells where ALL PFTs have BOTH npool==10 and ppool==1: {out['cells_both_all']}  ({100*out['cells_both_all']/out['n_cells']:.1f}% of cells)")
    print()
    print("Per-PFT: number of grid cells with npool==10 (PFT1..PFT16):")
    print("  " + ", ".join(str(x) for x in out["cells_per_pft_npool_10"]))
    print("Per-PFT: number of grid cells with ppool==1 (PFT1..PFT16):")
    print("  " + ", ".join(str(x) for x in out["cells_per_pft_ppool_1"]))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
