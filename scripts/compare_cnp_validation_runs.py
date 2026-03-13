#!/usr/bin/env python3
"""
Compare two CNP training runs using validation_stats.csv and optional quality reports.

Use this to compare no-filter vs natveg-aligned runs (or any two runs) on the same
validation metrics. For test-set alignment, run verify_natveg_test_subset.py separately.

Usage:
  python scripts/compare_cnp_validation_runs.py <run_dir_1> <run_dir_2> [--report FILE]

Example:
  python scripts/compare_cnp_validation_runs.py \\
    cnp_results/run_20260226_114546 \\
    cnp_results/run_20260226_114659 \\
    --report comparison_114546_vs_114659.txt
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _find_validation_stats(run_dir: Path) -> Optional[Path]:
    for sub in ("", "analysis"):
        p = run_dir / sub / "validation_stats.csv"
        if p.exists():
            return p
    return None


def _find_quality_report(run_dir: Path) -> Optional[Path]:
    p = run_dir / "analysis" / "quality_summary_report.txt"
    return p if p.exists() else None


def _make_key(df: pd.DataFrame) -> pd.Series:
    pft = df["pft"].fillna("").astype(str) if "pft" in df.columns else pd.Series([""] * len(df))
    layer = df["layer"].fillna("").astype(str) if "layer" in df.columns else pd.Series([""] * len(df))
    return (
        df["type"].astype(str)
        + "|"
        + df["variable"].astype(str)
        + "|"
        + pft
        + "|"
        + layer
    )


def _parse_quality_summary(path: Optional[Path]) -> Optional[dict]:
    if not path or not path.exists():
        return None
    text = path.read_text()
    out = {}
    # Overall Statistics
    m = re.search(r"Good predictions:\s*(\d+)\s*\([^)]+\)", text)
    if m:
        out["good"] = int(m.group(1))
    m = re.search(r"OK predictions:\s*(\d+)\s*\([^)]+\)", text)
    if m:
        out["ok"] = int(m.group(1))
    m = re.search(r"Bad predictions:\s*(\d+)\s*\([^)]+\)", text)
    if m:
        out["bad"] = int(m.group(1))
    m = re.search(r"Good Variables \((\d+) total\)", text)
    if m:
        out["good_variables"] = int(m.group(1))
    return out if out else None


def compare_validation_stats(
    run1: Path,
    run2: Path,
    label1: str,
    label2: str,
) -> Tuple[pd.DataFrame, str]:
    """Load both validation_stats.csv, merge, and return (merged_df, summary_text)."""
    p1 = _find_validation_stats(run1)
    p2 = _find_validation_stats(run2)
    if not p1:
        return pd.DataFrame(), f"Error: validation_stats.csv not found under {run1}"
    if not p2:
        return pd.DataFrame(), f"Error: validation_stats.csv not found under {run2}"

    a = pd.read_csv(p1)
    b = pd.read_csv(p2)
    a["key"] = _make_key(a)
    b["key"] = _make_key(b)
    merged = a.merge(b, on="key", suffixes=("_1", "_2"), how="inner")

    r2_1 = merged["r2_1"].astype(float)
    r2_2 = merged["r2_2"].astype(float)
    rmse_1 = merged["rmse_1"].astype(float)
    rmse_2 = merged["rmse_2"].astype(float)

    better_1 = (r2_1 > r2_2).sum()
    better_2 = (r2_2 > r2_1).sum()
    ties = (r2_1 == r2_2).sum()

    lines = [
        "=== validation_stats comparison (matched rows) ===",
        "",
        f"Run 1 ({label1}): {len(a)} rows  |  Run 2 ({label2}): {len(b)} rows  |  Matched: {len(merged)}",
        "",
        f"Mean R²   — Run 1: {r2_1.mean():.4f}  |  Run 2: {r2_2.mean():.4f}",
        f"Mean RMSE — Run 1: {rmse_1.mean():.4f}  |  Run 2: {rmse_2.mean():.4f}",
        "",
        f"Run 1 better (higher R²): {better_1} of {len(merged)}",
        f"Run 2 better (higher R²): {better_2} of {len(merged)}",
        f"Ties: {ties}",
        "",
        "By type:",
    ]
    type_col = "type_1" if "type_1" in merged.columns else "type"
    for t in merged[type_col].dropna().unique():
        m = merged[merged[type_col] == t]
        r2_1_t = m["r2_1"].astype(float).mean()
        r2_2_t = m["r2_2"].astype(float).mean()
        lines.append(f"  {t}: R² Run1={r2_1_t:.4f}  Run2={r2_2_t:.4f}")
    return merged, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two CNP runs using validation_stats and optional quality reports."
    )
    parser.add_argument("run1", type=Path, help="First run directory (e.g. cnp_results/run_20260226_114546)")
    parser.add_argument("run2", type=Path, help="Second run directory (e.g. cnp_results/run_20260226_114659)")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write a short comparison report to this file",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=None,
        metavar=("LABEL1", "LABEL2"),
        help="Labels for run1 and run2 (default: directory names)",
    )
    args = parser.parse_args()

    run1 = args.run1.resolve()
    run2 = args.run2.resolve()
    if not run1.is_dir():
        print(f"Error: not a directory: {run1}", file=sys.stderr)
        sys.exit(1)
    if not run2.is_dir():
        print(f"Error: not a directory: {run2}", file=sys.stderr)
        sys.exit(1)

    label1 = args.labels[0] if args.labels else run1.name
    label2 = args.labels[1] if args.labels else run2.name

    report_lines = [
        f"Comparison: {label1} vs {label2}",
        f"  Run 1: {run1}",
        f"  Run 2: {run2}",
        "",
    ]

    # Validation stats
    merged, stats_text = compare_validation_stats(run1, run2, label1, label2)
    print(stats_text)
    report_lines.append(stats_text)

    # Quality reports
    q1 = _parse_quality_summary(_find_quality_report(run1))
    q2 = _parse_quality_summary(_find_quality_report(run2))
    if q1 and q2:
        lines = [
            "",
            "=== quality_summary_report (Overall Statistics) ===",
            "",
            f"                Run 1 ({label1})   Run 2 ({label2})",
            f"Good            {q1.get('good', '—')}              {q2.get('good', '—')}",
            f"OK              {q1.get('ok', '—')}              {q2.get('ok', '—')}",
            f"Bad             {q1.get('bad', '—')}              {q2.get('bad', '—')}",
            f"Good variables  {q1.get('good_variables', '—')}              {q2.get('good_variables', '—')}",
        ]
        quality_text = "\n".join(lines)
        print(quality_text)
        report_lines.append(quality_text)

    if args.report:
        args.report.write_text("\n".join(report_lines))
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
