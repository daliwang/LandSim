#!/usr/bin/env python3
"""
Generate prediction quality report and top_bad_plots.

Runs docs/generate_prediction_quality_report.py with the same arguments, then
runs cnp_result_validationplot.py with --worst-only to produce analysis/top_bad_plots/
(gt vs pred scatter plots for worst variables and for npool/ppool).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SCRIPT = REPO_ROOT / "docs" / "generate_prediction_quality_report.py"


def main():
    if not DOCS_SCRIPT.exists():
        print(f"Error: {DOCS_SCRIPT} not found.", file=sys.stderr)
        sys.exit(1)

    # 1) Run the docs quality report (same argv)
    r = subprocess.run([sys.executable, str(DOCS_SCRIPT)] + sys.argv[1:])
    if r.returncode != 0:
        sys.exit(r.returncode)

    # 2) Resolve output_dir the same way as the docs script
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="./validation_stats.csv")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--npool-ppool-per-pft-json", default=None)
    args, _ = p.parse_known_args(sys.argv[1:])
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = input_path.resolve()
    output_dir = (Path(args.output_dir).resolve() if args.output_dir else input_path.parent / "analysis")

    report_path = output_dir / "quality_summary_report.txt"
    if not report_path.exists():
        print("Warning: quality_summary_report.txt not found; skipping top_bad_plots.", file=sys.stderr)
        return

    results_dir = str(output_dir.parent.resolve())
    plot_script = REPO_ROOT / "scripts" / "cnp_result_validationplot.py"

    # 3) Generate top_bad_plots (worst variables + npool/ppool) into analysis/top_bad_plots
    cmd = [
        sys.executable, str(plot_script),
        results_dir,
        "--worst-only",
        "--top-bad-report", str(report_path),
        "--plots-dir", "analysis/top_bad_plots",
    ]
    r2 = subprocess.run(cmd)
    if r2.returncode != 0:
        print("Warning: cnp_result_validationplot.py exited with code", r2.returncode, file=sys.stderr)


if __name__ == "__main__":
    main()
