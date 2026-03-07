#!/usr/bin/env python3
"""
Compare PFT 1D and soil 2D variables (from CNP_IO list) between two Amazon-site restart files.
Both inputs should be single-point (Amazon) restarts, e.g. extracted with
extract_elm_restart_point.py at the Amazon site.
- PFT 1D: uses PFT1–PFT16 only (first soil column, 0-based pft indices 1:17; skips PFT0).
- Soil 2D: uses first 10 layers only, first column only (column 0, levgrnd 0:10).
Reports R2, RMSE, MAE, and relative metrics in a style similar to quality_summary_report.txt.
"""
import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list

# Classification thresholds (aligned with quality_summary_report.txt)
THRESH_GOOD = {"r2_min": 0.9, "rel_rmse_max": 0.1, "rel_mae_max": 0.1}
THRESH_OK = {"r2_min": 0.7, "rel_rmse_max": 0.25, "rel_mae_max": 0.25}

# Scope: PFT1–PFT16 (1-based) = 0-based indices 1:17; first 10 soil layers; first column only
PFT_START_1BASED = 1   # PFT1
PFT_END_1BASED = 16    # PFT16
SOIL_N_LAYERS = 10
SOIL_COLUMN_INDEX = 0


def r2_score(y_ref, y_pred):
    """R² (coefficient of determination). y_ref = reference (A), y_pred = comparison (B)."""
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_ref) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    y_ref, y_pred = y_ref[mask], y_pred[mask]
    ss_res = np.sum((y_ref - y_pred) ** 2)
    ss_tot = np.sum((y_ref - np.nanmean(y_ref)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def rmse(y_ref, y_pred):
    """RMSE between two arrays (after masking NaNs)."""
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_ref) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((y_ref[mask] - y_pred[mask]) ** 2))


def mae(y_ref, y_pred):
    """MAE between two arrays (after masking NaNs)."""
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_ref) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(y_ref[mask] - y_pred[mask]))


def relative_rmse(y_ref, y_pred, eps=1e-12):
    """Relative RMSE = RMSE / (mean(|y_ref|) + eps)."""
    r = rmse(y_ref, y_pred)
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    mask = np.isfinite(y_ref)
    if mask.sum() == 0:
        return np.nan
    denom = np.nanmean(np.abs(y_ref[mask])) + eps
    return r / denom if denom > 0 else np.nan


def relative_mae(y_ref, y_pred, eps=1e-12):
    """Relative MAE = MAE / (mean(|y_ref|) + eps)."""
    m = mae(y_ref, y_pred)
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    mask = np.isfinite(y_ref)
    if mask.sum() == 0:
        return np.nan
    denom = np.nanmean(np.abs(y_ref[mask])) + eps
    return m / denom if denom > 0 else np.nan


def classify(r2, rel_rmse, rel_mae):
    """Return 'Good', 'OK', or 'Bad' using same thresholds as quality_summary_report."""
    if np.isnan(r2) or np.isnan(rel_rmse) or np.isnan(rel_mae):
        return "Bad"
    if r2 >= THRESH_GOOD["r2_min"] and rel_rmse <= THRESH_GOOD["rel_rmse_max"] and rel_mae <= THRESH_GOOD["rel_mae_max"]:
        return "Good"
    if r2 >= THRESH_OK["r2_min"] and rel_rmse <= THRESH_OK["rel_rmse_max"] and rel_mae <= THRESH_OK["rel_mae_max"]:
        return "OK"
    return "Bad"


def main():
    parser = argparse.ArgumentParser(
        description="Compare PFT 1D (PFT1–PFT16 only) and soil 2D (first column, first 10 layers) between two Amazon-site restart files."
    )
    parser.add_argument(
        "--restart-a",
        required=True,
        help="First Amazon-site restart NetCDF (e.g. ELM_data/Amazon_...elm.r.0801-01-01-00000.nc)",
    )
    parser.add_argument(
        "--restart-b",
        required=True,
        help="Second Amazon-site restart NetCDF (e.g. .../Amazon_phase2_tropical_4p_restart.nc)",
    )
    parser.add_argument(
        "--variable-list",
        required=True,
        help="CNP_IO file (e.g. CNP_IO_updated9_dev_dw.txt)",
    )
    parser.add_argument("--output", default=None, help="Write report to this file (default: stdout)")
    parser.add_argument("--plot-dir", default=None, help="Base directory to save plots: bad/, good/, ok/ (A vs B by PFT or layer)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for allclose")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for allclose")
    args = parser.parse_args()

    try:
        import xarray as xr
    except ImportError:
        print("Need xarray: pip install xarray", file=sys.stderr)
        sys.exit(1)

    parsed = parse_cnp_io_list(args.variable_list)
    pft_1d_vars = parsed.get("pft_1d_variables", [])
    soil_2d_vars = parsed.get("variables_2d_soil", [])
    if not pft_1d_vars and not soil_2d_vars:
        print("No pft_1d_variables or variables_2d_soil found in variable list.", file=sys.stderr)
        sys.exit(1)

    out = open(args.output, "w") if args.output else sys.stdout

    def log(s=""):
        print(s, file=out)

    log("PFT 1D + Soil 2D variable comparison (Amazon site restarts)")
    log("  Restart A: " + args.restart_a)
    log("  Restart B: " + args.restart_b)
    log("  Variable list: " + args.variable_list)
    log("  Scope: PFT1–PFT16 only (first column); soil 2D: first column, first 10 layers")
    log("")

    ds_a = xr.open_dataset(args.restart_a)
    ds_b = xr.open_dataset(args.restart_b)

    try:
        n_pft_a = int(ds_a.sizes.get("pft", 0))
        n_pft_b = int(ds_b.sizes.get("pft", 0))
        log("  Restart A: pft dimension = {}".format(n_pft_a))
        log("  Restart B: pft dimension = {}".format(n_pft_b))
        if n_pft_a != n_pft_b:
            log("  WARNING: PFT count mismatch. Comparison uses min(n_a, n_b).")
        log("")

        # Table header: Variable | R2 | RMSE | MAE | RelRMSE | RelMAE | Class | Match | Max|diff|
        log("Variable | R2 | RMSE | MAE | RelRMSE | RelMAE | Class | Match | Max|diff|")
        log("-" * 95)

        results = []  # list of (var, r2, rmse_val, mae_val, rel_rmse, rel_mae, cls, match, max_diff, var_type, va, vb)

        for v in pft_1d_vars:
            in_a = v in ds_a.variables
            in_b = v in ds_b.variables
            if not in_a:
                log("{} | - | - | - | - | - | - | - | - | missing in A".format(v))
                continue
            if not in_b:
                log("{} | - | - | - | - | - | - | - | - | missing in B".format(v))
                continue

            var_a = ds_a[v]
            var_b = ds_b[v]
            if "pft" not in var_a.dims:
                log("{} | - | - | - | - | - | - | - | - | not 1D pft in A".format(v))
                continue
            if "pft" not in var_b.dims:
                log("{} | - | - | - | - | - | - | - | - | not 1D pft in B".format(v))
                continue

            vals_a = np.asarray(var_a.values).ravel()
            vals_b = np.asarray(var_b.values).ravel()
            # PFT1–PFT16 only (0-based indices 1:17)
            pft_end = min(PFT_END_1BASED + 1, len(vals_a), len(vals_b))  # 17 if available
            if pft_end <= PFT_START_1BASED:
                log("{} | - | - | - | - | - | - | - | - | need at least {} pft entries".format(v, PFT_END_1BASED + 1))
                continue
            va = vals_a[PFT_START_1BASED:pft_end].astype(float)
            vb = vals_b[PFT_START_1BASED:pft_end].astype(float)

            r2 = r2_score(va, vb)
            rmse_val = rmse(va, vb)
            mae_val = mae(va, vb)
            rel_rmse_val = relative_rmse(va, vb)
            rel_mae_val = relative_mae(va, vb)
            cls = classify(r2, rel_rmse_val, rel_mae_val)
            try:
                match = np.allclose(va, vb, rtol=args.rtol, atol=args.atol, equal_nan=True)
            except Exception:
                match = False
            max_diff = np.nanmax(np.abs(va - vb))

            results.append((v, r2, rmse_val, mae_val, rel_rmse_val, rel_mae_val, cls, match, max_diff, "pft_1d", va, vb))
            log("{} | {:.4f} | {:.6g} | {:.6g} | {:.4f} | {:.4f} | {} | {} | {:.6g}".format(
                v, r2, rmse_val, mae_val, rel_rmse_val, rel_mae_val, cls, match, max_diff))

        # Soil 2D: first column, first 10 layers only
        for v in soil_2d_vars:
            in_a = v in ds_a.variables
            in_b = v in ds_b.variables
            if not in_a:
                log("{} | - | - | - | - | - | - | - | - | missing in A (2D)".format(v))
                continue
            if not in_b:
                log("{} | - | - | - | - | - | - | - | - | missing in B (2D)".format(v))
                continue

            var_a = ds_a[v]
            var_b = ds_b[v]
            dims_a = list(var_a.dims)
            dims_b = list(var_b.dims)
            if "column" not in dims_a or "levgrnd" not in dims_a:
                log("{} | - | - | - | - | - | - | - | - | not (column, levgrnd) in A".format(v))
                continue
            if "column" not in dims_b or "levgrnd" not in dims_b:
                log("{} | - | - | - | - | - | - | - | - | not (column, levgrnd) in B".format(v))
                continue

            try:
                slab_a = var_a.isel(column=SOIL_COLUMN_INDEX, levgrnd=slice(0, SOIL_N_LAYERS))
                slab_b = var_b.isel(column=SOIL_COLUMN_INDEX, levgrnd=slice(0, SOIL_N_LAYERS))
            except Exception as e:
                log("{} | - | - | - | - | - | - | - | - | isel failed: {}".format(v, e))
                continue
            va = np.asarray(slab_a.values, dtype=float).ravel()
            vb = np.asarray(slab_b.values, dtype=float).ravel()
            if len(va) != len(vb) or len(va) == 0:
                log("{} | - | - | - | - | - | - | - | - | length mismatch or zero (2D)".format(v))
                continue

            r2 = r2_score(va, vb)
            rmse_val = rmse(va, vb)
            mae_val = mae(va, vb)
            rel_rmse_val = relative_rmse(va, vb)
            rel_mae_val = relative_mae(va, vb)
            cls = classify(r2, rel_rmse_val, rel_mae_val)
            try:
                match = np.allclose(va, vb, rtol=args.rtol, atol=args.atol, equal_nan=True)
            except Exception:
                match = False
            max_diff = np.nanmax(np.abs(va - vb))

            results.append((v, r2, rmse_val, mae_val, rel_rmse_val, rel_mae_val, cls, match, max_diff, "soil_2d", va, vb))
            log("{} | {:.4f} | {:.6g} | {:.6g} | {:.4f} | {:.4f} | {} | {} | {:.6g}".format(
                v, r2, rmse_val, mae_val, rel_rmse_val, rel_mae_val, cls, match, max_diff))

        log("")
        # List Bad variables
        bad_results = [r for r in results if r[6] == "Bad"]
        if bad_results:
            log("## Bad variables (below OK thresholds)")
            for r in bad_results:
                log("  {}  (R2={:.4f}, RMSE={:.6g}, RelRMSE={:.4f})".format(r[0], r[1], r[2], r[4]))
            log("")

        # Plot variables by class if --plot-dir set (bad/, good/, ok/ subfolders)
        if args.plot_dir and results:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib required for --plot-dir; skipping plots.", file=sys.stderr)
            else:
                base_plot_dir = Path(args.plot_dir)
                base_plot_dir.mkdir(parents=True, exist_ok=True)
                for class_name, class_label in [("Bad", "bad"), ("Good", "good"), ("OK", "ok")]:
                    subset = [r for r in results if r[6] == class_name]
                    if not subset:
                        continue
                    out_dir = base_plot_dir / class_label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for r in subset:
                        var_name, var_type, va, vb = r[0], r[9], np.asarray(r[10]), np.asarray(r[11])
                        n = len(va)
                        if n == 0:
                            continue
                        fig, ax = plt.subplots()
                        if var_type == "pft_1d":
                            x = np.arange(PFT_START_1BASED, PFT_START_1BASED + n, dtype=float)
                            xlabel = "PFT (1-based)"
                        else:
                            x = np.arange(0, n, dtype=float)
                            xlabel = "Soil layer (0-based)"
                        ax.plot(x, va, "o-", label="Restart A (ref)", color="C0", markersize=4)
                        ax.plot(x, vb, "s-", label="Restart B", color="C1", markersize=4)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(var_name)
                        ax.set_title("{}  R2={:.4f}  RMSE={:.4g}".format(var_name, r[1], r[2]))
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        safe_name = var_name.replace("/", "_")
                        fig.savefig(out_dir / "{}.png".format(safe_name), dpi=120, bbox_inches="tight")
                        plt.close(fig)
                log("Plots saved under: {}  (bad/, good/, ok/)".format(base_plot_dir))
                log("")

        log("")
        # Summary section (similar to quality_summary_report.txt)
        log("# Summary (Restart A = reference, Restart B = comparison)")
        log("")
        n_compared = len(results)
        n_good = sum(1 for r in results if r[6] == "Good")
        n_ok = sum(1 for r in results if r[6] == "OK")
        n_bad = sum(1 for r in results if r[6] == "Bad")
        log("## Overall Statistics")
        log("Variables compared: {}".format(n_compared))
        log("Good: {} ({:.1f}%)".format(n_good, 100.0 * n_good / n_compared if n_compared else 0))
        log("OK: {} ({:.1f}%)".format(n_ok, 100.0 * n_ok / n_compared if n_compared else 0))
        log("Bad: {} ({:.1f}%)".format(n_bad, 100.0 * n_bad / n_compared if n_compared else 0))
        log("")
        log("## Classification Thresholds Used")
        log("Good: R2 >= {:.1f}, Relative RMSE <= {:.1f}, Relative MAE <= {:.1f}".format(
            THRESH_GOOD["r2_min"], THRESH_GOOD["rel_rmse_max"], THRESH_GOOD["rel_mae_max"]))
        log("OK: R2 >= {:.1f}, Relative RMSE <= {:.2f}, Relative MAE <= {:.2f}".format(
            THRESH_OK["r2_min"], THRESH_OK["rel_rmse_max"], THRESH_OK["rel_mae_max"]))
        log("Bad: Below OK thresholds")
        log("")
        if results:
            by_r2 = sorted(results, key=lambda x: (np.nan_to_num(x[1], nan=-1), -np.nan_to_num(x[2], nan=np.inf)), reverse=True)
            log("## Variables with Best Agreement (by R2, then lower RMSE)")
            for r in by_r2[:10]:
                log("  {}: R2={:.4f}, RMSE={:.6g}, Class={}".format(r[0], r[1], r[2], r[6]))
            log("")
            log("## Variables with Worst Agreement (lowest R2, worst first)")
            for r in reversed(by_r2[-10:]):
                log("  {}: R2={:.4f}, RMSE={:.6g}, Class={}".format(r[0], r[1], r[2], r[6]))
        log("")
        log("Done.")
    finally:
        ds_a.close()
        ds_b.close()
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
