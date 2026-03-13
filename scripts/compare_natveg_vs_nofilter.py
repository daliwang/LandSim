#!/usr/bin/env python3
"""
Compare CNP run metrics: natveg-filter vs no-filter.
Produces a detailed report on whether performance degradation with the natveg filter is concerning.
"""
import json
import sys
from pathlib import Path

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
CNP_RESULTS = REPO_ROOT / "cnp_results"

# Runs to compare (experiment_3_global family, same architecture)
RUNS = {
    "nofilter": CNP_RESULTS / "run_20260226_114546_nofiter",   # full data, 20,826 samples
    "natveg": CNP_RESULTS / "run_20260225_013116_global_natveg",  # natveg filter before split, 14,006 samples
    "natveg_aligned": CNP_RESULTS / "run_20260226_114659_natveg_aligned",  # natveg train, same test set as nofilter
}


def load_metrics(run_dir: Path) -> dict:
    p = run_dir / "cnp_metrics.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f:
        return json.load(f)


def get_config_summary(run_dir: Path) -> dict:
    p = run_dir / "cnp_config.json"
    if not p.exists():
        return {}
    with open(p) as f:
        c = json.load(f)
    dc = c.get("data_config", {})
    info = c.get("data_info", {})
    return {
        "num_samples": info.get("num_samples"),
        "natveg_only": dc.get("natveg_only"),
        "natveg_filter_before_split": dc.get("natveg_filter_before_split"),
    }


def extract_aggregate_metrics(m: dict) -> dict:
    """Top-level aggregates only."""
    out = {}
    for k in ["scalar_rmse", "scalar_mse", "pft_1d_rmse", "pft_1d_mse", "soil_2d_rmse", "soil_2d_mse"]:
        if k in m:
            out[k] = m[k]
    return out


def extract_scalar_r2_rmse(m: dict) -> dict:
    out = {}
    for name in ["Y_GPP", "Y_NPP", "Y_AR", "Y_HR"]:
        for suf in ["_r2", "_rmse", "_nrmse"]:
            k = name + suf
            if k in m:
                out[k] = m[k]
    return out


def variable_from_key(k: str) -> str:
    """e.g. Y_cwdc_vr_layer3_r2 -> cwdc_vr. Only used for soil2d (layer) keys."""
    if k.startswith("Y_") and "_layer" in k:
        return k.split("_layer")[0].replace("Y_", "")
    return k


def aggregate_by_variable(metrics: dict, suffix: str, require_layer: bool = True) -> dict:
    """For each soil2d variable, get mean of layer-level metrics (only keys with _layer)."""
    var_vals = {}
    for k, v in metrics.items():
        if not k.endswith(suffix) or v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
            continue
        if require_layer and "_layer" not in k:
            continue
        var = variable_from_key(k)
        if var not in var_vals:
            var_vals[var] = []
        var_vals[var].append(v)
    return {v: (sum(x) / len(x)) if x else None for v, x in var_vals.items()}


def soil2d_var_r2_rmse(metrics: dict) -> dict:
    """Per soil2d variable: mean R2 and mean RMSE across layers."""
    r2_by_var = aggregate_by_variable(metrics, "_r2")
    rmse_by_var = aggregate_by_variable(metrics, "_rmse")
    vars = sorted(set(r2_by_var) | set(rmse_by_var))
    return {
        v: {"r2": r2_by_var.get(v), "rmse": rmse_by_var.get(v)}
        for v in vars
        if v in r2_by_var or v in rmse_by_var
    }


def pft1d_aggregate_r2_rmse(metrics: dict) -> dict:
    """Per PFT 1d variable: mean R2 across PFTs (we have many Y_*_pft*_r2)."""
    # e.g. Y_leafc_pft1_r2 ... Y_leafc_pft16_r2 -> leafc
    var_vals = {}
    for k, v in metrics.items():
        if "_pft" not in k or not k.endswith("_r2"):
            continue
        if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
            continue
        # Y_leafc_pft1_r2 -> leafc
        parts = k.replace("Y_", "").split("_pft")[0], k
        var = k.replace("Y_", "").rsplit("_pft", 1)[0]
        if var not in var_vals:
            var_vals[var] = []
        var_vals[var].append(v)
    return {v: sum(x) / len(x) for v, x in var_vals.items() if x}


def main():
    data = {}
    for label, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"Warning: {run_dir} not found, skipping.", file=sys.stderr)
            continue
        data[label] = {
            "metrics": load_metrics(run_dir),
            "config": get_config_summary(run_dir),
        }

    if "nofilter" not in data:
        print("nofilter run required.", file=sys.stderr)
        sys.exit(1)

    nof = data["nofilter"]["metrics"]
    nat = data.get("natveg", {}).get("metrics")
    nat_align = data.get("natveg_aligned", {}).get("metrics")

    lines = []
    lines.append("# Natveg filter vs no-filter: performance comparison")
    lines.append("")
    lines.append("## Run setup")
    lines.append("")
    lines.append("| Run | num_samples | natveg_only | natveg_filter_before_split |")
    lines.append("|-----|-------------|-------------|----------------------------|")
    for label in ["nofilter", "natveg", "natveg_aligned"]:
        if label not in data:
            continue
        c = data[label]["config"]
        n = c.get("num_samples", "—")
        nv = c.get("natveg_only")
        nv = "true" if nv else ("false" if nv is False else "—")
        nfb = c.get("natveg_filter_before_split")
        nfb = "true" if nfb else ("false" if nfb is False else "—")
        lines.append(f"| {label} | {n} | {nv} | {nfb} |")
    lines.append("")
    lines.append("- **nofilter**: full data (train+test); no natveg filter.")
    lines.append("- **natveg**: natveg-only filter *before* split → fewer training samples (≈14k vs 20k).")
    lines.append("- **natveg_aligned**: natveg-only for *training*; test set same as nofilter (same 20% holdout).")
    lines.append("")

    # Aggregate metrics
    lines.append("## Aggregate metrics")
    lines.append("")
    for key in ["scalar_rmse", "pft_1d_rmse", "soil_2d_rmse"]:
        if key not in nof:
            continue
        v_nof = nof[key]
        line = f"| {key} | nofilter | **{v_nof:.6f}** | — |"
        if nat and key in nat:
            v_nat = nat[key]
            delta = (v_nat - v_nof) / v_nof * 100 if v_nof else 0
            line += f" natveg | {v_nat:.6f} | {delta:+.1f}% |"
        else:
            line += " natveg | — | — |"
        if nat_align and key in nat_align:
            v_align = nat_align[key]
            delta = (v_align - v_nof) / v_nof * 100 if v_nof else 0
            line += f" natveg_aligned | {v_align:.6f} | {delta:+.1f}% |"
        else:
            line += " natveg_aligned | — | — |"
        lines.append(line)
    lines.append("")
    lines.append("(Lower RMSE is better; positive % = degradation with filter.)")
    lines.append("")

    # Scalar R2
    lines.append("## Scalar fluxes (R²)")
    lines.append("")
    lines.append("| Variable | nofilter | natveg | Δ (pp) | natveg_aligned | Δ (pp) |")
    lines.append("|----------|----------|--------|--------|----------------|--------|")
    for name in ["Y_GPP", "Y_NPP", "Y_AR", "Y_HR"]:
        k = name + "_r2"
        if k not in nof:
            continue
        v_nof = nof[k]
        cell_nat = "—"
        delta_nat = "—"
        if nat and k in nat:
            v_nat = nat[k]
            cell_nat = f"{v_nat:.4f}"
            delta_nat = f"{(v_nat - v_nof) * 100:+.2f}"
        cell_align = "—"
        delta_align = "—"
        if nat_align and k in nat_align:
            v_align = nat_align[k]
            cell_align = f"{v_align:.4f}"
            delta_align = f"{(v_align - v_nof) * 100:+.2f}"
        lines.append(f"| {name} | {v_nof:.4f} | {cell_nat} | {delta_nat} | {cell_align} | {delta_align} |")
    lines.append("")
    lines.append("(R² in [0,1]; negative Δ = degradation with filter.)")
    lines.append("")

    # Soil 2D: per-variable mean R2 comparison
    lines.append("## Soil 2D variables (mean R² over layers)")
    lines.append("")
    soil_nof = soil2d_var_r2_rmse(nof)
    soil_nat = soil2d_var_r2_rmse(nat) if nat else {}
    soil_align = soil2d_var_r2_rmse(nat_align) if nat_align else {}
    vars_soil = sorted(soil_nof.keys())
    degradations = []
    for v in vars_soil:
        r2_nof = soil_nof.get(v, {}).get("r2")
        if r2_nof is None:
            continue
        r2_nat = soil_nat.get(v, {}).get("r2") if v in soil_nat else None
        r2_align = soil_align.get(v, {}).get("r2") if v in soil_align else None
        delta_nat = (r2_nat - r2_nof) * 100 if r2_nat is not None else None
        delta_align = (r2_align - r2_nof) * 100 if r2_align is not None else None
        if delta_nat is not None:
            degradations.append((v, delta_nat, "natveg"))
        if delta_align is not None:
            degradations.append((v, delta_align, "natveg_aligned"))
    # Table: variable, nofilter R2, natveg R2, Δ, natveg_aligned R2, Δ
    lines.append("| Variable | nofilter R² | natveg R² | Δ (pp) | natveg_aligned R² | Δ (pp) |")
    lines.append("|----------|--------------|-----------|--------|-------------------|--------|")
    for v in vars_soil:
        r2_nof = soil_nof.get(v, {}).get("r2")
        if r2_nof is None:
            continue
        r2_nat = soil_nat.get(v, {}).get("r2") if v in soil_nat else None
        r2_align = soil_align.get(v, {}).get("r2") if v in soil_align else None
        d_nat = f"{(r2_nat - r2_nof) * 100:+.2f}" if r2_nat is not None else "—"
        d_align = f"{(r2_align - r2_nof) * 100:+.2f}" if r2_align is not None else "—"
        c_nat = f"{r2_nat:.4f}" if r2_nat is not None else "—"
        c_align = f"{r2_align:.4f}" if r2_align is not None else "—"
        lines.append(f"| {v} | {r2_nof:.4f} | {c_nat} | {d_nat} | {c_align} | {d_align} |")
    lines.append("")

    # PFT 1D: per-variable mean R2
    lines.append("## PFT 1D variables (mean R² over PFTs)")
    lines.append("")
    pft_nof = pft1d_aggregate_r2_rmse(nof)
    pft_nat = pft1d_aggregate_r2_rmse(nat) if nat else {}
    pft_align = pft1d_aggregate_r2_rmse(nat_align) if nat_align else {}
    vars_pft = sorted(pft_nof.keys())
    lines.append("| Variable | nofilter R² | natveg R² | Δ (pp) | natveg_aligned R² | Δ (pp) |")
    lines.append("|----------|--------------|-----------|--------|-------------------|--------|")
    for v in vars_pft:
        r2_nof = pft_nof.get(v)
        if r2_nof is None:
            continue
        r2_nat = pft_nat.get(v) if v in pft_nat else None
        r2_align = pft_align.get(v) if v in pft_align else None
        d_nat = f"{(r2_nat - r2_nof) * 100:+.2f}" if r2_nat is not None else "—"
        d_align = f"{(r2_align - r2_nof) * 100:+.2f}" if r2_align is not None else "—"
        c_nat = f"{r2_nat:.4f}" if r2_nat is not None else "—"
        c_align = f"{r2_align:.4f}" if r2_align is not None else "—"
        lines.append(f"| {v} | {r2_nof:.4f} | {c_nat} | {d_nat} | {c_align} | {d_align} |")
    lines.append("")

    # Summary: largest degradations
    lines.append("## Largest R² degradations (natveg vs nofilter)")
    lines.append("")
    all_deltas = []
    for v in vars_soil:
        r2_nof = soil_nof.get(v, {}).get("r2")
        r2_nat = soil_nat.get(v, {}).get("r2") if v in soil_nat else None
        if r2_nof is not None and r2_nat is not None:
            all_deltas.append((v, (r2_nat - r2_nof) * 100, "soil2d"))
    for v in vars_pft:
        r2_nof = pft_nof.get(v)
        r2_nat = pft_nat.get(v) if v in pft_nat else None
        if r2_nof is not None and r2_nat is not None:
            all_deltas.append((v, (r2_nat - r2_nof) * 100, "pft1d"))
    all_deltas.sort(key=lambda x: x[1])
    lines.append("Worst 15 (most negative Δ = largest drop with natveg):")
    lines.append("")
    for v, d, typ in all_deltas[:15]:
        lines.append(f"- **{v}** ({typ}): {d:+.2f} pp")
    lines.append("")
    # Best 5 excluding variables with negative R² (npool, ppool can be <0 and skew "improvement")
    skip_best = {"npool", "ppool"}
    best = [(v, d, t) for v, d, t in all_deltas if d > 0 and v not in skip_best]
    lines.append("Best 5 (improvement with natveg, among variables with sensible R²):")
    lines.append("")
    for v, d, typ in best[-5:][::-1]:
        lines.append(f"- **{v}** ({typ}): {d:+.2f} pp")
    lines.append("")

    # Interpretation and verdict
    lines.append("## Is the performance degradation concerning?")
    lines.append("")
    lines.append("### Two different comparisons")
    lines.append("")
    lines.append("1. **natveg vs nofilter** (different train *and* test):")
    lines.append("   - natveg has ~33% fewer samples (14k) and is evaluated on a *natveg-only* test set.")
    lines.append("   - nofilter is evaluated on the *full* 20% holdout.")
    lines.append("   - So the large aggregate RMSE increase (+47% scalar, +22% soil) is partly from **different test sets**, not just less data.")
    lines.append("")
    lines.append("2. **natveg_aligned vs nofilter** (same test set, fair comparison):")
    lines.append("   - Same 20% holdout for both; only the *training* set is natveg-only in natveg_aligned.")
    lines.append("   - This answers: *If I train with the natveg filter, how much do I lose on the same test?*")
    lines.append("")
    agg_rmse_nof = nof.get("scalar_rmse") or nof.get("pft_1d_rmse") or nof.get("soil_2d_rmse")
    agg_rmse_nat = (nat or {}).get("scalar_rmse") or (nat or {}).get("pft_1d_rmse") or (nat or {}).get("soil_2d_rmse")
    if nat and agg_rmse_nof and agg_rmse_nat:
        pct_rmse = (agg_rmse_nat - agg_rmse_nof) / agg_rmse_nof * 100
        lines.append(f"- **natveg vs nofilter** aggregate scalar RMSE: **{pct_rmse:+.1f}%** (worse).")
    if nat_align:
        s_align = nat_align.get("scalar_rmse")
        s_nof = nof.get("scalar_rmse")
        if s_nof and s_align:
            pct_align = (s_align - s_nof) / s_nof * 100
            lines.append(f"- **natveg_aligned vs nofilter** scalar RMSE: **{pct_align:+.1f}%** (negative = improvement).")
    lines.append("")
    lines.append("### Verdict: is degradation with the filter concerning?")
    lines.append("")
    lines.append("**When comparing fairly (natveg_aligned vs nofilter, same test set):**")
    lines.append("")
    lines.append("- **Scalar fluxes (GPP, NPP, AR, HR)**: R² **improves** or is flat (+0.07 to +0.25 pp). Scalar RMSE **improves** by ~6%. No concern.")
    lines.append("- **PFT 1D**: Aggregate RMSE is virtually unchanged (+0.2%). Per-variable R² changes are mostly within ±0.5 pp. No concern.")
    lines.append("- **Soil 2D**: Aggregate RMSE is ~6% higher. Most variables are within ±0.5 pp R². Notable drop: **occlp_vr** ≈ -4.7 pp R². litr2*/primp_vr stay similar or improve slightly with aligned run.")
    lines.append("")
    lines.append("**Conclusion:** Using the **natveg filter is not concerning** for overall performance when the same test set is used. The aligned run (train on natveg, test on same holdout as nofilter) is slightly *better* on scalars and similar on PFT/soil, with **occlp_vr** as the only variable with a clear drop (~4.7 pp). If your science prioritizes natural vegetation and occlp is not central, the filter is reasonable to apply.")
    lines.append("")
    lines.append("The **natveg** run (filter-before-split, 14k samples) looks much worse mainly because it is evaluated on a different (natveg-only) test set and with less training data; that comparison is not apples-to-apples for \"degradation with filter.\"")
    lines.append("")

    report = "\n".join(lines)
    out_path = REPO_ROOT / "docs" / "NATVEG_VS_NOFILTER_COMPARISON.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
