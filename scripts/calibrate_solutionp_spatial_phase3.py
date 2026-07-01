#!/usr/bin/env python3
"""
Deployable Phase 3: spatial multiplicative calibration for solutionp_vr.

Fits on a train split within Amazon/Africa boxes (no GT needed at apply time).
Methods:
  binned   - k(bin, layer) from median clipped GT/Pred ratios; bin by pred-only feature
  learned  - Ridge on log(k) per layer vs pred-only features
  oracle   - per-cell k = clip(GT/Pred) (eval ceiling; requires GT)

Example:
  python scripts/calibrate_solutionp_spatial_phase3.py \\
    --inference-dir cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical/cnp_inference_tropical_only \\
    --method binned --n-bins 8 --bin-feature log1p_occlp_colsum \\
    --output-subdir soil_2d_predictions_solutionp_spatial_binned
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from region_box_utils import region_mask_arrays

VAR = "solutionp_vr"
FIVE_P = ["labilep_vr", "occlp_vr", "solutionp_vr", "secondp_vr", "primp_vr"]
EPS = 1e-12
N_LAYERS = 10

REFERENCE_SITES: Dict[str, Tuple[str, float, float]] = {
    "amazon": ("Amazon", 303.75, -17.434553),
    "africa": ("Africa", 28.0, 0.0),
}

REGION_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "amazon": (-30.0, 10.0, 270.0, 330.0),
    "africa": (-15.0, 15.0, 0.0, 30.0),
}

BIN_FEATURES = [
    "log1p_solutionp_colsum",
    "log1p_occlp_colsum",
    "log1p_total_p_colsum",
    "log1p_solutionp_layer1",
]


def layer_cols(df: pd.DataFrame, var: str) -> List[str]:
    pref = f"Y_{var}_col1_layer"
    cols = [c for c in df.columns if c.startswith(pref)]
    cols.sort(key=lambda c: int(c.replace(pref, "")))
    return cols[:N_LAYERS]


def colsum(mat: np.ndarray) -> np.ndarray:
    return np.nansum(mat, axis=1)


def region_error_metrics(gt_mat: np.ndarray, pred_mat: np.ndarray) -> Dict[str, float]:
    g = np.asarray(gt_mat, dtype=np.float64).ravel()
    p = np.asarray(pred_mat, dtype=np.float64).ravel()
    m = np.isfinite(g) & np.isfinite(p) & (np.abs(g) + np.abs(p) > EPS)
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


def metrics_by_gt_decile(
    gt_mat: np.ndarray, pred_mat: np.ndarray, n_deciles: int = 5
) -> List[Dict[str, object]]:
    col = colsum(gt_mat)
    valid = np.isfinite(col) & (col > EPS)
    if not np.any(valid):
        return []
    edges = np.percentile(col[valid], np.linspace(0, 100, n_deciles + 1))
    edges = np.unique(edges)
    out: List[Dict[str, object]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i < len(edges) - 2:
            mask = valid & (col >= lo) & (col < hi)
        else:
            mask = valid & (col >= lo) & (col <= hi)
        if not np.any(mask):
            continue
        m = region_error_metrics(gt_mat[mask], pred_mat[mask])
        m["decile"] = i + 1
        m["gt_colsum_lo"] = float(lo)
        m["gt_colsum_hi"] = float(hi)
        m["n_cells"] = int(np.sum(mask))
        out.append(m)
    return out


def nearest_row_index(df: pd.DataFrame, slon: float, slat: float) -> int:
    lat = df["Latitude"].astype(float).to_numpy()
    lon = df["Longitude"].astype(float).to_numpy()
    lon360 = lon.copy()
    lon360[lon360 < 0] += 360.0
    slon360 = slon if slon >= 0 else slon + 360.0
    d = (lat - slat) ** 2 + (lon360 - slon360) ** 2
    return int(np.argmin(d))


def site_median_log_ratio(
    gt_df: pd.DataFrame, pr_df: pd.DataFrame, cols: List[str], slon: float, slat: float
) -> float:
    i_gt = nearest_row_index(gt_df, slon, slat)
    i_pr = nearest_row_index(pr_df, slon, slat)
    g = gt_df.loc[i_gt, cols].astype(float).to_numpy()
    p = pr_df.loc[i_pr, cols].astype(float).to_numpy()
    m = np.isfinite(g) & np.isfinite(p) & (p > EPS) & (g > EPS)
    if not np.any(m):
        return np.nan
    return float(np.median(np.abs(np.log(p[m] / g[m]))))


@dataclass
class RegionArrays:
    name: str
    idx: np.ndarray
    gt: np.ndarray
    pred: np.ndarray
    features: np.ndarray
    feature_names: List[str]
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def load_var_matrix(pred_root: Path, var: str, kind: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    sub = "soil_2d_ground_truth" if kind == "gt" else "soil_2d_predictions"
    fname = f"ground_truth_Y_{var}.csv" if kind == "gt" else f"predictions_Y_{var}.csv"
    path = pred_root / sub / fname
    df = pd.read_csv(path)
    cols = layer_cols(df, var)
    mat = df[cols].astype(float).to_numpy()
    return df, mat, cols


def build_features(
    pred_root: Path,
    solutionp_pred: np.ndarray,
    row_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Pred-only features for spatial k (same row order as solutionp CSV)."""
    sol_df, _, sol_cols = load_var_matrix(pred_root, VAR, "pred")
    n = len(sol_df)
    if row_indices is None:
        row_indices = np.arange(n)

    occlp_sum = np.zeros(n)
    total_p = np.zeros(n)
    for var in FIVE_P:
        _, mat, _ = load_var_matrix(pred_root, var, "pred")
        s = colsum(mat)
        total_p += s
        if var == "occlp_vr":
            occlp_sum = s

    sol_sum = colsum(solutionp_pred)
    sol_l1 = solutionp_pred[:, 0]
    lat = sol_df["Latitude"].astype(float).to_numpy()
    lon = sol_df["Longitude"].astype(float).to_numpy()
    lon360 = lon.copy()
    lon360[lon360 < 0] += 360.0

    feats = np.column_stack(
        [
            np.log1p(np.maximum(sol_sum, 0)),
            np.log1p(np.maximum(occlp_sum, 0)),
            np.log1p(np.maximum(total_p, 0)),
            np.log1p(np.maximum(sol_l1, 0)),
            lat,
            lon360 / 360.0,
        ]
    )
    names = [
        "log1p_solutionp_colsum",
        "log1p_occlp_colsum",
        "log1p_total_p_colsum",
        "log1p_solutionp_layer1",
        "lat",
        "lon360_norm",
    ]
    return feats[row_indices], names


def feature_column(feats: np.ndarray, names: List[str], feature_name: str) -> np.ndarray:
    if feature_name not in names:
        raise ValueError(f"Unknown feature {feature_name}; choose from {names}")
    j = names.index(feature_name)
    return feats[:, j]


def train_test_split(n: int, frac_train: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(round(frac_train * n)))
    return perm[:n_train], perm[n_train:]


def clip_ratio_bounds(
    gt: np.ndarray, pred: np.ndarray, cap_low_q: float, cap_high_q: float
) -> Tuple[float, float]:
    valid = np.isfinite(gt) & np.isfinite(pred) & (pred > EPS) & (gt > EPS)
    ratios = (gt[valid] / pred[valid]).astype(np.float64)
    if ratios.size == 0:
        return 1.0, 1.0
    return (
        float(np.percentile(ratios, cap_low_q * 100)),
        float(np.percentile(ratios, cap_high_q * 100)),
    )


def per_layer_clip_ratios(
    gt: np.ndarray, pred: np.ndarray, cap_low_q: float, cap_high_q: float
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.ones(N_LAYERS)
    hi = np.ones(N_LAYERS)
    for li in range(N_LAYERS):
        lo[li], hi[li] = clip_ratio_bounds(gt[:, li], pred[:, li], cap_low_q, cap_high_q)
    return lo, hi


@dataclass
class BinnedModel:
    feature_name: str
    bin_edges: np.ndarray
    k_table: np.ndarray  # (n_bins, n_layers)
    k_lo: np.ndarray
    k_hi: np.ndarray

    def assign_bins(self, feat: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(feat, self.bin_edges[1:-1], right=False), 0, len(self.k_table) - 1)

    def k_for_cells(self, feat: np.ndarray) -> np.ndarray:
        bins = self.assign_bins(feat)
        k = self.k_table[bins]
        return np.clip(k, self.k_lo, self.k_hi)


@dataclass
class LearnedModel:
    feature_names: List[str]
    coef: np.ndarray  # (n_layers, n_features)
    intercept: np.ndarray  # (n_layers,)
    k_lo: np.ndarray
    k_hi: np.ndarray
    ridge_alpha: float

    def log_k(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef.T + self.intercept

    def k_for_cells(self, X: np.ndarray) -> np.ndarray:
        lk = self.log_k(X)
        k = np.exp(lk)
        return np.clip(k, self.k_lo, self.k_hi)


def global_to_local(region: RegionArrays) -> Dict[int, int]:
    return {int(g): i for i, g in enumerate(region.idx)}


def fit_binned(
    region: RegionArrays,
    feature_name: str,
    n_bins: int,
    cap_low_q: float,
    cap_high_q: float,
) -> BinnedModel:
    feat = feature_column(region.features, region.feature_names, feature_name)
    g2l = global_to_local(region)
    train_local = np.array([g2l[int(i)] for i in region.train_idx], dtype=int)
    train_feat = feat[train_local]
    edges = np.unique(np.percentile(train_feat, np.linspace(0, 100, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([train_feat.min(), train_feat.max() + 1e-9])

    n_bins_eff = len(edges) - 1
    k_table = np.ones((n_bins_eff, N_LAYERS))
    gt_tr = region.gt[train_local]
    pr_tr = region.pred[train_local]
    k_lo, k_hi = per_layer_clip_ratios(gt_tr, pr_tr, cap_low_q, cap_high_q)

    feat_tr = feat[train_local]
    bins_tr = np.clip(np.digitize(feat_tr, edges[1:-1], right=False), 0, n_bins_eff - 1)

    for b in range(n_bins_eff):
        mask = bins_tr == b
        if not np.any(mask):
            continue
        for li in range(N_LAYERS):
            g = gt_tr[mask, li]
            p = pr_tr[mask, li]
            valid = np.isfinite(g) & np.isfinite(p) & (p > EPS) & (g > EPS)
            if not np.any(valid):
                k_table[b, li] = 1.0
                continue
            ratios = g[valid] / p[valid]
            k_table[b, li] = float(np.median(ratios))

    k_table = np.clip(k_table, k_lo, k_hi)
    return BinnedModel(feature_name, edges, k_table, k_lo, k_hi)


def fit_learned(
    region: RegionArrays,
    cap_low_q: float,
    cap_high_q: float,
    ridge_alpha: float,
) -> LearnedModel:
    g2l = global_to_local(region)
    train_local = np.array([g2l[int(i)] for i in region.train_idx], dtype=int)
    gt_tr = region.gt[train_local]
    pr_tr = region.pred[train_local]
    X_tr = region.features[train_local]
    k_lo, k_hi = per_layer_clip_ratios(gt_tr, pr_tr, cap_low_q, cap_high_q)

    coef = np.zeros((N_LAYERS, X_tr.shape[1]))
    intercept = np.zeros(N_LAYERS)
    for li in range(N_LAYERS):
        g = gt_tr[:, li]
        p = pr_tr[:, li]
        valid = np.isfinite(g) & np.isfinite(p) & (p > EPS) & (g > EPS)
        if np.sum(valid) < 10:
            intercept[li] = 0.0
            continue
        y = np.log(np.clip(g[valid] / p[valid], k_lo[li], k_hi[li]))
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(X_tr[valid], y)
        coef[li] = model.coef_
        intercept[li] = float(model.intercept_)

    return LearnedModel(region.feature_names, coef, intercept, k_lo, k_hi, ridge_alpha)


def oracle_k(
    gt: np.ndarray, pred: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray
) -> np.ndarray:
    k = np.ones_like(pred)
    for li in range(N_LAYERS):
        g = gt[:, li]
        p = pred[:, li]
        valid = np.isfinite(g) & np.isfinite(p) & (p > EPS)
        k[valid, li] = np.clip(g[valid] / p[valid], k_lo[li], k_hi[li])
    return k


def apply_k(pred: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.maximum(pred * k, 0.0)


def pred_df_from_mat(base_df: pd.DataFrame, pred_mat: np.ndarray, cols: List[str]) -> pd.DataFrame:
    out = base_df[["Longitude", "Latitude"]].copy()
    for i, c in enumerate(cols):
        out[c] = pred_mat[:, i]
    return out


def evaluate_region_split(
    region: RegionArrays,
    pred_corr_region: np.ndarray,
    label: str,
) -> Dict[str, object]:
    """pred_corr_region: (n_cells_in_region, n_layers) aligned with region.idx order."""
    g2l = {int(g): i for i, g in enumerate(region.idx)}
    train_local = np.array([g2l[int(i)] for i in region.train_idx], dtype=int)
    test_local = np.array([g2l[int(i)] for i in region.test_idx], dtype=int)
    all_local = np.arange(len(region.idx), dtype=int)

    out: Dict[str, object] = {"label": label, "region": region.name}
    for split_name, local_idx in [
        ("train", train_local),
        ("test", test_local),
        ("all", all_local),
    ]:
        if local_idx.size == 0 and split_name != "all":
            continue
        out[split_name] = {
            "before": region_error_metrics(region.gt[local_idx], region.pred[local_idx]),
            "after": region_error_metrics(region.gt[local_idx], pred_corr_region[local_idx]),
            "deciles_after": metrics_by_gt_decile(region.gt[local_idx], pred_corr_region[local_idx]),
        }
    return out


def model_to_dict(model: object) -> dict:
    if isinstance(model, BinnedModel):
        return {
            "type": "binned",
            "feature_name": model.feature_name,
            "bin_edges": model.bin_edges.tolist(),
            "k_table": model.k_table.tolist(),
            "k_lo": model.k_lo.tolist(),
            "k_hi": model.k_hi.tolist(),
        }
    if isinstance(model, LearnedModel):
        return {
            "type": "learned_ridge",
            "feature_names": model.feature_names,
            "coef": model.coef.tolist(),
            "intercept": model.intercept.tolist(),
            "k_lo": model.k_lo.tolist(),
            "k_hi": model.k_hi.tolist(),
            "ridge_alpha": model.ridge_alpha,
        }
    raise TypeError(type(model))


def load_models_from_dict(d: dict) -> Dict[str, object]:
    out = {}
    for region_name, m in d.items():
        if m["type"] == "binned":
            out[region_name] = BinnedModel(
                m["feature_name"],
                np.asarray(m["bin_edges"]),
                np.asarray(m["k_table"]),
                np.asarray(m["k_lo"]),
                np.asarray(m["k_hi"]),
            )
        else:
            out[region_name] = LearnedModel(
                m["feature_names"],
                np.asarray(m["coef"]),
                np.asarray(m["intercept"]),
                np.asarray(m["k_lo"]),
                np.asarray(m["k_hi"]),
                m["ridge_alpha"],
            )
    return out


def apply_spatial_models(
    pred_mat: np.ndarray,
    features: np.ndarray,
    feature_names: List[str],
    region_idx: np.ndarray,
    model: object,
) -> np.ndarray:
    out = pred_mat.copy()
    if isinstance(model, BinnedModel):
        feat = feature_column(features, feature_names, model.feature_name)
        k = model.k_for_cells(feat[region_idx])
    else:
        k = model.k_for_cells(features[region_idx])
    out[region_idx] = apply_k(out[region_idx], k)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial Phase 3 calibration for solutionp_vr.")
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument(
        "--method",
        choices=["binned", "learned", "oracle", "compare_all"],
        default="compare_all",
    )
    parser.add_argument("--n-bins", type=int, default=8)
    parser.add_argument("--bin-feature", default="log1p_occlp_colsum", choices=BIN_FEATURES)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cap-low-quantile", type=float, default=0.1)
    parser.add_argument("--cap-high-quantile", type=float, default=0.9)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--eval-json", default=None)
    args = parser.parse_args()

    inf_dir = Path(args.inference_dir).resolve()
    pred_root = inf_dir / "cnp_predictions"
    gt_df, gt_all, cols = load_var_matrix(pred_root, VAR, "gt")
    _, pred_all, _ = load_var_matrix(pred_root, VAR, "pred")
    features_all, feature_names = build_features(pred_root, pred_all)

    lat = gt_df["Latitude"].astype(float).to_numpy()
    lon = gt_df["Longitude"].astype(float).to_numpy()

    regions: Dict[str, RegionArrays] = {}
    for name, box in REGION_BOXES.items():
        mask = region_mask_arrays(lat, lon, box)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        tr, te = train_test_split(idx.size, args.train_frac, args.seed)
        regions[name] = RegionArrays(
            name=name,
            idx=idx,
            gt=gt_all[idx],
            pred=pred_all[idx],
            features=features_all[idx],
            feature_names=feature_names,
            train_idx=idx[tr],
            test_idx=idx[te],
        )

    methods_to_run: List[str]
    if args.method == "compare_all":
        methods_to_run = ["binned", "learned", "oracle"]
    else:
        methods_to_run = [args.method]

    all_eval: Dict[str, object] = {
        "inference_dir": str(inf_dir),
        "train_frac": args.train_frac,
        "seed": args.seed,
        "cap_quantiles": [args.cap_low_quantile, args.cap_high_quantile],
        "methods": {},
        "reference_sites_before": {},
    }

    best_method = None
    best_p90 = np.inf
    best_pred = pred_all.copy()
    best_models: Dict[str, object] = {}

    pred_df_raw = pred_df_from_mat(gt_df, pred_all, cols)

    for site_key, (label, slon, slat) in REFERENCE_SITES.items():
        all_eval["reference_sites_before"][site_key] = {
            "label": label,
            "median_abs_log_ratio": site_median_log_ratio(gt_df, pred_df_raw, cols, slon, slat),
        }

    for method in methods_to_run:
        pred_corr = pred_all.copy()
        models: Dict[str, object] = {}
        region_evals = []

        for rname, region in regions.items():
            if method == "binned":
                model = fit_binned(
                    region,
                    args.bin_feature,
                    args.n_bins,
                    args.cap_low_quantile,
                    args.cap_high_quantile,
                )
            elif method == "learned":
                model = fit_learned(
                    region,
                    args.cap_low_quantile,
                    args.cap_high_quantile,
                    args.ridge_alpha,
                )
            else:
                g2l = global_to_local(region)
                train_local = np.array([g2l[int(i)] for i in region.train_idx], dtype=int)
                gt_tr = region.gt[train_local]
                pr_tr = region.pred[train_local]
                k_lo, k_hi = per_layer_clip_ratios(
                    gt_tr, pr_tr, args.cap_low_quantile, args.cap_high_quantile
                )
                k = oracle_k(region.gt, region.pred, k_lo, k_hi)
                pred_corr_region = apply_k(region.pred, k)
                pred_corr[region.idx] = pred_corr_region
                region_evals.append(evaluate_region_split(region, pred_corr_region, method))
                continue

            models[rname] = model
            pred_corr = apply_spatial_models(
                pred_corr, features_all, feature_names, region.idx, model
            )
            pred_corr_region = pred_corr[region.idx]
            region_evals.append(evaluate_region_split(region, pred_corr_region, method))

        pr_corr_df = pred_df_from_mat(gt_df, pred_corr, cols)
        site_after: Dict[str, float] = {}
        for site_key, (label, slon, slat) in REFERENCE_SITES.items():
            site_after[site_key] = site_median_log_ratio(gt_df, pr_corr_df, cols, slon, slat)

        comb_p90 = np.nanmean(
            [
                re["test"]["after"]["p90_rel_error"]
                for re in region_evals
                if re.get("test") and re["test"]["after"]["n_pairs"] > 0
            ]
        )
        all_eval["methods"][method] = {
            "regions": region_evals,
            "reference_sites_median_abs_log_ratio_after": site_after,
            "mean_test_p90_rel_after": comb_p90,
        }

        print(f"\n=== {method} ===")
        for re in region_evals:
            te = re.get("test", {})
            if not te:
                continue
            b, a = te["before"], te["after"]
            print(
                f"  {re['region']} test: p90_rel {b['p90_rel_error']:.4f} -> {a['p90_rel_error']:.4f}  "
                f"median|log| {b['median_abs_log_ratio']:.4f} -> {a['median_abs_log_ratio']:.4f}  "
                f"frac<0.5 {b['frac_rel_lt_0.5']:.3f} -> {a['frac_rel_lt_0.5']:.3f}"
            )
        for sk, v in site_after.items():
            before = all_eval["reference_sites_before"][sk]["median_abs_log_ratio"]
            print(f"  {sk} site median|log|: {before:.4f} -> {v:.4f}")

        if method != "oracle" and np.isfinite(comb_p90) and comb_p90 < best_p90:
            best_p90 = comb_p90
            best_method = method
            best_pred = pred_corr.copy()
            best_models = models

    deploy_method = args.method if args.method != "compare_all" else (best_method or "binned")
    if args.method == "compare_all":
        print(f"\nBest deployable method by mean test p90: {best_method} (p90={best_p90:.4f})")

    out_subdir = args.output_subdir or f"soil_2d_predictions_solutionp_spatial_{deploy_method}"
    out_dir = pred_root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = gt_df[["Longitude", "Latitude"]].copy()
    for i, c in enumerate(cols):
        out_df[c] = best_pred[:, i]
    out_path = out_dir / f"predictions_Y_{VAR}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    run_dir = inf_dir.parent if inf_dir.name.startswith("cnp_inference") else inf_dir
    params_path = Path(args.params_json) if args.params_json else run_dir / "analysis" / f"solutionp_spatial_phase3_{deploy_method}_params.json"
    eval_path = Path(args.eval_json) if args.eval_json else run_dir / "analysis" / f"solutionp_spatial_phase3_compare_eval.json"

    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_out = {
        "deploy_method": deploy_method,
        "output_predictions": str(out_path),
        "bin_feature": args.bin_feature,
        "n_bins": args.n_bins,
        "regions": {k: model_to_dict(v) for k, v in best_models.items()},
    }
    params_path.write_text(json.dumps(params_out, indent=2), encoding="utf-8")
    all_eval["deploy_method"] = deploy_method
    all_eval["output_predictions"] = str(out_path)
    eval_path.write_text(json.dumps(all_eval, indent=2), encoding="utf-8")
    print(f"Wrote {params_path}")
    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
