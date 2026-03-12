import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# This script lives in the same directory as extract_elm_restart_point.py
from extract_elm_restart_point import extract_single_point_elm


# Fixed run directories and global restart paths (same as current workflow)
PHASE1_RUN_DIR = (
    "/mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20260311_165538_phase1_global"
)
PHASE2_RUN_DIR = (
    "/mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20260311_174843_phase2_tropical"
)
PHASE3_RUN_DIR = (
    "/mnt/proj-shared/AI4BGC_7xw/AI4BGC/cnp_results/run_20260311_204845_phase3_tworegions"
)

PHASE1_GLOBAL_RESTART = os.path.join(PHASE1_RUN_DIR, "updated_restart_base.nc")
PHASE2_GLOBAL_RESTART = os.path.join(
    PHASE2_RUN_DIR, "updated_restart_phase2_tropical_5P_raw.nc"
)
PHASE3_GLOBAL_RESTART = os.path.join(
    PHASE3_RUN_DIR, "updated_restart_phase3_tworegions_5P_bias_corrected_tropical.nc"
)

# Ground-truth directory (per-variable CSVs ground_truth_Y_{var}.csv)
GT_PROFILES_DIR = os.path.join(
    PHASE1_RUN_DIR,
    "cnp_inference_entire_dataset",
    "cnp_predictions",
    "soil_2d_ground_truth",
)

FIVE_P = ["labilep_vr", "occlp_vr", "solutionp_vr", "secondp_vr", "primp_vr"]
LEV_DIM = "levgrnd"


def _safe_site_name(site_name: str, lon: float, lat: float) -> str:
    """Create a filesystem-safe site identifier."""
    if site_name:
        base = site_name.strip().replace(" ", "_")
    else:
        base = "site"
    return f"{base}_lon{lon:.2f}_lat{lat:.2f}".replace(".", "p").replace("-", "m")


def _find_row_for_site(df: pd.DataFrame, lon: float, lat: float, atol: float = 1e-4) -> int:
    """Return row index matching lon/lat; fall back to nearest neighbour if needed."""
    if "Longitude" not in df.columns or "Latitude" not in df.columns:
        raise ValueError("DataFrame must contain 'Longitude' and 'Latitude' columns.")

    lon_vals = df["Longitude"].values
    lat_vals = df["Latitude"].values
    mask = np.isclose(lon_vals, lon, atol=atol) & np.isclose(lat_vals, lat, atol=atol)

    idx = np.where(mask)[0]
    if idx.size == 1:
        return int(idx[0])

    # Fallback: nearest neighbour
    d2 = (lon_vals - lon) ** 2 + (lat_vals - lat) ** 2
    return int(np.argmin(d2))


def _extract_profile_from_gt(df: pd.DataFrame, var: str, row_idx: int) -> np.ndarray:
    """Extract vertical profile for one variable at row_idx from GT CSV."""
    prefix = f"Y_{var}_col1_layer"
    layer_cols = sorted(
        [c for c in df.columns if c.startswith(prefix)],
        key=lambda c: int(c.split("layer")[-1]),
    )
    if not layer_cols:
        raise ValueError(f"No columns starting with '{prefix}' found.")
    values = df.loc[row_idx, layer_cols].astype(float).values
    return values


def _load_profiles_from_restart(path: str) -> Dict[str, np.ndarray]:
    """Load 1D vertical profiles (top 10 layers) for the 5P variables from a single-point restart."""
    ds = xr.open_dataset(path, decode_times=False)
    profs: Dict[str, np.ndarray] = {}
    for var in FIVE_P:
        if var not in ds:
            print(f"  WARNING: {var} not found in {path}")
            continue
        da = ds[var]
        arr = da.values
        if arr.ndim == 1:
            prof = arr
        elif arr.ndim == 2:
            if da.dims[0] == LEV_DIM:
                prof = arr[:, 0]
            elif da.dims[1] == LEV_DIM:
                prof = arr[0, :]
            else:
                prof = arr.reshape(-1)
        else:
            prof = arr.reshape(-1)
        profs[var] = np.asarray(prof[:10], dtype=float)
    return profs


def generate_site_restarts(lon: float, lat: float, site_name: str) -> Dict[str, str]:
    """Extract phase1/phase2/phase3 single-point restarts for this site."""
    site_id = _safe_site_name(site_name, lon, lat)

    out_paths = {
        "phase1_global": os.path.join(PHASE1_RUN_DIR, f"{site_id}_phase1_global_restart.nc"),
        "phase2_tropical": os.path.join(
            PHASE2_RUN_DIR, f"{site_id}_phase2_tropical_restart.nc"
        ),
        "phase3_tworegions": os.path.join(
            PHASE3_RUN_DIR, f"{site_id}_phase3_tworegions_restart.nc"
        ),
    }

    print(f"\n=== Extracting single-point restarts for {site_id} ===")
    extract_single_point_elm(
        source_nc=PHASE1_GLOBAL_RESTART,
        output_nc=out_paths["phase1_global"],
        target_lat=lat,
        target_lon=lon,
    )
    extract_single_point_elm(
        source_nc=PHASE2_GLOBAL_RESTART,
        output_nc=out_paths["phase2_tropical"],
        target_lat=lat,
        target_lon=lon,
    )
    extract_single_point_elm(
        source_nc=PHASE3_GLOBAL_RESTART,
        output_nc=out_paths["phase3_tworegions"],
        target_lat=lat,
        target_lon=lon,
    )

    return out_paths


def compare_5p_for_site(
    lon: float,
    lat: float,
    site_name: str,
    restart_paths: Dict[str, str],
    output_dir: str,
) -> None:
    """Compute and plot 5P profiles for GT + three restarts at the given site."""
    os.makedirs(output_dir, exist_ok=True)
    site_id = _safe_site_name(site_name, lon, lat)

    # Load ground truth profiles
    gt_profiles: Dict[str, np.ndarray] = {}
    if os.path.isdir(GT_PROFILES_DIR):
        print(f"\nLoading ground truth from {GT_PROFILES_DIR}")
        for var in FIVE_P:
            csv_path = os.path.join(GT_PROFILES_DIR, f"ground_truth_Y_{var}.csv")
            if not os.path.exists(csv_path):
                print(f"  WARNING: GT CSV not found for {var}: {csv_path}")
                continue
            df_gt = pd.read_csv(csv_path)
            row_idx = _find_row_for_site(df_gt, lon=lon, lat=lat, atol=1e-4)
            prof = _extract_profile_from_gt(df_gt, var=var, row_idx=row_idx)
            gt_profiles[var] = prof[:10]
    else:
        print(
            f"WARNING: GT profiles directory not found at {GT_PROFILES_DIR}; "
            "plots will omit ground truth."
        )

    # Load restart profiles
    all_profiles: Dict[str, Dict[str, np.ndarray]] = {}
    print("\nLoading site restarts:")
    for name, path in restart_paths.items():
        print(f"  {name}: {path}")
        if not os.path.exists(path):
            print("    WARNING: file does not exist, skipping.")
            continue
        all_profiles[name] = _load_profiles_from_restart(path)

    # Pairwise numeric diffs
    print("\n=== Pairwise max |diff| between restarts (per 5P variable) ===")
    restart_names = list(all_profiles.keys())
    for var in FIVE_P:
        print(f"\nVariable: {var}")
        for i in range(len(restart_names)):
            for j in range(i + 1, len(restart_names)):
                n1, n2 = restart_names[i], restart_names[j]
                p1 = all_profiles[n1].get(var)
                p2 = all_profiles[n2].get(var)
                if p1 is None or p2 is None:
                    continue
                nlev = min(len(p1), len(p2))
                diff = float(np.nanmax(np.abs(p1[:nlev] - p2[:nlev])))
                print(f"  max |{n1} - {n2}| = {diff}")

    # Line plots
    print(f"\nSaving plots to: {output_dir}")
    for var in FIVE_P:
        plt.figure(figsize=(5, 6))
        has_any = False

        # GT first
        if var in gt_profiles:
            gt = np.asarray(gt_profiles[var], dtype=float)
            nlev_gt = min(10, len(gt))
            layers_gt = np.arange(1, nlev_gt + 1)
            plt.plot(
                layers_gt,
                gt[:nlev_gt],
                marker="o",
                linestyle="-",
                color="black",
                label="ground_truth",
                linewidth=2.0,
            )
            has_any = True

        # Restarts
        for name, profs in all_profiles.items():
            if var not in profs:
                continue
            prof = np.asarray(profs[var], dtype=float)
            nlev = min(10, len(prof))
            layers = np.arange(1, nlev + 1)
            plt.plot(
                layers,
                prof[:nlev],
                marker="o",
                linestyle="-",
                label=name,
                alpha=0.9,
            )
            has_any = True

        if not has_any:
            plt.close()
            continue

        plt.xlabel("Soil layer index")
        plt.ylabel(f"{var} value")
        plt.title(
            f"{var} vertical profile at {site_id}\n"
            f"(lon={lon}, lat={lat})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{site_id}_restart_profile_{var}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Given a lon/lat, extract single-point restarts for phase1_global, "
            "phase2_tropical, phase3_tworegions and generate 5P vertical-profile "
            "comparison plots vs ground truth."
        )
    )
    parser.add_argument("--lon", type=float, required=True, help="Site longitude (degrees, 0–360).")
    parser.add_argument("--lat", type=float, required=True, help="Site latitude (degrees).")
    parser.add_argument(
        "--site-name",
        type=str,
        default="site",
        help="Optional human-readable site name used in filenames and plot titles.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help=(
            "Output directory for plots. Default: "
            f"{PHASE3_RUN_DIR}/analysis/<site_id>_5p_restart_comparison"
        ),
    )
    args = parser.parse_args()

    lon = args.lon
    lat = args.lat
    site_name = args.site_name
    site_id = _safe_site_name(site_name, lon, lat)

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(
            PHASE3_RUN_DIR,
            "analysis",
            f"{site_id}_5p_restart_comparison",
        )

    print(f"Site: {site_id} (lon={lon}, lat={lat})")
    print(f"Output directory: {output_dir}")

    restart_paths = generate_site_restarts(lon=lon, lat=lat, site_name=site_name)
    compare_5p_for_site(
        lon=lon,
        lat=lat,
        site_name=site_name,
        restart_paths=restart_paths,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

