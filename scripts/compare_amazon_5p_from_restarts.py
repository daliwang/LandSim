import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# List of restart files to compare at the Amazon site (single-point restarts)
RESTARTS = {
    "natveg_amazon": (
        "cnp_results/run_20260228_214757_natveg_improved/"
        "Amazon_updated_natveg_improved_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024.nc"
    ),
    "phase2_tropical_4p": (
        "cnp_results/run_20260305_153217_phase2_pvariable_focus/"
        "Amazon_phase2_tropical_4p_restart.nc"
    ),
    "phase2_5P_bias_corrected": (
        "cnp_results/run_20260305_153217_phase2_pvariable_focus/"
        "Amazon_phase2_5P_bias_corrected_tropical_restart.nc"
    ),
    # Add more here if you like, e.g. finetuned runs:
    # "phase2_finetune_tworegions": (
    #     "cnp_results/finetune_20260308_232215_phase2_pvariab_focus_finetune_with_tworegions/"
    #     "Amazon_phase2_finetune_restart.nc"
    # ),
}

FIVE_P = ["labilep_vr", "occlp_vr", "solutionp_vr", "secondp_vr", "primp_vr"]
LEV_DIM = "levgrnd"  # adjust if different in your restarts
OUTPUT_DIR = (
    "cnp_results/run_20260305_153217_phase2_pvariable_focus/"
    "analysis/amazon_5p_restart_comparison"
)

# Ground truth profiles from previous Amazon 5P comparison
GT_PROFILES_CSV = (
    "cnp_results/run_20260228_214757_natveg_improved/"
    "analysis/amazon_5p_comparison_bias_correction/amazon_site_5p_profiles.csv"
)


def load_profiles(path: str):
    ds = xr.open_dataset(path, decode_times=False)
    profs = {}
    for var in FIVE_P:
        if var not in ds:
            print(f"  WARNING: {var} not found in {path}")
            continue
        da = ds[var]
        arr = da.values
        # Try to get a 1D vertical profile
        if arr.ndim == 1:
            profs[var] = arr
        elif arr.ndim == 2:
            if da.dims[0] == LEV_DIM:
                profs[var] = arr[:, 0]
            elif da.dims[1] == LEV_DIM:
                profs[var] = arr[0, :]
            else:
                profs[var] = arr.reshape(-1)
        else:
            profs[var] = arr.reshape(-1)
        # Always truncate to top 10 layers
        profs[var] = profs[var][:10]
    return profs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load ground truth profiles if available
    gt_profiles = {}
    if os.path.exists(GT_PROFILES_CSV):
        df_gt = pd.read_csv(GT_PROFILES_CSV)
        for var in FIVE_P:
            sub = df_gt[df_gt["variable"] == var].sort_values("layer_index")
            if sub.empty:
                continue
            # Top 10 layers only
            gt_profiles[var] = sub["gt"].to_numpy()[:10]
    else:
        print(f"WARNING: GT profiles CSV not found at {GT_PROFILES_CSV}; plots will skip GT.")

    all_profiles = {}
    for name, path in RESTARTS.items():
        print(f"\nLoading restart: {name} -> {path}")
        if not os.path.exists(path):
            print(f"  WARNING: file does not exist, skipping.")
            continue
        profs = load_profiles(path)
        all_profiles[name] = profs

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

    # Line plots (same x/y convention as earlier Amazon 5P plots: x = layer index, y = value)
    print(f"\nSaving plots to: {OUTPUT_DIR}")
    for var in FIVE_P:
        plt.figure(figsize=(5, 6))
        has_any = False

        # Plot ground truth first (if available)
        if var in gt_profiles:
            gt = np.array(gt_profiles[var], dtype=float)
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

        for name, profs in all_profiles.items():
            if var not in profs:
                continue
            prof = np.array(profs[var], dtype=float)
            nlev = min(10, len(prof))
            layers = np.arange(1, nlev + 1)
            plt.plot(
                layers,
                prof,
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
        plt.title(f"{var} vertical profile at Amazon site\n(restart comparison)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"amazon_restart_profile_{var}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()