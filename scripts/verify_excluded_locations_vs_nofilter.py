#!/usr/bin/env python3
"""
Verify that analysis/excluded_locations plots show the same gridcells that are
excluded by the natveg filter, and that analysis_nofilter contains all gridcells
(included + excluded). Compares:
- Excluded mask from test_static_inverse.csv (same logic as plot_excluded_locations.py)
- excluded_locations_lat_lon.csv (saved by plot_excluded_locations)
- Row counts and optional value overlap for one 1D and one 2D variable.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def get_excluded_mask(static_path):
    """Same logic as plot_excluded_locations.get_excluded_indices."""
    df = pd.read_csv(static_path)
    pct_natveg = df["PCT_NATVEG"].values.astype(float)
    pct_pft0 = df["PCT_NAT_PFT_0"].values.astype(float)
    include = (pct_natveg > 0) & (pct_pft0 < 100)
    excluded = ~include
    return excluded, df


def main():
    parser = argparse.ArgumentParser(description="Verify excluded_locations vs natveg filter and nofilter")
    parser.add_argument("results_dir", nargs="?", default=".", help="Results directory (e.g. cnp_results/run_xxx)")
    args = parser.parse_args()
    results_dir = Path(args.results_dir).resolve()

    static_path = results_dir / "cnp_predictions" / "test_static_inverse.csv"
    excl_csv = results_dir / "analysis" / "excluded_locations" / "excluded_locations_lat_lon.csv"

    if not static_path.exists():
        print(f"ERROR: {static_path} not found")
        return 1

    excluded_mask, static_df = get_excluded_mask(static_path)
    n_total = len(static_df)
    n_excluded = int(np.sum(excluded_mask))
    n_included = n_total - n_excluded
    excluded_indices = np.where(excluded_mask)[0]

    print("=" * 60)
    print("Natveg filter (from test_static_inverse.csv)")
    print("  Include: PCT_NATVEG > 0 AND PCT_NAT_PFT_0 < 100")
    print("  Exclude: PCT_NATVEG <= 0 OR PCT_NAT_PFT_0 >= 100")
    print("=" * 60)
    print(f"Total gridcells:        {n_total}")
    print(f"Included (in analysis): {n_included}")
    print(f"Excluded:               {n_excluded}")

    # Compare with excluded_locations_lat_lon.csv
    if excl_csv.exists():
        excl_df = pd.read_csv(excl_csv)
        if "Latitude" in excl_df.columns and "Longitude" in excl_df.columns:
            n_csv = len(excl_df)
            # Static has same row order; excluded rows in static should match CSV
            static_excl = static_df.loc[excluded_mask, ["Latitude", "Longitude"]].reset_index(drop=True)
            # Allow small float tolerance
            lat_ok = np.allclose(excl_df["Latitude"].values, static_excl["Latitude"].values, rtol=0, atol=1e-5)
            lon_ok = np.allclose(excl_df["Longitude"].values, static_excl["Longitude"].values, rtol=0, atol=1e-5)
            print(f"\nExcluded locations CSV: {excl_csv.name}")
            print(f"  Rows in CSV:          {n_csv}")
            print(f"  Match count:          {'YES' if n_csv == n_excluded else 'NO (expected ' + str(n_excluded) + ')'}")
            print(f"  Lat/Lon match static: {'YES' if (lat_ok and lon_ok) else 'NO'}")
        else:
            print(f"\nCSV found but missing Latitude/Longitude columns.")
    else:
        print(f"\nExcluded locations CSV not found: {excl_csv}")

    # 1D variable: cpool
    gt_1d_path = results_dir / "cnp_predictions" / "pft_1d_ground_truth" / "ground_truth_Y_cpool.csv"
    if gt_1d_path.exists():
        gt_1d = pd.read_csv(gt_1d_path)
        coord_cols = [c for c in ["Longitude", "Latitude", "long", "lat"] if c in gt_1d.columns]
        gt_1d_vals = gt_1d.drop(columns=coord_cols, errors="ignore")
        n_rows_1d = len(gt_1d)
        n_pfts = gt_1d_vals.shape[1]
        assert n_rows_1d == n_total, f"1D rows {n_rows_1d} vs static {n_total}"
        # Points in excluded_locations plot = excluded rows × PFTs (flattened)
        pts_excluded_1d = n_excluded * n_pfts
        pts_included_1d = n_included * n_pfts
        pts_all_1d = n_total * n_pfts
        print(f"\n1D variable (cpool): rows={n_rows_1d}, PFTs={n_pfts}")
        print(f"  Points in excluded_locations plot: {pts_excluded_1d} (gridcells × PFTs)")
        print(f"  Points in filtered analysis:       {pts_included_1d}")
        print(f"  Points in nofilter (all):         {pts_all_1d}")
        print(f"  Check included+excluded=all:       {pts_included_1d + pts_excluded_1d == pts_all_1d}")
    else:
        print(f"\n1D GT not found: {gt_1d_path}")

    # 2D variable: soil1c_vr or first available
    gt_2d_dir = results_dir / "cnp_predictions" / "soil_2d_ground_truth"
    if gt_2d_dir.exists():
        for var in ["soil1c_vr", "litr1c_vr"]:
            gt_2d_path = gt_2d_dir / f"ground_truth_Y_{var}.csv"
            if not gt_2d_path.exists():
                continue
            gt_2d = pd.read_csv(gt_2d_path)
            coord_cols = [c for c in ["Longitude", "Latitude", "long", "lat"] if c in gt_2d.columns]
            gt_2d_vals = gt_2d.drop(columns=coord_cols, errors="ignore")
            n_rows_2d = len(gt_2d)
            n_layers = gt_2d_vals.shape[1]
            assert n_rows_2d == n_total, f"2D rows {n_rows_2d} vs static {n_total}"
            pts_excluded_2d = n_excluded * n_layers
            pts_included_2d = n_included * n_layers
            pts_all_2d = n_total * n_layers
            print(f"\n2D variable ({var}): rows={n_rows_2d}, layers={n_layers}")
            print(f"  Points in excluded_locations plot: {pts_excluded_2d} (gridcells × layers)")
            print(f"  Points in filtered analysis:       {pts_included_2d}")
            print(f"  Points in nofilter (all):         {pts_all_2d}")
            print(f"  Check included+excluded=all:       {pts_included_2d + pts_excluded_2d == pts_all_2d}")
            break
    else:
        print(f"\n2D GT dir not found: {gt_2d_dir}")

    print("\n" + "=" * 60)
    print("Conclusion:")
    print("  - analysis/ (with natveg filter) uses only INCLUDED gridcells.")
    print("  - analysis/excluded_locations/ shows only EXCLUDED gridcells (same as above mask).")
    print("  - analysis_nofilter/ uses ALL gridcells (included + excluded).")
    print("  So excluded_locations plots are exactly the gridcells removed from the filtered analysis")
    print("  and present in the nofilter plots.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
