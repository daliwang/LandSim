import argparse
import re
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


DEFAULT_SIM_DIR = (
    "/global/cfs/cdirs/m4814/daweigao/14_Code/0_dataset_construction/"
    "3_restarted simulation"
)
DEFAULT_GT_PATH = (
    "/global/cfs/cdirs/m4814/daweigao/14_Code/0_dataset_construction/"
    "20251201_TRENDY2024_default_ICB1850CNPRDCTCBC.elm.h0.0801-01-01-00000.nc"
)


def _to_nan_fillvalue(arr, fill_threshold=1e35):
    data = np.asarray(arr, dtype=float)
    data[np.abs(data) >= fill_threshold] = np.nan
    return data


def _apply_ijcai_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "legend.frameon": False,
    })


def _extract_year(path: Path):
    match = re.search(r"\.h0\.(\d{4})-", path.name)
    if not match:
        return None
    return int(match.group(1))


def _sum_variable(ds, variable):
    if variable not in ds:
        raise KeyError(f"Variable {variable} not found in {ds.encoding.get('source', 'dataset')}.")
    data = _to_nan_fillvalue(ds[variable].values)
    return float(np.nansum(data))


def _collect_simulation_series(sim_dir: Path, variable: str):
    nc_files = sorted(sim_dir.glob("*.h0.*.nc"))
    years = []
    sums = []
    for path in nc_files:
        year = _extract_year(path)
        if year is None:
            continue
        with xr.open_dataset(path) as ds:
            value = _sum_variable(ds, variable)
        years.append(year)
        sums.append(value)

    if not years:
        raise FileNotFoundError(f"No .h0.*.nc files with year found in {sim_dir}")

    order = np.argsort(years)
    years = np.asarray(years)[order]
    sums = np.asarray(sums)[order]
    return years, sums


def plot_timeseries(sim_dir, gt_path, variable="NEE", output=None):
    sim_dir = Path(sim_dir)
    gt_path = Path(gt_path)
    if output:
        output = Path(output)
    else:
        output = Path(__file__).resolve().parent / f"{variable.lower()}_timeseries_ijcai.png"

    years, sim_sums = _collect_simulation_series(sim_dir, variable)
    with xr.open_dataset(gt_path) as ds_gt:
        gt_sum = _sum_variable(ds_gt, variable)

    _apply_ijcai_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    ax.plot(years, sim_sums, color="#1f77b4", linewidth=2.0, label="Simulation")
    ax.hlines(gt_sum, years.min(), years.max(), colors="#d62728", linestyles="--", linewidth=2.0, label="Ground Truth")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"{variable} Sum")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output)
    plt.close(fig)

    print(f"Saved time series plot: {output}")
    print(f"Saved time series plot: {pdf_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot IJCAI-style time series of NEE sum.")
    parser.add_argument("--sim-dir", type=str, default=DEFAULT_SIM_DIR, help="Directory of simulation .h0.*.nc files.")
    parser.add_argument("--gt-path", type=str, default=DEFAULT_GT_PATH, help="Ground-truth NetCDF path.")
    parser.add_argument("--variable", type=str, default="NEE", help="Variable name to plot.")
    parser.add_argument("--output", type=str, default=None, help="Output figure path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_timeseries(
        sim_dir=args.sim_dir,
        gt_path=args.gt_path,
        variable=args.variable,
        output=args.output,
    )
