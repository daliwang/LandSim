#!/usr/bin/env python3
"""
Geospatial workflow orchestrator for TVA locations.

This script reads latitude/longitude points from a CSV file, extracts the
corresponding samples from the TVA training dataset, runs the trained CNP model
to generate AI predictions, converts the predictions to NetCDF, and updates the
restart NetCDF file with the new values. All outputs are written to per-location
directories so additional locations can be added to the CSV and reprocessed
without modifying the script.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr

# Ensure the project root (LandSim) is on sys.path when running from TVA_1_Sample/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - fail fast for missing dependency
    raise SystemExit("pandas is required to run this workflow. Please install it before proceeding.") from exc

from config.training_config import parse_cnp_io_list
from scripts.run_inference_all import run_inference_all
from scripts.ai_predictions_to_netcdf import (
    load_ai_predictions,
    create_netcdf_structure,
    add_scalar_variables,
    add_pft_variables,
    add_soil_variables,
)
from scripts.ai_predictions_to_restart import (
    load_datasets,
    create_spatial_mapping,
    create_updated_restart_file,
    auto_detect_variable_list,
)


LOGGER = logging.getLogger("landsim.workflow")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_locations = script_dir / "locations.csv"
    default_variable_list = project_root / "CNP_IO_updated9_dev.txt"
    default_model_config = project_root / "CNP_model_config_v01.txt"
    default_output_root = project_root / "final_restartfile"
    default_work_root = script_dir / "workflow_runs"

    parser = argparse.ArgumentParser(
        description="Run TVA geospatial extraction, model inference, and restart updating workflow."
    )
    parser.add_argument(
        "--locations",
        type=Path,
        default=default_locations,
        help=f"CSV file containing latitude/longitude entries (default: {default_locations})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/global/cfs/cdirs/m4814/daweigao/14_Code/TVA_training_dataset_all"),
        help="Directory containing TVA training dataset pickle batches.",
    )
    parser.add_argument(
        "--dataset-pattern",
        default="enhanced_monthly_training_data_batch_*.pkl",
        help="Glob pattern used to discover TVA dataset batches.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "/global/cfs/cdirs/m4814/daweigao/15_code_Landsim/LandSim/"
            "cnp_results/run_20251030_192921/cnp_predictions/model.pth"
        ),
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--variable-list",
        type=Path,
        default=default_variable_list,
        help="Path to the CNP IO configuration used for training (CNP_IO_*.txt).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=default_model_config,
        help="Optional model config override used during inference (CNP_model_config_*.txt).",
    )
    parser.add_argument(
        "--restart-file",
        type=Path,
        default=Path(
            "/global/cfs/cdirs/m4814/daweigao/14_Code/TVA_restart/"
            "uELM_knox_I1850CNPRDCTCBC.elm.r.0021-01-01-00000.nc"
        ),
        help="Original restart NetCDF file that will be updated with AI predictions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Directory where updated restart files will be written.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=default_work_root,
        help="Directory for intermediate per-location artifacts (datasets, predictions, NetCDF files).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of TVA dataset batches to scan (useful for testing).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing a location if the final restart file already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for the workflow run.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def slugify(name: str) -> str:
    """Convert a location label into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    slug = slug.strip("_")
    if slug:
        return slug
    # Provide deterministic fallback when location name is missing
    return f"location_{abs(hash(name)) % 10_000}"


def read_locations(csv_path: Path) -> pd.DataFrame:
    """Load and validate the locations CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Locations CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"latitude", "longitude"}
    missing = required - set(c.lower() for c in df.columns)
    if missing:
        raise ValueError(f"Locations CSV must contain columns: {', '.join(sorted(required))}")
    # Normalize column names to lower-case for convenience
    df.columns = [c.lower() for c in df.columns]
    return df


def determine_variable_map(variable_list_path: Optional[Path]) -> Dict[str, List[str]]:
    """Parse the CNP IO variable list into groups needed for NetCDF conversion."""
    if variable_list_path is None:
        return {"scalar": [], "pft1d": [], "soil2d": []}
    if not variable_list_path.exists():
        raise FileNotFoundError(f"Variable list file not found: {variable_list_path}")

    parsed = parse_cnp_io_list(variable_list_path)
    return {
        "scalar": list(parsed.get("scalar_variables", [])),
        "pft1d": list(parsed.get("pft_1d_variables", [])),
        "soil2d": list(parsed.get("variables_2d_soil", parsed.get("x_list_columns_2d", []))),
    }


def extract_location_dataset(
    dataset_root: Path,
    file_pattern: str,
    latitude: float,
    longitude: float,
    output_path: Path,
    max_files: Optional[int] = None,
) -> Optional[Path]:
    """
    Build a location-specific dataset pickle by scanning TVA batches and
    collecting the rows whose coordinates are closest to the requested latitude/longitude.
    """
    files = sorted(dataset_root.glob(file_pattern))
    if max_files is not None:
        files = files[:max_files]

    best_frames: List[pd.DataFrame] = []
    LOGGER.debug("Scanning %d TVA batches for samples near (lat=%.6f, lon=%.6f)", len(files), latitude, longitude)

    lon_candidates = {float(longitude)}
    # Include 0-360 representation when the requested longitude is negative
    lon_candidates.add((longitude + 360.0) % 360.0)
    # Include -180–180 representation in case dataset already stores positive values over 180
    lon_candidates.add(((longitude + 180.0) % 360.0) - 180.0)

    def _minimum_lon_difference(series: pd.Series, targets: Iterable[float]) -> pd.Series:
        diffs = []
        for target in targets:
            diffs.append((series - target).abs())
        if not diffs:
            return pd.Series(np.nan, index=series.index)
        stacked = pd.concat(diffs, axis=1)
        return stacked.min(axis=1, skipna=True)

    closest_example: Optional[Tuple[str, float, float, float, float]] = None
    min_distance = math.inf

    for file_path in files:
        try:
            df_batch = pd.read_pickle(file_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to load batch %s: %s", file_path, exc)
            continue

        if "Latitude" not in df_batch.columns or "Longitude" not in df_batch.columns:
            continue

        # Ensure numeric types to avoid comparison surprises (NaNs are preserved)
        lat_series = pd.to_numeric(df_batch["Latitude"], errors="coerce")
        lon_series = pd.to_numeric(df_batch["Longitude"], errors="coerce")

        lat_diff_series = (lat_series - latitude).abs()
        lon_diff_series = _minimum_lon_difference(lon_series, lon_candidates)

        combined_diff = np.sqrt(np.square(lat_diff_series) + np.square(lon_diff_series))

        valid_diff = combined_diff.dropna()
        if not valid_diff.empty:
            idx = valid_diff.idxmin()
            distance = float(valid_diff.loc[idx])
            if distance < min_distance - 1e-12:
                min_distance = distance
                closest_example = (
                    file_path.name,
                    float(lat_series.loc[idx]),
                    float(lon_series.loc[idx]),
                    float(lat_diff_series.loc[idx]),
                    float(lon_diff_series.loc[idx]),
                )
                best_frames = [df_batch.loc[[idx]].copy()]
            elif abs(distance - min_distance) <= 1e-12:
                best_frames.append(df_batch.loc[[idx]].copy())

    if not best_frames:
        LOGGER.warning("No samples found in TVA dataset for latitude %.6f and longitude %.6f", latitude, longitude)
        return None

    combined = pd.concat(best_frames, ignore_index=True).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(output_path)

    if closest_example:
        LOGGER.info(
            "Nearest sample sourced from %s at (lat=%.6f, lon=%.6f) [Δlat=%.3e, Δlon=%.3e]",
            *closest_example,
        )
    LOGGER.info("Saved %d nearest samples (min distance %.3e) for location to %s", combined.shape[0], min_distance, output_path)
    return output_path


def run_model_inference(
    model_path: Path,
    dataset_file: Path,
    output_dir: Path,
    variable_list: Optional[Path],
    model_config: Optional[Path],
) -> Path:
    """Invoke the shared inference routine on a prepared dataset file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running inference for dataset %s", dataset_file)
    results_dir = run_inference_all(
        model_path=str(model_path),
        data_paths=str(dataset_file.parent),
        file_pattern=dataset_file.name,
        output_dir=str(output_dir),
        variable_list=str(variable_list) if variable_list else None,
        model_config=str(model_config) if model_config else None,
        scalers_dir=None,
        use_training_config=True,
        strict_loading=True,
        debug_vars=False,
        loader="auto",
        mask_pft_with_gt=False,
    )
    LOGGER.info("Inference outputs available in %s", results_dir)
    return results_dir


def convert_predictions_to_netcdf(
    predictions_dir: Path,
    variable_map: Dict[str, List[str]],
    output_path: Path,
) -> Path:
    """Convert inference CSV outputs into a NetCDF file for restart updates."""
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    LOGGER.info("Converting predictions in %s to NetCDF", predictions_dir)
    ai_preds = load_ai_predictions(predictions_dir)
    if "test_static_inverse" not in ai_preds:
        raise RuntimeError(
            "Prediction outputs are missing test_static_inverse.csv; cannot build NetCDF coordinates."
        )

    ds = create_netcdf_structure(ai_preds, variable_map, output_path)
    add_scalar_variables(ds, ai_preds, variable_map)
    add_pft_variables(ds, ai_preds, variable_map)
    add_soil_variables(ds, ai_preds, variable_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    LOGGER.info("NetCDF predictions written to %s", output_path)
    return output_path


def collect_cnp_variables(variable_map: Dict[str, List[str]]) -> List[str]:
    """Gather the list of variables that should be updated in the restart file."""
    soil_keys = variable_map.get("soil2d", [])
    if not soil_keys:
        # When list is empty try to auto-detect to maintain compatibility with restart script.
        LOGGER.debug("No soil variables provided; relying on auto-detection during restart update.")
    return sorted(set(variable_map.get("pft1d", []) + soil_keys))


def update_restart_file(
    restart_file: Path,
    ai_predictions_nc: Path,
    output_restart: Path,
    cnp_variables: List[str],
) -> Path:
    """Use the shared restart updater to write predictions into a new NetCDF file."""
    LOGGER.info("Updating restart file %s with predictions from %s", restart_file, ai_predictions_nc)
    ds_ai, ds_model = load_datasets(ai_predictions_nc, restart_file)
    try:
        model_to_ai_mapping, variable_mapping = create_spatial_mapping(ds_ai, ds_model)
    finally:
        ds_ai.close()
        ds_model.close()

    if not cnp_variables:
        cnp_variables = auto_detect_variable_list(ai_predictions_nc)

    output_restart.parent.mkdir(parents=True, exist_ok=True)
    create_updated_restart_file(
        restart_file_path=restart_file,
        output_path=output_restart,
        ai_predictions_path=ai_predictions_nc,
        cnp_io_variables=cnp_variables,
        model_to_ai_mapping=model_to_ai_mapping,
        variable_mapping=variable_mapping,
    )
    LOGGER.info("Updated restart file saved to %s", output_restart)
    return output_restart


def process_location(
    location_row: pd.Series,
    args: argparse.Namespace,
    variable_map: Dict[str, List[str]],
    cnp_variables: List[str],
) -> Optional[Path]:
    """Execute the full workflow for a single row in the locations CSV."""
    latitude = float(location_row["latitude"])
    longitude = float(location_row["longitude"])
    location_label = f"{latitude:.4f}_{longitude:.4f}"
    slug = slugify(location_label)
    LOGGER.info("Processing location (lat=%.6f, lon=%.6f)", latitude, longitude)

    location_work_root = args.work_root / slug
    dataset_file = location_work_root / f"{slug}_dataset.pkl"
    predictions_root = location_work_root / "inference"
    predictions_dir = predictions_root / "cnp_predictions"
    predictions_nc = location_work_root / f"{slug}_ai_predictions.nc"
    restart_output = args.output_root / f"{args.restart_file.stem}_{slug}.nc"

    if args.skip_existing and restart_output.exists():
        LOGGER.info("Skipping location %s because %s already exists", location_label, restart_output)
        return restart_output

    dataset_path = extract_location_dataset(
        dataset_root=args.dataset_root,
        file_pattern=args.dataset_pattern,
        latitude=latitude,
        longitude=longitude,
        output_path=dataset_file,
        max_files=args.max_files,
    )
    if dataset_path is None:
        return None

    try:
        df_preview = pd.read_pickle(dataset_path)
        LOGGER.info(
            "  [Stage 1] Extracted %d rows (sample columns: %s)",
            len(df_preview),
            ", ".join(map(str, df_preview.columns[:5])),
        )
    except Exception as exc:
        LOGGER.warning("  [Stage 1] Unable to validate dataset %s: %s", dataset_path, exc)

    results_dir = run_model_inference(
        model_path=args.model_path,
        dataset_file=dataset_path,
        output_dir=predictions_root,
        variable_list=args.variable_list,
        model_config=args.model_config,
    )

    predictions_dir = results_dir / "cnp_predictions"
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory missing after inference: {predictions_dir}")

    csv_count = sum(1 for _ in predictions_dir.rglob("*.csv"))
    LOGGER.info(
        "  [Stage 2] Inference artifacts ready at %s (CSV files: %d)",
        predictions_dir,
        csv_count,
    )

    predictions_nc = convert_predictions_to_netcdf(predictions_dir, variable_map, predictions_nc)

    try:
        ds_preview = xr.open_dataset(predictions_nc)
        dims_str = ", ".join(f"{k}={v}" for k, v in ds_preview.sizes.items())
        LOGGER.info(
            "  [Stage 3] NetCDF generated with dims [%s] and %d variables",
            dims_str,
            len(ds_preview.data_vars),
        )
    except Exception as exc:
        LOGGER.warning("  [Stage 3] Could not inspect NetCDF %s: %s", predictions_nc, exc)
    finally:
        try:
            ds_preview.close()
        except Exception:
            pass

    updated_restart = update_restart_file(
        restart_file=args.restart_file,
        ai_predictions_nc=predictions_nc,
        output_restart=restart_output,
        cnp_variables=cnp_variables,
    )
    if updated_restart.exists():
        size_mb = updated_restart.stat().st_size / (1024 * 1024)
        LOGGER.info("  [Stage 4] Restart file created at %s (%.2f MB)", updated_restart, size_mb)
    else:
        LOGGER.warning("  [Stage 4] Restart file expected but not found: %s", updated_restart)
    return updated_restart


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    LOGGER.info("Starting TVA workflow with locations file: %s", args.locations)
    variable_map = determine_variable_map(args.variable_list)
    cnp_variables = collect_cnp_variables(variable_map)
    LOGGER.debug("Variable groups loaded: %s", variable_map)

    locations_df = read_locations(args.locations)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[str, Optional[Path]]] = []
    for _, row in locations_df.iterrows():
        try:
            updated_restart = process_location(row, args, variable_map, cnp_variables)
            coord_label = f"{row['latitude']:.4f}_{row['longitude']:.4f}"
            results.append((coord_label, updated_restart))
        except Exception as exc:  # pragma: no cover - logging safety
            LOGGER.exception("Failed to process location row %s: %s", row.to_dict(), exc)
            coord_label = f"{row['latitude']:.4f}_{row['longitude']:.4f}"
            results.append((coord_label, None))

    LOGGER.info("Workflow complete. Summary of generated restart files:")
    for label, path in results:
        name = label or "<unnamed>"
        if path is None:
            LOGGER.warning("  %s: failed (no restart generated)", name)
        else:
            LOGGER.info("  %s: %s", name, path)


if __name__ == "__main__":
    main()
