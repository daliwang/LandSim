#!/usr/bin/env python3
"""
Fine-tuning driver for the CNP model.

This script performs model fine-tuning using a configuration-only workflow:
all runtime parameters (paths, hyperparameters, dataset descriptions) are
expected to originate from the `CNP_IO_updated9_dev_gao.txt` configuration
file supplied by the user. The implementation reuses the existing training
infrastructure (data loaders, trainer, model definition) to stay aligned with
the primary training entry point.

Usage:
    python scripts/run_finetuning.py [--config PATH] [--normalization MODE]

Key responsibilities:
    * Parse the configuration text file into structured overrides.
    * Load TVA datasets using the same preprocessing pipeline as training.
    * Restore a pretrained checkpoint and continue training.
    * Persist the fine-tuned model along with metrics and configuration
      metadata in an output directory specified by the configuration file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch

# Ensure the project root (LandSim) is on sys.path so internal modules resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.training_config import (
    TrainingConfigManager,
    get_cnp_combined_config,
    parse_cnp_io_list,
)
from data.data_loader_individual import DataLoaderIndividual
from models.cnp_combined_model import CNPCombinedModel
from scripts.run_inference_all import verify_locations
from training.trainer import ModelTrainer

# --------------------------------------------------------------------------- #
# Configuration parsing helpers
# --------------------------------------------------------------------------- #


def _coerce_scalar(value: str) -> Any:
    """Best-effort conversion of textual configuration values to Python types."""
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return ""

    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False

    # Numeric conversion (int preferred when possible).
    try:
        if "." not in text:
            return int(text)
        return float(text)
    except ValueError:
        pass

    # Comma-separated lists (e.g., multi-path values).
    if "," in text:
        parts = [segment.strip() for segment in text.split(",")]
        return [segment for segment in parts if segment]

    return text


def _parse_key_value_pairs(config_path: Path) -> Dict[str, Any]:
    """Parse simple key/value lines from the configuration text file."""
    results: Dict[str, Any] = {}
    with config_path.open("r") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            # Ignore bullet lists; they are handled by parse_cnp_io_list.
            if stripped.startswith("•"):
                continue
            # Support both ":" and "=" delimiters.
            delimiter = ":" if ":" in stripped else ("=" if "=" in stripped else None)
            if delimiter is None:
                continue
            key, raw_value = stripped.split(delimiter, 1)
            key = key.strip()
            value = _coerce_scalar(raw_value)
            if key:
                results[key.lower()] = value
    return results


def load_configuration(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration overrides from the text file.

    Returns
    -------
    tuple
        - combined (dict): Lowercase key dictionary combining structured
          groups returned by `parse_cnp_io_list` and simple key/value pairs.
        - parsed_groups (dict): Full output from `parse_cnp_io_list` containing
          variable lists and dataset metadata.
    """
    parsed_groups = parse_cnp_io_list(str(config_path))
    simple_pairs = _parse_key_value_pairs(config_path)

    combined: Dict[str, Any] = {}
    for key, value in parsed_groups.items():
        combined[key.lower()] = value
    combined.update(simple_pairs)

    return combined, parsed_groups


def require_config(
    config: Dict[str, Any],
    *keys: str,
    required: bool = True,
    default: Any = None,
) -> Any:
    """
    Retrieve the first available configuration value among provided keys.

    Parameters
    ----------
    config : dict
        Configuration dictionary with lowercase keys.
    keys : tuple[str]
        Key candidates to attempt, in priority order.
    required : bool
        Whether to raise an error when the keys are not found.
    default : Any
        Fallback value returned when `required=False` and nothing is present.
    """
    for key in keys:
        candidate = key.lower()
        if candidate in config:
            value = config[candidate]
            if value is None or value == "":
                continue
            return value
    if required:
        names = ", ".join(keys)
        raise KeyError(f"Missing required configuration entry. Tried keys: {names}")
    return default


def resolve_path(base_file: Path, raw_path: Union[str, Path]) -> Path:
    """Resolve paths relative to the configuration file directory."""
    if isinstance(raw_path, Path):
        candidate = raw_path
    else:
        expanded = os.path.expandvars(str(raw_path))
        candidate = Path(expanded)
    candidate = candidate.expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_file.parent / candidate).resolve()


def materialize_directory(path: Path) -> Path:
    """Ensure a directory exists and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def attach_file_logger(run_dir: Path) -> None:
    """Attach a file handler pointing to the run directory log file."""
    root_logger = logging.getLogger()
    log_path = Path(run_dir) / "finetune.log"
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(getattr(handler, "baseFilename", "")) == log_path:
                    return
            except TypeError:
                continue
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(root_logger.level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.info("Log file created at %s", log_path)


# --------------------------------------------------------------------------- #
# Configuration application helpers
# --------------------------------------------------------------------------- #


def apply_variable_lists(
    manager: TrainingConfigManager,
    parsed_groups: Dict[str, Any],
) -> None:
    """Populate the `DataConfig` lists from the parsed CNP IO specification."""
    data_cfg = manager.data_config

    # Optional overrides only trigger when a non-empty list is provided.
    def _set_if_present(attr: str, values: Iterable[str]) -> None:
        items = list(values) if values else []
        if items:
            setattr(data_cfg, attr, items)

    _set_if_present("time_series_columns", parsed_groups.get("time_series_variables"))
    _set_if_present("static_columns", parsed_groups.get("surface_properties"))
    _set_if_present("pft_param_columns", parsed_groups.get("pft_parameters"))
    _set_if_present("x_list_scalar_columns", parsed_groups.get("scalar_variables"))
    _set_if_present("x_list_columns_1d", parsed_groups.get("pft_1d_variables"))
    _set_if_present("x_list_columns_2d", parsed_groups.get("variables_2d_soil"))

    # Derive target variable lists when not explicitly supplied.
    if parsed_groups.get("scalar_variables"):
        data_cfg.y_list_scalar_columns = [
            f"Y_{name}" if not name.startswith("Y_") else name
            for name in parsed_groups["scalar_variables"]
        ]
    if parsed_groups.get("pft_1d_variables"):
        data_cfg.y_list_columns_1d = [
            f"Y_{name}" if not name.startswith("Y_") else name
            for name in parsed_groups["pft_1d_variables"]
        ]
    if parsed_groups.get("variables_2d_soil"):
        data_cfg.y_list_columns_2d = [
            f"Y_{name}" if not name.startswith("Y_") else name
            for name in parsed_groups["variables_2d_soil"]
        ]

    longitudes = parsed_groups.get("longitudes_to_drop") or []
    if longitudes:
        try:
            data_cfg.longitudes_to_drop = [float(val) for val in longitudes]
        except (TypeError, ValueError):
            logging.getLogger(__name__).warning(
                "Failed to parse longitude filtering list: %s", longitudes
            )


def apply_training_overrides(
    manager: TrainingConfigManager,
    config: Dict[str, Any],
    run_dir: Path,
    model_filename: str,
) -> None:
    """Update training configuration fields based on key/value overrides."""
    train_cfg = manager.training_config
    data_cfg = manager.data_config

    # Training hyperparameters (optional).
    for key in ("num_epochs", "batch_size", "learning_rate", "patience"):
        if key in config:
            try:
                value = float(config[key]) if key == "learning_rate" else int(config[key])
                setattr(train_cfg, key, value if key == "learning_rate" else int(value))
            except (TypeError, ValueError):
                logging.getLogger(__name__).warning(
                    "Skipping invalid value for %s: %r", key, config[key]
                )

    # Train/test split ratio.
    if "train_split" in config:
        try:
            data_cfg.train_split = float(config["train_split"])
        except (TypeError, ValueError):
            logging.getLogger(__name__).warning(
                "Invalid train_split override: %r", config["train_split"]
            )

    # Max files constraint for rapid experimentation.
    if "max_files" in config:
        try:
            data_cfg.max_files = int(config["max_files"])
        except (TypeError, ValueError):
            logging.getLogger(__name__).warning(
                "Invalid max_files override: %r", config["max_files"]
            )

    # Enable or disable early stopping if requested.
    if "use_early_stopping" in config:
        train_cfg.use_early_stopping = bool(config["use_early_stopping"])

    # Device selection honoring config preference when specified.
    if "device" in config:
        train_cfg.device = str(config["device"])
    else:
        train_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths for artifacts relative to run directory.
    train_cfg.model_save_path = str(run_dir / model_filename)
    train_cfg.losses_save_path = str(run_dir / "finetune_losses.csv")
    train_cfg.predictions_dir = str(run_dir / "cnp_predictions")
    train_cfg.save_model = True
    train_cfg.save_predictions = True

    # Disable GPU monitoring by default for fine-tuning runs unless explicitly enabled.
    train_cfg.log_gpu_memory = bool(config.get("log_gpu_memory", False))
    train_cfg.log_gpu_utilization = bool(config.get("log_gpu_utilization", False))

    # Optional masking configuration.
    if "mask_absent_pfts" in config:
        train_cfg.mask_absent_pfts = bool(config["mask_absent_pfts"])

    # Tropical-only filtering: restrict data to latitude band before train/test split.
    if "tropical_only" in config:
        data_cfg.tropical_only = bool(config["tropical_only"])
    if "tropical_lat_range" in config:
        raw = config["tropical_lat_range"]
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            data_cfg.tropical_lat_range = (float(raw[0]), float(raw[1]))
        elif isinstance(raw, str):
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) >= 2:
                data_cfg.tropical_lat_range = (float(parts[0]), float(parts[1]))


def load_training_config_from_checkpoint(model_path: Path) -> Optional[Dict[str, Any]]:
    """Discover and load `cnp_config.json` located near the checkpoint."""
    for directory in [model_path.parent] + list(model_path.parents):
        candidate = directory / "cnp_config.json"
        if candidate.exists():
            try:
                with candidate.open("r") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError) as err:
                logging.getLogger(__name__).warning(
                    "Failed to parse %s: %s", candidate, err
                )
            break
    return None


def apply_model_config_from_json(
    manager: TrainingConfigManager,
    config_json: Dict[str, Any],
) -> None:
    """Bring forward model architecture overrides stored in training JSON."""
    model_overrides = config_json.get("model_config")
    if not isinstance(model_overrides, dict):
        return
    model_cfg = manager.model_config
    for key, value in model_overrides.items():
        if hasattr(model_cfg, key):
            setattr(model_cfg, key, value)


def apply_data_info_overrides(
    manager: TrainingConfigManager,
    config_json: Dict[str, Any],
) -> None:
    """Sync data configuration lists using historical `data_info` metadata."""
    data_info = config_json.get("data_info")
    if not isinstance(data_info, dict):
        return

    data_cfg = manager.data_config

    mappings = {
        "time_series_columns": "time_series_columns",
        "static_columns": "static_columns",
        "pft_param_columns": "pft_param_columns",
        "x_list_scalar_columns": "x_list_scalar_columns",
        "y_list_scalar_columns": "y_list_scalar_columns",
        "variables_1d_pft": "x_list_columns_1d",
        "y_list_columns_1d": "y_list_columns_1d",
        "x_list_columns_2d": "x_list_columns_2d",
        "y_list_columns_2d": "y_list_columns_2d",
    }
    for source_key, target_attr in mappings.items():
        values = data_info.get(source_key)
        if isinstance(values, list) and values:
            setattr(data_cfg, target_attr, values)


# --------------------------------------------------------------------------- #
# Fine-tuning workflow
# --------------------------------------------------------------------------- #


def build_config_manager(
    args: argparse.Namespace,
    config: Dict[str, Any],
    parsed_groups: Dict[str, Any],
) -> Tuple[TrainingConfigManager, Path, Path, str, str]:
    """
    Prepare the configuration manager, dataset paths, and output locations.

    Returns
    -------
    tuple
        manager, dataset_path, model_path, file_pattern, model_filename
    """
    cfg_file = Path(args.config).resolve()
    dataset_path_raw = require_config(
        config,
        "fine_tuning_dataset_path",
        "tva_dataset_path",
        "tva4km_path",
        "data_paths",
    )
    if isinstance(dataset_path_raw, list):
        if not dataset_path_raw:
            raise ValueError("Dataset path list from configuration is empty.")
        dataset_path_raw = dataset_path_raw[0]
    dataset_path = resolve_path(cfg_file, dataset_path_raw)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    file_pattern_raw = require_config(
        config,
        "tva_dataset_pattern",
        "tva4km_file_pattern",
        "file_pattern",
    )
    if isinstance(file_pattern_raw, list):
        if not file_pattern_raw:
            raise ValueError("File pattern list from configuration is empty.")
        file_pattern_raw = file_pattern_raw[0]
    file_pattern = str(file_pattern_raw)
    model_path_raw = require_config(
        config,
        "pretrained_model_path",
        "model_default",
    )
    if isinstance(model_path_raw, list):
        if not model_path_raw:
            raise ValueError("Model path list from configuration is empty.")
        model_path_raw = model_path_raw[0]
    model_path = resolve_path(cfg_file, model_path_raw)
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model checkpoint not found: {model_path}")

    output_dir_raw = require_config(
        config,
        "output_finetuned_model_dir",
        "fine_tuning_output_dir",
        "finetune_output_dir",
    )
    output_root = materialize_directory(resolve_path(cfg_file, output_dir_raw))
    run_name = datetime.now().strftime("finetune_%Y%m%d_%H%M%S")
    run_directory = materialize_directory(output_root / run_name)

    model_filename = str(require_config(
        config,
        "finetuned_model_filename",
        "finetuned_checkpoint_name",
        required=False,
        default="finetuned_model.pth",
    ))

    # Build base configuration using the existing helper so that defaults
    # remain consistent with primary training runs.
    manager = get_cnp_combined_config(
        use_trendy1=False,
        use_trendy05=False,
        use_tva4km=True,
        max_files=None,
        include_water=False,
        variable_list_path=str(cfg_file),
    )

    # Override dataset paths/patterns explicitly.
    manager.update_data_config(
        data_paths=[str(dataset_path)],
        file_pattern=str(file_pattern),
        dataset_file_patterns={str(dataset_path): str(file_pattern)},
    )

    # Apply variable groups from configuration file.
    apply_variable_lists(manager, parsed_groups)

    # Attach artifact locations and optional hyperparameter overrides.
    apply_training_overrides(manager, config, run_directory, model_filename)

    return manager, dataset_path, model_path, run_directory, model_filename


def configure_normalization_mode(
    data_loader: DataLoaderIndividual,
    config: Dict[str, Any],
    explicit_mode: Optional[str],
) -> Tuple[Dict[str, Any], str]:
    """Select normalization routine based on CLI override or configuration."""
    mode = (
        explicit_mode
        or str(config.get("normalization_mode", "individual")).lower()
    )
    logger = logging.getLogger(__name__)
    logger.info("Normalization mode selected: %s", mode)

    if mode == "group":
        return data_loader.normalize_data(), "group"
    if mode == "hybrid":
        # Default hybrid settings: mimic training script behaviour when unspecified.
        return (
            data_loader.normalize_data_hybrid(
                use_individual_for=[
                    "scalar",
                    "y_scalar",
                    "pft_1d",
                    "y_pft_1d",
                    "soil_2d",
                    "y_soil_2d",
                ],
                group_soil_vars=["sminn_vr", "smin_no3_vr", "smin_nh4_vr"],
            ),
            "hybrid",
        )
    # Fallback to individual normalization.
    return data_loader.normalize_data_individual(), "individual"


def load_pretrained_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load weights from the provided checkpoint path into the model."""
    logger = logging.getLogger(__name__)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
    model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded pretrained weights from %s", checkpoint_path)


def fine_tune(args: argparse.Namespace) -> Dict[str, Any]:
    """Entry point driving the fine-tuning pipeline."""
    config_map, parsed_groups = load_configuration(Path(args.config))
    manager, dataset_path, model_path, run_dir, model_filename = build_config_manager(
        args,
        config_map,
        parsed_groups,
    )

    logger = logging.getLogger(__name__)
    logger.info("Dataset directory: %s", dataset_path)
    logger.info("Checkpoint to fine-tune: %s", model_path)
    logger.info("Run directory: %s", run_dir)
    attach_file_logger(run_dir)

    # CLI overrides for tropical-only filtering (override config file if set).
    if getattr(args, "tropical_only", False):
        manager.data_config.tropical_only = True
        if getattr(args, "tropical_lat_range", None):
            parts = [p.strip() for p in args.tropical_lat_range.split(",")]
            if len(parts) >= 2:
                manager.data_config.tropical_lat_range = (float(parts[0]), float(parts[1]))
        logger.info(
            "Tropical-only fine-tuning: lat range %s",
            manager.data_config.tropical_lat_range,
        )

    # Load historical configuration from the checkpoint when available.
    prior_config = load_training_config_from_checkpoint(model_path)
    if prior_config:
        apply_model_config_from_json(manager, prior_config)
        apply_data_info_overrides(manager, prior_config)

    # Set up the data loader using the populated configuration.
    data_loader = DataLoaderIndividual(
        manager.data_config,
        manager.preprocessing_config,
    )

    raw_df = data_loader.load_data()
    if hasattr(raw_df, "head"):
        verify_locations(raw_df, "Loaded dataset")

    data_loader.preprocess_data()
    normalized, normalization_mode = configure_normalization_mode(
        data_loader,
        config_map,
        args.normalization,
    )

    split = data_loader.split_data(normalized)
    data_info = data_loader.get_data_info()

    # Prepare model and trainer.
    include_water = bool(config_map.get("include_water", False))
    model = CNPCombinedModel(
        manager.model_config,
        data_info,
        include_water=include_water,
        use_learnable_loss_weights=manager.training_config.use_learnable_loss_weights,
    )
    load_pretrained_weights(model, model_path)

    predictions_dir = Path(manager.training_config.predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    scalers = normalized.get("scalers")
    if scalers is None:
        raise KeyError("Normalized data did not include scalers; cannot proceed.")

    trainer = ModelTrainer(
        manager.training_config,
        model,
        split["train"],
        split["test"],
        scalers,
        data_info,
    )

    results = trainer.run_training_pipeline()

    # Persist metrics and configuration summary alongside the checkpoint.
    metrics = results.get("metrics", {})
    (Path(manager.training_config.losses_save_path).parent).mkdir(parents=True, exist_ok=True)
    with (Path(run_dir) / "cnp_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    config_payload = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_path),
        "normalization_mode": normalization_mode,
        "include_water": include_water,
        "source_checkpoint": str(model_path),
        "model_save_path": manager.training_config.model_save_path,
        "data_info": data_info,
        "model_config": vars(manager.model_config),
        "training_config": vars(manager.training_config),
    }
    with (Path(run_dir) / "cnp_config.json").open("w") as handle:
        json.dump(config_payload, handle, indent=2)

    logger.info("Fine-tuning completed successfully. Artifacts saved to %s", run_dir)

    return {
        "metrics": metrics,
        "run_directory": str(run_dir),
        "model_path": manager.training_config.model_save_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_config = PROJECT_ROOT / "CNP_IO_updated9_dev_gao.txt"
    parser = argparse.ArgumentParser(
        description="Fine-tune the CNP model using TVA datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to the configuration text file.",
    )
    parser.add_argument(
        "--normalization",
        choices=("group", "individual", "hybrid"),
        help="Override normalization strategy (otherwise use config setting).",
    )
    parser.add_argument(
        "--tropical-only",
        action="store_true",
        help="Restrict fine-tuning data to tropical latitude band (overrides config file).",
    )
    parser.add_argument(
        "--tropical-lat-range",
        type=str,
        metavar="MIN,MAX",
        default=None,
        help='Latitude range for tropical filter, e.g. "-30,30" (default: -23.5,23.5). Used when --tropical-only is set.',
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for the run.",
    )
    return parser.parse_args()


def configure_root_logger(level: str) -> None:
    """Configure root logger according to requested verbosity."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    configure_root_logger(args.log_level)
    try:
        results = fine_tune(args)
        logging.getLogger(__name__).info("Fine-tune metrics: %s", results["metrics"])
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Fine-tuning failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
