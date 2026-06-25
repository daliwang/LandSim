"""
Disk cache for preprocessed + normalized + split training tensors.

Skips pickle load, preprocess, normalize, and split on cache hits when data
files and pipeline settings are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config.training_config import DataConfig, PreprocessingConfig

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def list_data_files(data_config: DataConfig) -> List[Path]:
    """Resolve pickle/parquet files the same way as DataLoaderIndividual.load_data."""
    files: List[Path] = []
    per_dataset_patterns = getattr(data_config, "dataset_file_patterns", {}) or {}
    sort_files = getattr(data_config, "sort_file_list", True)

    for path in data_config.data_paths:
        path_normalized = str(Path(path).resolve())
        pattern = per_dataset_patterns.get(
            path_normalized,
            per_dataset_patterns.get(path, data_config.file_pattern),
        )
        path_files = list(Path(path).glob(pattern))
        if sort_files:
            path_files = sorted(path_files, key=lambda p: p.name)
        files.extend(path_files)

    if getattr(data_config, "max_files", None) is not None:
        files = files[: data_config.max_files]
    return files


def _file_manifest(files: List[Path]) -> List[Dict[str, Any]]:
    manifest = []
    for fp in files:
        try:
            st = fp.stat()
            manifest.append(
                {
                    "path": str(fp.resolve()),
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                }
            )
        except OSError as exc:
            manifest.append({"path": str(fp), "error": str(exc)})
    return manifest


def compute_cache_fingerprint(
    data_config: DataConfig,
    preprocessing_config: PreprocessingConfig,
    *,
    normalization_method: str,
    pft_presence_threshold: float = 0.0,
    files: Optional[List[Path]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return (hex digest, manifest dict) used for cache invalidation."""
    if files is None:
        files = list_data_files(data_config)

    payload = {
        "cache_version": CACHE_VERSION,
        "normalization_method": normalization_method,
        "pft_presence_threshold": float(pft_presence_threshold),
        "data_config": {
            "data_paths": [str(Path(p).resolve()) for p in data_config.data_paths],
            "file_pattern": data_config.file_pattern,
            "max_files": data_config.max_files,
            "train_split": data_config.train_split,
            "test_split": data_config.test_split,
            "random_state": data_config.random_state,
            "tropical_only": getattr(data_config, "tropical_only", False),
            "tropical_lat_range": getattr(data_config, "tropical_lat_range", None),
            "natveg_only": getattr(data_config, "natveg_only", False),
            "natveg_filter_before_split": getattr(
                data_config, "natveg_filter_before_split", True
            ),
            "longitudes_to_drop": list(getattr(data_config, "longitudes_to_drop", []) or []),
            "region_boxes": getattr(data_config, "region_boxes", None),
            "time_series_length": data_config.time_series_length,
            "time_series_columns": list(data_config.time_series_columns),
            "static_columns": list(data_config.static_columns),
            "pft_param_columns": list(data_config.pft_param_columns),
            "x_list_scalar_columns": list(data_config.x_list_scalar_columns),
            "y_list_scalar_columns": list(data_config.y_list_scalar_columns),
            "x_list_columns_1d": list(data_config.x_list_columns_1d),
            "y_list_columns_1d": list(data_config.y_list_columns_1d),
            "x_list_columns_2d": list(data_config.x_list_columns_2d),
            "y_list_columns_2d": list(data_config.y_list_columns_2d),
            "max_2d_rows": getattr(data_config, "max_2d_rows", None),
            "max_2d_cols": getattr(data_config, "max_2d_cols", None),
        },
        "preprocessing_config": {
            "data_type": str(getattr(preprocessing_config, "data_type", torch.float32)),
            "scalar_normalization": getattr(
                preprocessing_config, "scalar_normalization", None
            ),
            "list_1d_normalization": getattr(
                preprocessing_config, "list_1d_normalization", None
            ),
            "list_2d_normalization": getattr(
                preprocessing_config, "list_2d_normalization", None
            ),
        },
        "files": _file_manifest(files),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    ).hexdigest()
    return digest, payload


def cache_paths(cache_dir: Path, fingerprint: str) -> Dict[str, Path]:
    base = cache_dir / fingerprint
    return {
        "tensors": base.with_suffix(".pt"),
        "scalers": base.with_suffix(".scalers.pkl"),
        "meta": base.with_suffix(".meta.json"),
    }


def _tensor_dict_to_cpu(split: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in split.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().to(torch.float32)
        elif isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(value.astype(np.float32, copy=False))
        else:
            raise TypeError(f"Cannot cache split key '{key}' with type {type(value)}")
    return out


def _tensor_dict_from_cache(split: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in split.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            out[key] = value.to(torch.float32)
        else:
            out[key] = torch.tensor(value, dtype=torch.float32)
    return out


def cache_is_valid(paths: Dict[str, Path], fingerprint: str) -> bool:
    if not paths["tensors"].is_file() or not paths["scalers"].is_file():
        return False
    if not paths["meta"].is_file():
        return False
    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("fingerprint") == fingerprint and meta.get("cache_version") == CACHE_VERSION
    except Exception:
        return False


def save_preprocessed_cache(
    paths: Dict[str, Path],
    *,
    fingerprint: str,
    manifest: Dict[str, Any],
    split_data: Dict[str, Any],
    scalers: Dict[str, Any],
    data_info: Dict[str, Any],
) -> None:
    paths["tensors"].parent.mkdir(parents=True, exist_ok=True)
    tensor_payload = {
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "train": _tensor_dict_to_cpu(split_data["train"]),
        "test": _tensor_dict_to_cpu(split_data["test"]),
        "train_size": split_data.get("train_size"),
        "test_size": split_data.get("test_size"),
        "data_info": data_info,
    }
    tmp_tensors = paths["tensors"].with_suffix(".pt.tmp")
    tmp_scalers = paths["scalers"].with_suffix(".scalers.pkl.tmp")
    tmp_meta = paths["meta"].with_suffix(".meta.json.tmp")
    t0 = time.perf_counter()
    torch.save(tensor_payload, tmp_tensors)
    with open(tmp_scalers, "wb") as f:
        pickle.dump(scalers, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "created_unix": time.time(),
        "manifest": manifest,
        "train_samples": int(tensor_payload["train"]["time_series"].shape[0]),
        "test_samples": int(tensor_payload["test"]["time_series"].shape[0]),
    }
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    tmp_tensors.replace(paths["tensors"])
    tmp_scalers.replace(paths["scalers"])
    tmp_meta.replace(paths["meta"])
    elapsed = time.perf_counter() - t0
    logger.info(
        "Saved preprocessed cache to %s (%.1fs)",
        paths["tensors"].parent / fingerprint,
        elapsed,
    )


def load_preprocessed_cache(paths: Dict[str, Path]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        tensor_payload = torch.load(
            paths["tensors"], map_location="cpu", weights_only=False
        )
    except TypeError:
        tensor_payload = torch.load(paths["tensors"], map_location="cpu")
    with open(paths["scalers"], "rb") as f:
        scalers = pickle.load(f)
    train = _tensor_dict_from_cache(tensor_payload["train"])
    test = _tensor_dict_from_cache(tensor_payload["test"])
    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded preprocessed cache from %s in %.1fs (train=%s, test=%s samples)",
        paths["tensors"],
        elapsed,
        train["time_series"].shape[0],
        test["time_series"].shape[0],
    )
    return {
        "train": train,
        "test": test,
        "train_size": tensor_payload.get("train_size"),
        "test_size": tensor_payload.get("test_size"),
        "scalers": scalers,
        "data_info": tensor_payload.get("data_info", {}),
        "fingerprint": tensor_payload.get("fingerprint"),
    }


def resolve_cache_dir(
    cache_dir: Optional[str],
    data_config: DataConfig,
) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    if data_config.data_paths:
        return Path(data_config.data_paths[0]).resolve() / ".preprocessed_cache"
    return Path(".preprocessed_cache").resolve()


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
