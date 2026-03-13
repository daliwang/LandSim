# Unified Training Configuration

This document explains how to use a single JSON file to configure all training parameters.

## Overview

Instead of managing multiple separate JSON files, you can now use a single unified configuration file that contains all user-defined settings:

- Variable weights (PFT1D, Soil2D, Scalar)
- Tail-aware weights
- PFT zero sparsity weights
- PFT1D activation overrides
- **Data filtering** (tropical band, natveg-only) — see `data_filtering_config` below

### Where to put options: config JSON vs CNP_IO.txt

| Purpose | Use |
|--------|-----|
| **Variable lists**, **data paths**, **file patterns** | **CNP_IO.txt** (and optional paths in variable list) |
| **Training filters** (tropical, natveg, longitude drop), **hyperparameters**, **loss weights**, **reproducibility** | **config/training_*.json** (unified or per-experiment) |

Put **natveg filter**, **tropical selection**, and **longitude filtering** in the **training config JSON** under `data_filtering_config`, not in CNP_IO. That way one variable list can be reused with different filter choices, and the run’s saved config records exactly which filters were applied.

## JSON File Format

Create a single JSON file (`training_config_unified.json`) with the following structure:

```json
{
  "training_hyperparameters": {
    "num_epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "optimizer_type": "adam",
    "weight_decay": 0.0,
    "use_scheduler": false,
    "scheduler_type": "step",
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.1,
    "xsmrpool_loss_weight": 10.0,
    "litter_c_loss_weight": 1.0,
    "litter_n_loss_weight": 1.0,
    "litter_p_loss_weight": 1.0,
    "scalar_loss_weight": 1.0,
    "vector_loss_weight": 1.0,
    "matrix_loss_weight": 1.0
  },
  "variable_weights": {
    "pft1d_weights": {
      "cpool": 2.0,
      "npool": 2.0,
      "tlai": 3.0,
      "litr2p_vr": 5.0
    },
    "soil2d_weights": {
      "primp_vr": 3.0,
      "litr2p_vr": 5.0,
      "litr2n_vr": 5.0
    },
    "scalar_weights": {
      "GPP": 1.5,
      "NPP": 1.5
    }
  },
  "tail_aware_config": {
    "loss": "log1p_mse",
    "epsilon": 1e-8
  },
  "tail_aware_weights": {
    "cpool": 5.0,
    "deadstemc": 5.0,
    "litr2p_vr": 5.0,
    "litr2n": 5.0
  },
  "pft_mask_config": {
    "mask_absent_pfts": true,
    "pft_presence_threshold": 0.0
  },
  "pft_zero_sparsity_config": {
    "weight": 1.0,
    "threshold": 1e-8
  },
  "pft_zero_sparsity_weights": {
    "cpool": 1.0,
    "deadstemc": 1.0
  },
  "pft1d_activation_overrides": {
    "cpool": "abs",
    "deadstemc": "abs"
  }
}
```

**Notes:**
- All sections are optional - only include what you need
- Individual JSON files take precedence if both are specified
- You can mix unified config with individual files (unified as base, individual for overrides)

## Usage

### Single Unified File (Recommended)

```bash
python train_cnp_model.py \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --training-config-json config/training_config_unified.json \
  --epoch 100
```

### Mix Unified + Individual Files (for overrides)

```bash
python train_cnp_model.py \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --training-config-json config/training_config_unified.json \
  --tail-aware-weights-json custom_tail_weights.json \
  --epoch 100
```

In this case:
- Base settings come from `training_config_unified.json`
- Tail-aware weights are overridden by `custom_tail_weights.json`

### Legacy: Individual Files (Still Supported)

```bash
python train_cnp_model.py \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variable-weights-json config/variable_weights_config.json \
  --tail-aware-weights-json list1_tail_weights.json \
  --pft-zero-sparsity-weights-json pft_zero_weights.json \
  --pft1d-activation-overrides-json pft1d_activation_overrides.json \
  --epoch 100
```

## Configuration Sections

### `training_hyperparameters`
Core training hyperparameters and loss weights:
- **Basic training**: `num_epochs`, `batch_size`, `learning_rate`
- **Optimizer**: `optimizer_type` (`"adam"`, `"sgd"`, `"adamw"`), `weight_decay`
- **Scheduler**: `use_scheduler` (boolean), `scheduler_type` (`"step"`, `"cosine"`, `"plateau"`), `scheduler_step_size`, `scheduler_gamma`
- **Loss weights**: `scalar_loss_weight`, `vector_loss_weight`, `matrix_loss_weight`, `xsmrpool_loss_weight`, `litter_c_loss_weight`, `litter_n_loss_weight`, `litter_p_loss_weight`

**Note**: CLI arguments take precedence over JSON values for `num_epochs`, `batch_size`, `learning_rate`, and loss weights (allows easy overrides).

### `reproducibility_config`
Settings for reproducibility and data handling:
- `random_seed`: Random seed for training and data splitting (default: `42`)
- `strict_determinism`: Enable strict deterministic mode (default: `false`)
- `train_split`: Train/validation split ratio (default: `0.8`)
- `normalization`: Normalization method - `"group"`, `"individual"`, or `"hybrid"` (default: `"individual"`)
- `dropout_p`: Global dropout probability override (default: `null` to use model default)

**Note**: CLI arguments (`--split-seed`, `--strict-determinism`, `--train-split`, `--normalization`, `--dropout-p`) take precedence over JSON values.

### `data_filtering_config`
Settings for which samples are included at training time. **Use the training config JSON here, not CNP_IO.txt**: CNP_IO defines variables and data paths; filtering (tropical, natveg) is a run choice and belongs in config so the same variable list can be reused with different filters.

- `tropical_only`: Filter dataset to tropical latitudes (default: `false`)
- `tropical_lat_range`: Latitude range as `[min, max]` or `"min,max"` string (default: `[-23.5, 23.5]`)
- `tropical_lat_column`: Optional column name for latitude (default: auto-detect, e.g. `Latitude`)
- `natveg_only`: If `true`, keep only gridcells with natural vegetation: `PCT_NATVEG > 0` and `PCT_NAT_PFT_0 < 100` (default: `false`). See `docs/EXCLUDED_SAMPLE_ANALYSIS.md` for impact (~33% of samples excluded on typical data).
- `longitudes_to_drop`: List of longitudes (degrees) to exclude from training, e.g. `[0, 358.75]`. Samples whose longitude matches (within tolerance) are dropped. Overrides the same option in CNP_IO. Use 0–360° convention to match data.
- `region_boxes`: Optional list of boxes to **keep**; only gridcells inside at least one box are used. Each box is `[lat_min, lat_max, lon_min, lon_max]` with longitude in **0–360°**. E.g. Amazon + Africa: `[[-30, 10, 270, 330], [-15, 15, 0, 30]]`. See `docs/PLAN_TWO_REGION_FIVE_P_TRAINING.md`.

**Note**: CLI arguments (`--tropical-only`, `--tropical-lat-range`, `--natveg-only`, `--longitudes-to-drop`) take precedence over JSON values. JSON `data_filtering_config` overrides values from the CNP_IO variable list (e.g. longitude filtering in CNP_IO).

### `variable_weights`
Contains three subsections:
- `pft1d_weights`: Weights for 1D PFT variables
- `soil2d_weights`: Weights for 2D soil variables  
- `scalar_weights`: Weights for scalar output variables

### `tail_aware_config`
Configuration for tail-aware loss:
- `loss`: Loss type (`"log1p_mse"`, `"log1p_huber"`, `"log1p_quantile"`, or `"mse"`)
- `epsilon`: Small epsilon value for numerical stability (default: `1e-8`)
- `huber_delta`: Delta parameter for Huber loss (optional, default: `1.0`)

### `tail_aware_weights`
Per-variable multipliers for tail-aware loss. Applied to variables listed in `tail_aware_vars`.

### `pft_mask_config`
Configuration for PFT presence masking:
- `mask_absent_pfts`: Boolean to enable/disable masking absent PFTs using PCT_NAT_PFT_1..16 (default: `false`)
- `pft_presence_threshold`: Minimum PFT percent for training-only mask (default: `0.0` meaning `pct > 0`; e.g. `2.0` means `pct >= 2%`). Inference always uses `pct > 0`.

**Note**: When `mask_absent_pfts` is enabled, predictions are zeroed where PFTs are absent, and loss is only computed for present PFTs. The `pft_presence_threshold` allows stricter masking during training (e.g., ignore PFTs with <2% coverage) while inference uses the standard `pct > 0` threshold.

### `pft_zero_sparsity_config`
Configuration for PFT zero sparsity regularization:
- `weight`: Global weight multiplier for sparsity penalty (default: `0.0`, set `>0` to enable)
- `threshold`: Threshold in normalized target space for zero mask (default: `1e-8`)

### `pft_zero_sparsity_weights`
Per-variable weights for PFT zero sparsity penalty. Applied when `pft_zero_sparsity_config.weight > 0`.

### `pft1d_activation_overrides`
Per-variable activation function overrides. Valid values: `"abs"`, `"relu"`, `"softplus"`, `"linear"`.

## Precedence Rules

When both unified config and CLI arguments are specified:

1. **Training hyperparameters**: CLI args (`--epochs`, `--batch-size`, `--learning-rate`, `--xsmrpool-loss-weight`, `--litter-*-loss-weight`) override unified config
2. **Reproducibility config**: CLI args (`--split-seed`, `--strict-determinism`, `--train-split`, `--normalization`, `--dropout-p`) override unified config
3. **Data filtering config**: CLI args (`--tropical-only`, `--tropical-lat-range`, `--natveg-only`, `--longitudes-to-drop`) override unified config
4. **PFT mask config**: CLI args (`--mask-absent-pfts` / `--no-mask-absent-pfts`, `--pft-presence-threshold`) override unified config
5. **Tail-aware config**: CLI args (`--tail-aware-loss`, `--tail-aware-eps`) override unified config
6. **Tail-aware weights**: Individual `--tail-aware-weights-json` overrides unified config
7. **Sparsity config**: CLI args (`--pft-zero-sparsity-weight`, `--pft-zero-threshold`) override unified config
8. **Sparsity weights**: Individual `--pft-zero-sparsity-weights-json` overrides unified config
9. **Activation overrides**: Individual `--pft1d-activation-overrides-json` overrides unified config
10. **Variable weights**: Individual `--variable-weights-json` overrides unified config

This allows you to:
- Use unified config as a base/default configuration
- Override specific sections with individual files when needed
- Gradually migrate from individual files to unified config

## Example: Complete Configuration

```json
{
  "training_hyperparameters": {
    "num_epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "optimizer_type": "adamw",
    "weight_decay": 0.01,
    "use_scheduler": true,
    "scheduler_type": "cosine",
    "xsmrpool_loss_weight": 10.0
  },
  "reproducibility_config": {
    "random_seed": 42,
    "strict_determinism": false,
    "train_split": 0.8,
    "normalization": "individual",
    "dropout_p": null
  },
  "data_filtering_config": {
    "tropical_only": true,
    "tropical_lat_range": [-23.5, 23.5],
    "longitudes_to_drop": [0, 358.75]
  },
  "variable_weights": {
    "pft1d_weights": {
      "xsmrpool": 5.0,
      "cpool": 2.0,
      "npool": 2.0,
      "ppool": 2.0,
      "tlai": 3.0,
      "totvegc": 2.0,
      "litr2p_vr": 5.0,
      "litr2n": 5.0
    },
    "soil2d_weights": {
      "primp_vr": 3.0,
      "labilep_vr": 2.5,
      "secondp_vr": 2.5,
      "litr2p_vr": 5.0,
      "litr2n_vr": 5.0
    },
    "scalar_weights": {
      "GPP": 1.5,
      "NPP": 1.5,
      "AR": 1.2,
      "HR": 1.2
    }
  },
  "tail_aware_config": {
    "loss": "log1p_mse",
    "epsilon": 1e-8
  },
  "tail_aware_weights": {
    "cpool": 5.0,
    "deadstemc": 5.0,
    "deadcrootc": 5.0,
    "livestemc": 5.0,
    "livecrootc": 5.0,
    "npool": 5.0,
    "ppool": 5.0,
    "labilep": 5.0,
    "primp_vr": 5.0,
    "occld_vr": 5.0,
    "litr2c_vr": 5.0,
    "litr2p_vr": 5.0,
    "litr2n": 5.0,
    "litr3c_vr": 5.0,
    "Litr3p_vr": 5.0,
    "Litr3n_vr": 5.0
  },
  "pft_mask_config": {
    "mask_absent_pfts": true,
    "pft_presence_threshold": 0.0
  },
  "pft_zero_sparsity_config": {
    "weight": 1.0,
    "threshold": 1e-8
  },
  "pft_zero_sparsity_weights": {
    "cpool": 1.0,
    "deadstemc": 1.0,
    "deadcrootc": 1.0,
    "livestemc": 1.0,
    "livecrootc": 1.0
  },
  "pft1d_activation_overrides": {
    "cpool": "abs",
    "deadstemc": "abs",
    "deadcrootc": "abs",
    "livestemc": "abs",
    "livecrootc": "abs"
  }
}
```

## Benefits

✅ **Single source of truth** - All user-defined configs in one place  
✅ **Easier to manage** - No need to track multiple files  
✅ **Version control friendly** - One file to commit/compare  
✅ **Backward compatible** - Individual files still work  
✅ **Flexible** - Can override specific sections with individual files
