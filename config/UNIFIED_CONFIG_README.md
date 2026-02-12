# Unified Training Configuration

This document explains how to use a single JSON file to configure all training parameters.

## Overview

Instead of managing multiple separate JSON files, you can now use a single unified configuration file that contains all user-defined settings:

- Variable weights (PFT1D, Soil2D, Scalar)
- Tail-aware weights
- PFT zero sparsity weights
- PFT1D activation overrides

## JSON File Format

Create a single JSON file (`training_config_unified.json`) with the following structure:

```json
{
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
  "tail_aware_weights": {
    "cpool": 5.0,
    "deadstemc": 5.0,
    "litr2p_vr": 5.0,
    "litr2n": 5.0
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

### `variable_weights`
Contains three subsections:
- `pft1d_weights`: Weights for 1D PFT variables
- `soil2d_weights`: Weights for 2D soil variables  
- `scalar_weights`: Weights for scalar output variables

### `tail_aware_weights`
Per-variable multipliers for tail-aware loss. Applied to variables listed in `tail_aware_vars`.

### `pft_zero_sparsity_weights`
Per-variable weights for PFT zero sparsity penalty. Applied when `pft_zero_sparsity_weight > 0`.

### `pft1d_activation_overrides`
Per-variable activation function overrides. Valid values: `"abs"`, `"relu"`, `"softplus"`, `"linear"`.

## Precedence Rules

When both unified config and individual files are specified:

1. **Tail-aware weights**: Individual `--tail-aware-weights-json` overrides unified config
2. **Sparsity weights**: Individual `--pft-zero-sparsity-weights-json` overrides unified config
3. **Activation overrides**: Individual `--pft1d-activation-overrides-json` overrides unified config
4. **Variable weights**: Individual `--variable-weights-json` overrides unified config

This allows you to:
- Use unified config as a base/default configuration
- Override specific sections with individual files when needed
- Gradually migrate from individual files to unified config

## Example: Complete Configuration

```json
{
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
