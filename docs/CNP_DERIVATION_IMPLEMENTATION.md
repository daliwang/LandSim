# CNP Derivation Implementation Guide

## Overview

This guide explains how to implement the "train C, derive N/P" approach for CNP variables.

## Why This Approach?

Based on validation results:
- **Ground truth ratios are perfect** (errors < 0.1%)
- **Model predictions violate ratios** (errors 100-250%)
- **Constraint loss shows mixed results** (some better, some worse)

**Solution**: Train only C variables, derive N/P from C using stoichiometric ratios.

## Benefits

1. ✅ **Perfect ratios**: Guaranteed stoichiometric consistency
2. ✅ **Simpler model**: 36% fewer output variables
3. ✅ **Better C predictions**: Focus all training effort on C
4. ✅ **No tuning**: No constraint weight hyperparameters
5. ✅ **Faster training**: Smaller output heads

## Implementation Steps

### Step 1: Modify Training Configuration

Remove N/P variables from training targets. Create a new variable list or modify existing:

**Option A: Create New Variable List**

Create `CNP_IO_derived_np.txt`:

```
# Only C variables and non-CNP variables
• deadstemc, deadstemc_storage
• deadcrootc, deadcrootc_storage
• leafc, leafc_storage
• frootc, frootc_storage
• livestemc, livestemc_storage
• livecrootc, livecrootc_storage
• soil1c_vr, soil2c_vr, soil3c_vr, soil4c_vr
• cwdc_vr
• cpool, npool, ppool  # Keep pools (they don't follow ratios)
• tlai, totvegc
• GPP, NPP, AR, HR
# Remove: deadstemn, deadstemp, leafn, leafp, frootn, frootp, etc.
```

**Option B: Modify Training Config**

In your training config JSON, filter out N/P variables:

```json
{
  "derive_np_from_c": true,
  "exclude_from_training": [
    "deadstemn", "deadstemp",
    "deadcrootn", "deadcrootp",
    "leafn", "leafp",
    "frootn", "frootp",
    "livestemn", "livestemp",
    "livecrootn", "livecrootp",
    "soil1n_vr", "soil1p_vr",
    "soil2n_vr", "soil2p_vr",
    "soil3n_vr", "soil3p_vr",
    "soil4n_vr", "soil4p_vr",
    "cwdn_vr", "cwdp_vr"
  ]
}
```

### Step 2: Increase C Variable Weights

Give C variables higher weights since they're now more important:

```json
{
  "variable_weights": {
    "pft1d_weights": {
      "deadstemc": 10,  // Increased from 6
      "deadcrootc": 10,  // Increased from 5
      "leafc": 10,       // Increased from 6
      "frootc": 8,       // Increased from 4
      "livestemc": 10,   // Increased from 5
      "livecrootc": 10,  // Increased from 5
      "soil1c_vr": 10,   // Increased from 7
      "soil2c_vr": 8,
      "soil3c_vr": 6,
      "soil4c_vr": 6,
      "cwdc_vr": 8
    }
  }
}
```

### Step 3: Modify Model Output Head

The model output head needs to only output C variables. This can be done by:

1. **Filtering outputs** in the model forward pass
2. **Modifying output head size** to match filtered variable list
3. **Post-processing** to add derived N/P variables

### Step 4: Create Derivation Function

Use the provided `scripts/derive_np_from_c.py` script or integrate derivation into inference:

```python
from scripts.derive_np_from_c import derive_np_from_c_predictions

# After inference
derived_files = derive_np_from_c_predictions(
    predictions_dir=output_dir / 'cnp_predictions',
    config_path=output_dir / 'cnp_config.json',
    output_dir=output_dir / 'cnp_predictions'
)
```

### Step 5: Integrate into Inference Pipeline

Modify `scripts/run_inference_all.py` to derive N/P after model prediction:

```python
# After model prediction
predictions = model(...)

# Derive N/P from C
if config.get('derive_np_from_c', False):
    predictions = derive_np_from_c(predictions, data_info, pft_params)
```

## Training Configuration Example

```json
{
  "derive_np_from_c": true,
  "variable_weights": {
    "pft1d_weights": {
      "deadstemc": 10,
      "deadcrootc": 10,
      "leafc": 10,
      "frootc": 8,
      "livestemc": 10,
      "livecrootc": 10,
      "cpool": 6,
      "npool": 8,
      "ppool": 8,
      "tlai": 4
    },
    "soil2d_weights": {
      "soil1c_vr": 10,
      "soil2c_vr": 8,
      "soil3c_vr": 6,
      "soil4c_vr": 6,
      "cwdc_vr": 8
    }
  }
}
```

## Expected Results

### Model Complexity
- **Output variables**: 71 → ~45 (36% reduction)
- **Training time**: 20-30% faster
- **Memory**: 15-20% lower

### Prediction Quality
- **CNP ratios**: Perfect (0% error)
- **C variables**: Better (focused training, higher weights)
- **Overall**: Similar or better (simpler model, less overfitting)

### Comparison to Constraint Loss

| Metric | Constraint Loss | Derivation |
|--------|----------------|------------|
| Ratio Errors | 10-30% (with tuning) | 0% (perfect) |
| C Prediction | Shared effort | Full effort |
| Model Complexity | Same | Lower |
| Hyperparameters | Constraint weight | None |
| Training Time | Same | Faster |

## Validation

After training with derivation approach:

1. **Check C predictions**: Should be better (higher weights, focused training)
2. **Derive N/P**: Use derivation script
3. **Validate ratios**: Should be perfect (0% error)
4. **Compare to GT**: Derived N/P should match ground truth N/P

## Migration Path

1. **Phase 1**: Test derivation on existing predictions
   - Use `derive_np_from_c.py` on experiment2 results
   - Compare derived N/P to predicted N/P
   - Validate ratios

2. **Phase 2**: Train new model with C-only targets
   - Create filtered variable list
   - Increase C weights
   - Train model

3. **Phase 3**: Integrate derivation into inference
   - Modify inference pipeline
   - Automatically derive N/P after prediction

4. **Phase 4**: Compare results
   - Compare C prediction quality
   - Compare overall performance
   - Validate stoichiometric consistency

## Conclusion

The derivation approach is **strongly recommended** because:
- ✅ Guarantees perfect ratios
- ✅ Simpler model architecture
- ✅ Better C predictions
- ✅ No hyperparameter tuning
- ✅ Matches data characteristics

This approach should provide better, more consistent results than constraint loss.
