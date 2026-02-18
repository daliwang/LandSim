# CNP Ratio Improvement Summary

## Quick Reference

This document provides a quick summary of the CNP ratio improvement implementation. For detailed information, see the referenced documents.

## Problem Statement

Experiment2 results show several variables with poor performance that violate CNP stoichiometric relationships:

- **soil1c/n/p**: 30% good, 50% bad (WORST)
- **deadstemc/n/p**: 62.5% good, 31.2% bad
- **leafc/n/p**: 68.8% good, 31.2% bad
- **frootc/n/p**: 81.2% good, 18.8% bad
- **litr2c/n/p**: 10% good, 80% bad

These variables should follow stoichiometric ratios but currently don't, indicating the model needs explicit ratio constraints.

## Solution Components

### 1. Analysis Document
**File**: `docs/CNP_RATIO_IMPROVEMENT_PLAN.md`
- Comprehensive list of variables needing improvement
- Priority ordering
- Expected improvements

### 2. Loss Function
**File**: `training/losses.py` → `CNPRatioConstraintLoss`
- Enforces CNP ratios during training
- Supports PFT-specific and constant ratios
- Handles woody/non-woody PFT filtering

### 3. Validation Script
**File**: `scripts/validate_cnp_ratios.py`
- Validates CNP ratios in predictions
- Generates violation statistics
- Can be run on any experiment results

### 4. Integration Guide
**File**: `docs/CNP_RATIO_INTEGRATION_GUIDE.md`
- Step-by-step integration instructions
- Configuration examples
- Troubleshooting guide

## Quick Start

### 1. Validate Current Ratios
```bash
python scripts/validate_cnp_ratios.py \
    --results-dir cnp_results/run_20260212_162802_experiment_2
```

### 2. Add to Training
```python
from training.losses import CNPRatioConstraintLoss

cnp_ratio_loss_fn = CNPRatioConstraintLoss(
    data_info=config.data_info,
    constraint_weight=1.0,
    mode="soft"
)

# Add to training loop
cnp_ratio_loss = cnp_ratio_loss_fn(
    pft_1d_pred=outputs['pft_1d'],
    pft_1d_target=targets['pft_1d'],
    soil_2d_pred=outputs['soil_2d'],
    soil_2d_target=targets['soil_2d'],
    pft_params=pft_params
)

total_loss = scalar_loss + vector_loss + matrix_loss + cnp_ratio_loss
```

### 3. Validate After Training
```bash
python scripts/validate_cnp_ratios.py \
    --results-dir cnp_results/run_NEW_EXPERIMENT
```

## Variables Covered

### PFT 1D Variables
- ✅ deadstemc → deadstemn, deadstemp
- ✅ deadcrootc → deadcrootn, deadcrootp
- ✅ leafc → leafn, leafp
- ✅ frootc → frootn, frootp
- ✅ livestemc → livestemn, livestemp
- ✅ livecrootc → livecrootn, livecrootp
- ✅ Storage variants of all above

### Soil 2D Variables
- ✅ soil1c_vr → soil1n_vr, soil1p_vr
- ✅ soil2c_vr → soil2n_vr, soil2p_vr
- ✅ soil3c_vr → soil3n_vr, soil3p_vr
- ✅ soil4c_vr → soil4n_vr, soil4p_vr
- ✅ cwdc_vr → cwdn_vr, cwdp_vr

## Key Ratios

### Dead Wood
- C:N = 500 (deadwdcn)
- C:P = 3000 (deadwdcp)
- Only for woody PFTs

### Leaf
- C:N = 25-40 (leafcn, PFT-specific)
- C:P = 250-600 (leafcp, PFT-specific)

### Fine Root
- C:N = 42 (frootcn)
- C:P = 1000 (frootcp)

### Live Wood
- C:N = 50 (livewdcn)
- C:P = 3000 (livewdcp)

### Soil Layers
- Layer 1 & 2: C:N = 12, C:P = 360
- Layer 3 & 4: C:N = 10, C:P = 500

## Expected Results

After implementing CNP ratio constraints:

| Variable | Current | Expected |
|----------|---------|----------|
| soil1c/n/p | 30% good | 70-80% good |
| deadstemc/n/p | 62.5% good | 85-90% good |
| leafc/n/p | 68.8% good | 85-90% good |
| frootc/n/p | 81.2% good | 90-95% good |

## Next Steps

1. **Immediate**: Run validation script on experiment2 to baseline current ratio violations
2. **Short-term**: Integrate CNP ratio loss into training with `constraint_weight=1.0`
3. **Medium-term**: Tune constraint weight based on validation results
4. **Long-term**: Consider post-processing option for inference

## Files Created/Modified

1. ✅ `docs/CNP_RATIO_IMPROVEMENT_PLAN.md` - Analysis document
2. ✅ `training/losses.py` - Added `CNPRatioConstraintLoss` class
3. ✅ `scripts/validate_cnp_ratios.py` - Validation script
4. ✅ `docs/CNP_RATIO_INTEGRATION_GUIDE.md` - Integration guide
5. ✅ `docs/CNP_RATIO_SUMMARY.md` - This summary

## References

- **Stoichiometric Relationships**: `docs/CNP_STOICHIOMETRIC_RELATIONSHIPS.md`
- **Model Variable Quantities**: `docs/model_variable_quantities.txt`
- **Experiment2 Results**: `cnp_results/run_20260212_162802_experiment_2/`
