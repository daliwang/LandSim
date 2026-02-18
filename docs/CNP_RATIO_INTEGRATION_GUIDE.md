# CNP Ratio Constraint Integration Guide

This guide explains how to integrate CNP stoichiometric ratio constraints into the training pipeline to improve model performance for variables with poor CNP ratio adherence.

## Overview

The CNP ratio constraint loss (`CNPRatioConstraintLoss`) enforces stoichiometric relationships between Carbon, Nitrogen, and Phosphorus variables during training. This helps the model learn to maintain consistent CNP ratios as defined in `CNP_STOICHIOMETRIC_RELATIONSHIPS.md`.

## Implementation Components

### 1. Loss Function (`training/losses.py`)

The `CNPRatioConstraintLoss` class has been added to `training/losses.py`. It supports:

- **Soft constraints**: Penalizes deviations from expected ratios
- **PFT-specific ratios**: Handles variable ratios per PFT
- **Woody PFT filtering**: Correctly handles non-woody PFTs for dead wood variables
- **1D and 2D variables**: Supports both PFT 1D and soil 2D variables

### 2. Validation Script (`scripts/validate_cnp_ratios.py`)

A validation script to check CNP ratios in predictions:

```bash
python scripts/validate_cnp_ratios.py --results-dir cnp_results/run_20260212_162802_experiment_2
```

## Integration Steps

### Step 1: Modify Training Script

Add CNP ratio constraint loss to your training loop. Here's an example modification to `train_cnp_model.py`:

```python
from training.losses import CNPRatioConstraintLoss

# In the training function, after model initialization:
# Initialize CNP ratio constraint loss
cnp_ratio_loss_fn = CNPRatioConstraintLoss(
    data_info=config.data_info,
    constraint_weight=1.0,  # Adjust based on performance
    ratio_tolerance=0.1,
    mode="soft"
)

# In the training loop, after computing standard losses:
# Compute CNP ratio constraint loss
pft_1d_var_indices = {var: i for i, var in enumerate(config.data_info['variables_1d_pft'])}
soil_2d_var_indices = {var: i for i, var in enumerate(config.data_info['variables_2d_soil'])}

# Extract PFT parameters from batch
pft_params = batch_data.get('pft_param', None)  # Adjust based on your data loader

cnp_ratio_loss = cnp_ratio_loss_fn(
    pft_1d_pred=outputs['pft_1d'],
    pft_1d_target=targets['pft_1d'],
    soil_2d_pred=outputs['soil_2d'],
    soil_2d_target=targets['soil_2d'],
    pft_params=pft_params,
    pft_1d_var_indices=pft_1d_var_indices,
    soil_2d_var_indices=soil_2d_var_indices
)

# Add to total loss
total_loss = scalar_loss + vector_loss + matrix_loss + cnp_ratio_loss
```

### Step 2: Update Configuration

Add CNP ratio constraint configuration to your training config JSON:

```json
{
  "cnp_ratio_constraints": {
    "enabled": true,
    "constraint_weight": 1.0,
    "ratio_tolerance": 0.1,
    "mode": "soft",
    "variables": {
      "pft_1d": ["deadstemc", "deadcrootc", "leafc", "frootc", "livestemc", "livecrootc"],
      "soil_2d": ["soil1c_vr", "soil2c_vr", "soil3c_vr", "soil4c_vr", "cwdc_vr"]
    }
  }
}
```

### Step 3: Ensure PFT Parameters are Available

The loss function needs access to PFT parameters during training. Ensure your data loader provides:

- `pft_deadwdcn`: Dead wood C:N ratio
- `pft_leafcn`: Leaf C:N ratio
- `pft_frootcn`: Fine root C:N ratio
- `pft_livewdcn`: Live wood C:N ratio
- `pft_woody`: Woody flag (for dead wood filtering)

These should be included in `data_config.pft_param_columns`.

### Step 4: Adjust Constraint Weight

The `constraint_weight` parameter controls how strongly ratio constraints are enforced:

- **Low weight (0.1-0.5)**: Soft guidance, allows some ratio violations
- **Medium weight (1.0-2.0)**: Balanced enforcement
- **High weight (5.0-10.0)**: Strong enforcement, may impact overall loss

Start with `constraint_weight=1.0` and adjust based on validation results.

## Post-Processing Option

For inference, you can also enforce ratios as a post-processing step:

```python
def enforce_cnp_ratios_post_process(
    predictions: Dict[str, torch.Tensor],
    data_info: dict,
    pft_params: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Enforce CNP ratios on predictions as post-processing.
    This ensures stoichiometric consistency even if model predictions violate ratios.
    """
    # Extract variable indices
    pft_1d_vars = data_info['variables_1d_pft']
    pft_1d_var_indices = {var: i for i, var in enumerate(pft_1d_vars)}
    
    # For each C variable, derive N and P from C prediction
    # Example for deadstemc:
    if 'deadstemc' in pft_1d_var_indices:
        c_idx = pft_1d_var_indices['deadstemc']
        n_idx = pft_1d_var_indices['deadstemn']
        p_idx = pft_1d_var_indices['deadstemp']
        
        # Extract C prediction
        c_pred = predictions['pft_1d'][:, c_idx * n_pfts:(c_idx + 1) * n_pfts]
        
        # Derive N and P from C
        deadwdcn = pft_params[:, pft_deadwdcn_idx, :]
        n_derived = c_pred / (deadwdcn + 1e-8)
        p_derived = c_pred / 3000.0
        
        # Replace predictions with derived values
        predictions['pft_1d'][:, n_idx * n_pfts:(n_idx + 1) * n_pfts] = n_derived
        predictions['pft_1d'][:, p_idx * n_pfts:(p_idx + 1) * n_pfts] = p_derived
    
    return predictions
```

## Validation

After training, validate CNP ratios using the validation script:

```bash
python scripts/validate_cnp_ratios.py \
    --results-dir cnp_results/run_YYYYMMDD_HHMMSS_experiment \
    --output cnp_results/run_YYYYMMDD_HHMMSS_experiment/analysis/cnp_ratio_validation.json
```

The script will:
1. Load predictions and ground truth
2. Check CNP ratios for all configured variables
3. Compute ratio violation statistics
4. Generate a validation report

## Expected Improvements

Based on the analysis in `CNP_RATIO_IMPROVEMENT_PLAN.md`, enforcing CNP ratios should improve:

### High Priority Variables
- **soil1c/n/p**: From 30% good → Expected 70-80% good
- **deadstemc/n/p**: From 62.5% good → Expected 85-90% good
- **leafc/n/p**: From 68.8% good → Expected 85-90% good
- **litr2c/n/p**: From 10% good → Expected 50-70% good

### Medium Priority Variables
- **frootc/n/p**: From 81.2% good → Expected 90-95% good
- **Storage variants**: Should improve proportionally

## Troubleshooting

### Issue: Loss becomes too large
- **Solution**: Reduce `constraint_weight` or increase `ratio_tolerance`

### Issue: Model performance degrades
- **Solution**: Start with lower `constraint_weight` (0.1-0.5) and gradually increase

### Issue: PFT parameters not found
- **Solution**: Ensure PFT parameters are included in `data_config.pft_param_columns` and passed to the loss function

### Issue: Variable indices mismatch
- **Solution**: Verify variable order matches between `data_info['variables_1d_pft']` and model outputs

## Advanced Usage

### Selective Variable Constraints

You can selectively apply constraints to specific variables:

```python
# Only enforce constraints for high-priority variables
cnp_ratio_loss_fn = CNPRatioConstraintLoss(
    data_info=config.data_info,
    constraint_weight=2.0,  # Higher weight for critical variables
    enabled_variables={
        'pft_1d': ['deadstemc', 'leafc', 'frootc'],
        'soil_2d': ['soil1c_vr']
    }
)
```

### Adaptive Constraint Weight

Adjust constraint weight based on training progress:

```python
# Start with low weight, increase as training progresses
if epoch < 10:
    constraint_weight = 0.5
elif epoch < 50:
    constraint_weight = 1.0
else:
    constraint_weight = 2.0

cnp_ratio_loss_fn.constraint_weight = constraint_weight
```

## References

- `docs/CNP_STOICHIOMETRIC_RELATIONSHIPS.md`: Complete ratio definitions
- `docs/CNP_RATIO_IMPROVEMENT_PLAN.md`: Analysis of variables needing improvement
- `training/losses.py`: Implementation of `CNPRatioConstraintLoss`
- `scripts/validate_cnp_ratios.py`: Validation script
