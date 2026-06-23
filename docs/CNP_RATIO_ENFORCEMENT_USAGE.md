# CNP Ratio Enforcement Usage Guide

## Overview

This guide explains how to enable CNP stoichiometric ratio enforcement during inference. The system will automatically derive N and P variables from C predictions using PFT-specific ratios, ensuring perfect stoichiometric relationships.

## Quick Start

### Option 1: Enable during inference (default)

By default, inference derives N/P from C after the forward pass. To enable explicitly:

```bash
python scripts/run_inference_all.py \
    --model cnp_results/run_20260212_162802_experiment_2/model.pth \
    --output-dir cnp_inference_with_ratios \
    --derive-np-from-c
```

To **skip** derivation (e.g. Phase2/Phase3 tropical restarts without fixed CNP ratios):

```bash
python scripts/run_inference_all.py \
    --model cnp_results/run_YYYYMMDD_phase2_tropical/cnp_model.pt \
    --output-dir cnp_inference_entire_dataset \
    --inference-full-grid \
    --no-derive-np-from-c
```

See [RERUN_PHASE2_PHASE3_NO_CNP_FIX.md](./RERUN_PHASE2_PHASE3_NO_CNP_FIX.md) for the full Phase2 + Phase3 rerun workflow.

This will:
1. Run normal inference (predicting all CNP variables)
2. Automatically derive N/P from C predictions after inference completes
3. Overwrite the N/P prediction files with stoichiometrically-correct values

### Option 2: Run derivation separately

If you've already run inference, you can derive N/P from C predictions separately:

```bash
python scripts/derive_np_from_c.py \
    --predictions-dir cnp_results/run_YYYYMMDD_HHMMSS/cnp_predictions \
    --config-path cnp_results/run_YYYYMMDD_HHMMSS/cnp_config.json \
    --output-dir cnp_results/run_YYYYMMDD_HHMMSS/cnp_predictions
```

**Note:** If `--output-dir` is omitted, it defaults to the same as `--predictions-dir`, which will overwrite existing N/P files.

## What Gets Modified

The derivation process overwrites the following N/P prediction files:

### PFT 1D Variables:
- `predictions_Y_deadstemn.csv` (derived from `deadstemc`)
- `predictions_Y_deadstemp.csv` (derived from `deadstemc`)
- `predictions_Y_frootn.csv` (derived from `frootc`)
- `predictions_Y_frootp.csv` (derived from `frootc`)
- `predictions_Y_leafn.csv` (derived from `leafc`)
- `predictions_Y_leafp.csv` (derived from `leafc`)
- `predictions_Y_livestemn.csv` (derived from `livestemc`)
- `predictions_Y_livestemp.csv` (derived from `livestemc`)

### Soil 2D Variables:
- `predictions_Y_soil1n_vr.csv` (derived from `soil1c_vr`)
- `predictions_Y_soil1p_vr.csv` (derived from `soil1c_vr`)
- `predictions_Y_soil2n_vr.csv` (derived from `soil2c_vr`)
- `predictions_Y_soil2p_vr.csv` (derived from `soil2c_vr`)
- `predictions_Y_soil3n_vr.csv` (derived from `soil3c_vr`)
- `predictions_Y_soil3p_vr.csv` (derived from `soil3c_vr`)

## Stoichiometric Ratios Used

The derivation uses PFT-specific ratios from the model configuration:

- **Dead stem**: C:N = 500, C:P = 3000 (PFT-specific for C:N)
- **Fine root**: C:N = 42, C:P = 1000
- **Leaf**: C:N = 25-40 (PFT-specific), C:P = 250-600 (PFT-specific)
- **Live stem**: C:N = 50, C:P = 3000
- **Soil layers**: C:N = 10-12, C:P = 360

See `scripts/derive_np_from_c.py` for the complete ratio definitions.

## Training Considerations

**Important:** Training remains unchanged. You should still train with the **full CNP variable list** to benefit from multi-task learning. The derivation step only happens at inference time.

Training with only C variables (as attempted in `run_20260212_102026_updated9_reduced`) actually leads to **worse** C predictions because:
- Multi-task learning improves shared representations
- N and P targets provide useful auxiliary signals
- The model learns better C predictions when trained on all variables

See `docs/CNP_DERIVATION_CLARIFICATION.md` for detailed explanation.

## Validation

After running inference with `--derive-np-from-c`, you can validate the ratios:

```bash
python scripts/validate_cnp_ratios.py \
    --predictions-dir cnp_results/run_YYYYMMDD_HHMMSS/cnp_predictions \
    --ground-truth-dir cnp_results/run_YYYYMMDD_HHMMSS/cnp_predictions \
    --output-dir analysis
```

This will generate `analysis/cnp_ratio_validation.json` showing:
- CN and CP ratio errors (should be near zero after derivation)
- RMSE for derived vs. original N/P predictions
- Per-variable statistics

## Workflow Summary

1. **Train** with full CNP variable list (e.g., `CNP_IO_updated9_LT.txt`)
2. **Run inference** with `--derive-np-from-c` flag
3. **Validate** ratios using `validate_cnp_ratios.py`
4. **Use predictions** - N/P values are now stoichiometrically consistent with C

## Troubleshooting

### Error: "derive_np_from_c module not available"
- Ensure `scripts/derive_np_from_c.py` exists
- Check Python path includes project root

### Error: "cnp_config.json not found"
- The script searches for `cnp_config.json` in the model directory and parent directories
- Ensure your training run saved `cnp_config.json` alongside `model.pth`

### Derived files not appearing
- Check that C prediction files exist (e.g., `predictions_Y_leafc.csv`)
- Verify the variable names match expected patterns
- Check logs for specific error messages

## See Also

- `docs/CNP_DERIVATION_CLARIFICATION.md` - Why train full CNP but derive N/P at inference
- `docs/CNP_DERIVATION_APPROACH.md` - Detailed approach explanation
- `docs/CNP_DERIVATION_IMPLEMENTATION.md` - Implementation details
- `scripts/derive_np_from_c.py` - Derivation script source code
