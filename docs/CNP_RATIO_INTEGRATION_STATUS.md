# CNP Ratio Constraint Integration Status

## ✅ Completed

1. **Loss Function Implementation** (`training/losses.py`)
   - ✅ `CNPRatioConstraintLoss` class fully implemented
   - ✅ Supports PFT 1D and soil 2D variables
   - ✅ Handles PFT-specific ratios and woody PFT filtering
   - ✅ Configurable constraint weight

2. **Training Integration** (`training/trainer.py`)
   - ✅ CNP ratio constraint loss initialized in trainer
   - ✅ Added to training loop loss computation
   - ✅ Added to validation loop loss computation
   - ✅ Error handling for graceful degradation

3. **Validation Script** (`scripts/validate_cnp_ratios.py`)
   - ✅ Validates CNP ratios in predictions
   - ✅ Generates detailed reports
   - ✅ Works with experiment results

4. **Documentation**
   - ✅ Integration guide (`docs/CNP_RATIO_INTEGRATION_GUIDE.md`)
   - ✅ Interpretation guide (`docs/CNP_RATIO_VALIDATION_INTERPRETATION.md`)
   - ✅ Improvement plan (`docs/CNP_RATIO_IMPROVEMENT_PLAN.md`)

## 🔧 How to Enable

### Option 1: Via Training Config JSON

Add to your training config JSON file:

```json
{
  "use_cnp_ratio_constraints": true,
  "cnp_ratio_constraint_weight": 1.0,
  "cnp_ratio_tolerance": 0.1
}
```

### Option 2: Via Command Line (if supported)

You may need to add command-line arguments to `train_cnp_model.py`:

```python
parser.add_argument('--use-cnp-ratio-constraints', action='store_true',
                   help='Enable CNP ratio constraint loss')
parser.add_argument('--cnp-ratio-constraint-weight', type=float, default=1.0,
                   help='Weight for CNP ratio constraint loss')
```

### Option 3: Direct Code Modification

In `train_cnp_model.py`, when creating the training config, add:

```python
config.use_cnp_ratio_constraints = True
config.cnp_ratio_constraint_weight = 1.0
config.cnp_ratio_tolerance = 0.1
```

## 📊 Expected Behavior

When enabled, the training will:

1. **Compute CNP ratio constraint loss** alongside standard losses
2. **Penalize ratio violations** for:
   - deadstemc → deadstemn, deadstemp
   - deadcrootc → deadcrootn, deadcrootp
   - leafc → leafn, leafp
   - frootc → frootn, frootp
   - livestemc → livestemn, livestemp
   - livecrootc → livecrootn, livecrootp
   - soil1c_vr → soil1n_vr, soil1p_vr
   - soil2c_vr → soil2n_vr, soil2p_vr
   - soil3c_vr → soil3n_vr, soil3p_vr
   - soil4c_vr → soil4n_vr, soil4p_vr
   - cwdc_vr → cwdn_vr, cwdp_vr

3. **Log the constraint loss** as part of total loss

## 🧪 Testing

To test the integration:

1. **Enable constraints** in config
2. **Run training** for a few epochs
3. **Check training logs** for CNP ratio loss values
4. **Run validation script** on results:
   ```bash
   python scripts/validate_cnp_ratios.py --results-dir <your_results_dir>
   ```
5. **Compare** ratio errors before/after

## ⚙️ Tuning

### Constraint Weight (`cnp_ratio_constraint_weight`)

- **Low (0.1-0.5)**: Soft guidance, allows some violations
- **Medium (1.0-2.0)**: Balanced enforcement (recommended starting point)
- **High (5.0-10.0)**: Strong enforcement, may dominate loss

**Recommendation**: Start with `1.0` and adjust based on validation results.

### Ratio Tolerance (`cnp_ratio_tolerance`)

Currently not used in soft constraint mode, but reserved for future use.

## 🐛 Troubleshooting

### Issue: CNP ratio loss is NaN
- **Check**: PFT parameters are loaded correctly
- **Check**: Variable indices match between data_info and outputs
- **Solution**: Reduce constraint_weight or check data

### Issue: Training loss increases significantly
- **Solution**: Reduce `cnp_ratio_constraint_weight` (try 0.5 or 0.1)

### Issue: Ratio violations still high after training
- **Solution**: Increase `cnp_ratio_constraint_weight` (try 2.0 or 5.0)

### Issue: Model performance degrades
- **Solution**: Start with lower weight (0.1-0.5) and gradually increase

## 📝 Next Steps

1. **Test Integration**: Enable constraints and run a short training test
2. **Baseline Comparison**: Compare results with/without constraints
3. **Tune Weight**: Adjust constraint_weight based on validation
4. **Monitor**: Track CNP ratio errors during training
5. **Iterate**: Fine-tune based on results

## 🔗 Related Files

- `training/losses.py`: CNPRatioConstraintLoss implementation
- `training/trainer.py`: Training loop integration
- `scripts/validate_cnp_ratios.py`: Validation script
- `docs/CNP_RATIO_INTEGRATION_GUIDE.md`: Detailed integration guide
