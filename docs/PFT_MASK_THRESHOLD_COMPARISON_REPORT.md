# PFT Mask Threshold Comparison Report

**Date:** February 15, 2026  
**Purpose:** Evaluate the impact of different PFT presence mask thresholds on model prediction quality

---

## Executive Summary

This report compares model performance across different PFT (Plant Functional Type) presence mask thresholds. The mask determines which PFTs are considered "present" during training and evaluation. We tested thresholds of 0% (baseline), 1%, and 2% to understand if filtering out small PFT percentages improves prediction quality.

**Key Finding:** The baseline (0% threshold, experiment2) performs best overall. While 1% and 2% thresholds with masked loss show improvement over the initial buggy 2% implementation, they still underperform the baseline by ~2-3% on good predictions and have ~2% more bad predictions.

---

## Experimental Setup

### Configurations Tested

| Run ID | Threshold | Mask Applied To | Loss Masking | Notes |
|--------|-----------|-----------------|--------------|-------|
| **experiment2** | 0% (pct > 0) | Training & Eval | No | Baseline - all non-zero PFTs included |
| **225702** | 2% (pct ≥ 2) | Training & Eval | No | **Bug:** Mask incorrectly applied to evaluation |
| **231556** | 2% (pct ≥ 2) | Training only | Yes | Fixed: Mask only for training, loss masked |
| **232944** | 1% (pct ≥ 1) | Training only | Yes | Fixed: Mask only for training, loss masked |

### Implementation Details

- **Training mask:** Only PFTs with `pct >= threshold` contribute to loss and have non-zero predictions
- **Evaluation mask:** Always uses `pct > 0` (all non-zero PFTs) for fair comparison
- **Loss masking:** Loss computed only over PFT slots where mask == 1 (prevents noise from tiny PFTs)

---

## Overall Performance Comparison

### Aggregate Statistics (920 total predictions)

| Metric | Experiment2<br/>(Baseline) | 2% Run<br/>(225702)<br/>Buggy | 2% Run<br/>(231556)<br/>Fixed | 1% Run<br/>(232944)<br/>Fixed |
|--------|---------------------------|------------------------------|------------------------------|------------------------------|
| **Good** | **619 (67.3%)** | 482 (52.4%) | 595 (64.7%) | 594 (64.6%) |
| **OK** | 138 (15.0%) | 90 (9.8%) | 140 (15.2%) | 147 (16.0%) |
| **Bad** | **163 (17.7%)** | 348 (37.8%) | 185 (20.1%) | 179 (19.5%) |
| **Good Variables** | **57** | 39 | 57 | 56 |
| **Bad Variables** | **9** | 25 | 9 | 10 |

### Key Observations

1. **Baseline (experiment2) is best:** 67.3% good vs 64.6-64.7% for threshold runs
2. **Bug fix critical:** Run 225702 (buggy) had 37.8% bad; fixed runs have ~20% bad
3. **1% vs 2% are very similar:** 1% has slightly fewer bad predictions (19.5% vs 20.1%) but one fewer good variable
4. **Threshold runs improve over buggy version:** Fixed runs recover ~12% good predictions and reduce bad by ~18%

---

## Detailed Variable-Level Comparison

### Variables with Best Predictions (All Runs)

All runs achieve 100% good predictions for:
- **Scalar variables:** AR, GPP, HR, NPP
- **Soil layer 3:** soil3c_vr, soil3n_vr, soil3p_vr
- **Phosphorus pools:** labilep_vr, secondp_vr

### Variables with Worst Predictions

| Variable | Experiment2 | 2% Fixed<br/>(231556) | 1% Fixed<br/>(232944) |
|----------|-------------|----------------------|----------------------|
| **primp_vr** | 0% good, 100% bad | 0% good, 100% bad | 0% good, 100% bad |
| **ppool** | 12.5% good, 75% bad | 12.5% good, 75% bad | 12.5% good, 75% bad |
| **npool** | 25% good, 62.5% bad | 18.8% good, 62.5% bad | 18.8% good, 62.5% bad |
| **litr2*_vr** | 10-20% good, 80% bad | 20% good, 80% bad | 10-20% good, 80% bad |
| **soil1*_vr** | 30% good, 50% bad | 30% good, 60% bad | 30% good, 60% bad |

**Note:** These "bad variables" remain problematic across all configurations, suggesting the issue is not mask-related but inherent to these variables (sparsity, heavy tails, etc.).

### Notable Differences Between Runs

#### Leaf Variables (leafc, leafn, leafp)

| Variable | Experiment2 | 2% Fixed | 1% Fixed |
|----------|-------------|-----------|----------|
| **leafc** | 68.8% good, 31.2% bad | 56.2% good, 31.2% bad | 56.2% good, 25% bad |
| **leafn** | 68.8% good, 31.2% bad | 56.2% good, 31.2% bad | 56.2% good, 25% bad |
| **leafp** | 68.8% good, 31.2% bad | 56.2% good, 37.5% bad | 56.2% good, 31.2% bad |

**Finding:** Threshold runs perform worse on leaf variables (~12% fewer good predictions).

#### Dead Stem Variables (deadstemc, deadstemn, deadstemp)

| Variable | Experiment2 | 2% Fixed | 1% Fixed |
|----------|-------------|-----------|----------|
| **deadstemc** | 62.5% good, 31.2% bad | 56.2% good, 31.2% bad | 62.5% good, 25% bad |
| **deadstemn** | 62.5% good, 31.2% bad | 56.2% good, 31.2% bad | 62.5% good, 25% bad |
| **deadstemp** | 62.5% good, 25% bad | 62.5% good, 31.2% bad | 62.5% good, 31.2% bad |

**Finding:** 1% run matches baseline on deadstemc/deadstemn; 2% run slightly worse.

#### Total Vegetation Carbon (totvegc)

| Run | Good | Bad |
|-----|------|-----|
| Experiment2 | 50% | 37.5% |
| 2% Fixed | 56.2% | 37.5% |
| 1% Fixed | 50% | 37.5% |

**Finding:** 2% run shows improvement (+6.2% good), but 1% matches baseline.

#### Carbon Pool (cpool)

| Run | Good | OK | Bad |
|-----|------|----|-----|
| Experiment2 | 31.2% | 43.8% | 25% |
| 2% Fixed | 37.5% | 37.5% | 25% |
| 1% Fixed | 31.2% | 50% | 18.8% |

**Finding:** 1% run has best bad rate (18.8% vs 25%), but experiment2 has more OK predictions.

#### occlp_vr (Notable Anomaly)

| Run | Good | OK | Bad |
|-----|------|----|-----|
| Experiment2 | 0% | 100% | 0% |
| 2% Fixed | 0% | 100% | 0% |
| 1% Fixed | 0% | 0% | 100% |

**Finding:** 1% run uniquely fails on occlp_vr (100% bad vs 100% OK in others). This explains why 1% has one more "bad variable" than 2%.

---

## Impact of Bug Fix

### Run 225702 (Buggy 2% Implementation)

**Problem:** The 2% mask was incorrectly applied to both training AND evaluation, causing:
- Predictions for PFTs with `0 < pct < 2%` were forced to zero during evaluation
- This created artificial "zero predictions" when ground truth was non-zero
- Result: 37.8% bad predictions (vs 17.7% baseline)

**Fix:** Separated training mask (strict, e.g. pct ≥ 2%) from evaluation mask (lenient, pct > 0), and masked loss to only compute over present PFTs.

**Result:** Fixed runs (231556, 232944) recover most performance, achieving ~64.6-64.7% good vs 67.3% baseline.

---

## Analysis: Why Threshold Runs Underperform Baseline

### Hypothesis 1: Loss of Signal from Small PFTs
- **Theory:** PFTs with 1-2% cover may still contain meaningful signal
- **Evidence:** 1% run (includes more small PFTs) performs slightly better than 2% on bad count
- **Conclusion:** Partially supported - small PFTs do contribute some signal

### Hypothesis 2: Reduced Training Data
- **Theory:** Masking out small PFTs reduces effective training samples
- **Evidence:** Threshold runs have fewer "good" predictions overall
- **Conclusion:** Supported - reducing training signal hurts overall performance

### Hypothesis 3: Variable-Specific Effects
- **Theory:** Some variables benefit from threshold (e.g. totvegc), others hurt (e.g. leafc)
- **Evidence:** Mixed results - totvegc improves with 2%, but leaf variables degrade
- **Conclusion:** Supported - threshold has variable-specific effects

---

## Recommendations

### 1. **Use Baseline (0% Threshold) for Production**
- Best overall performance (67.3% good, 17.7% bad)
- Most consistent across variable types
- No implementation complexity

### 2. **If Using Thresholds, Prefer 1% Over 2%**
- 1% has slightly fewer bad predictions (19.5% vs 20.1%)
- Includes more training signal from small PFTs
- **Exception:** If occlp_vr is critical, 2% avoids the 100% bad failure

### 3. **Variable-Specific Thresholds (Future Work)**
- Consider per-variable thresholds based on sparsity/tail behavior
- Variables like totvegc may benefit from stricter masking
- Variables like leafc may need lenient masking

### 4. **Alternative Approaches**
Instead of global thresholds, consider:
- **Tail-aware loss** (already implemented) - focuses on high-value predictions
- **Zero-inflated loss** for sparse variables (soil1*, litr2*)
- **Per-variable weighting** (already implemented) - increase weights for problematic variables
- **Oversampling** tropical/tail samples during training

---

## Technical Implementation Notes

### Mask Architecture
- **Training mask:** `pft_presence_mask_training` - strict threshold (e.g. pct ≥ 2%)
- **Evaluation mask:** `pft_presence_mask` - lenient (pct > 0)
- **Loss masking:** Loss computed only where training mask == 1

### Code Changes
1. **Data loader:** Creates both masks when threshold > 0
2. **Trainer:** Uses training mask for loss computation and prediction zeroing
3. **Evaluation:** Always uses lenient mask (pct > 0) for fair comparison

### Configuration
```bash
# Baseline (no threshold)
python train_cnp_model.py ... --mask-absent-pfts

# With threshold (e.g. 2%)
python train_cnp_model.py ... --mask-absent-pfts --pft-presence-threshold 2.0
```

---

## Conclusion

While PFT mask thresholds (1-2%) with masked loss show promise for reducing noise from tiny PFTs, they do not outperform the baseline (0% threshold) on aggregate metrics. The baseline achieves:
- **+2.6-2.7% more good predictions**
- **-1.8-1.9% fewer bad predictions**
- **More consistent performance across variable types**

The threshold approach may be valuable for:
- Specific variables that benefit from stricter masking (e.g. totvegc)
- Reducing computational cost (fewer PFT slots to process)
- Future experiments with variable-specific thresholds

**Recommendation:** Continue using the baseline (experiment2) configuration for production, but keep threshold implementation available for variable-specific tuning.

---

## Appendix: Run Details

| Run ID | Config File | Threshold | Epochs | Notes |
|--------|-------------|-----------|--------|-------|
| experiment2 | training_config_experiment_2.json | 0% | - | Baseline |
| 225702 | training_config_experiment_2.json | 2% | - | Buggy implementation |
| 231556 | training_config_experiment_2.json | 2% | - | Fixed with masked loss |
| 232944 | training_config_experiment_2.json | 1% | - | Fixed with masked loss |

All runs use the same base configuration (`training_config_experiment_2.json`) with variable weights, tail-aware loss, and other optimizations enabled.

---

**Report Generated:** February 15, 2026  
**Author:** AI Assistant  
**Review Status:** Ready for team discussion
