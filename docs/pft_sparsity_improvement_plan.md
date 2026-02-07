# PFT1D Sparsity Improvement Plan

This plan targets sparse PFT1D variables with many zero ground-truth values
and under-predicted maxima. The goal is to reduce false positives on zeros
while improving tail accuracy for large values.

## Scope

Focus variables:
- `cpool`
- `deadstemc`
- `deadcrootc`
- `livestemc`
- `livecrootc`

Primary evaluation subset:
- Tropical band (|lat| <= 23.5) and PFT present (`PCT_NAT_PFT_k > 0`)

## Baseline and diagnostics

1) Record baseline metrics for each variable:
   - R2, relative RMSE, relative MAE
   - Zero fraction in GT vs prediction
   - Max and p95 values
2) Save per-variable plots from `prediction_quality_by_variable`
   and top-bad plots for quick visual checks.

## Proposed improvements

### 1) Variable-specific sparsity penalty

Problem: global `pft_zero_sparsity_weight` can over-penalize or under-penalize
different variables.

Plan:
- Introduce per-variable sparsity weights for PFT1D variables.
- Start with higher weights for very sparse variables:
  - `cpool`, `deadcrootc`, `livestemc`, `livecrootc`, `deadstemc`
- Keep lower weights for variables with broader support.

Implementation idea:
- Add a map in training config, e.g. `pft_zero_sparsity_weights = {var: w}`.
- Apply weight per variable slice inside the PFT1D loss block.

### 2) Zero-inflated two-head modeling

Problem: regression head struggles with many zeros and large tails.

Plan:
- Head A: predict `p(y > 0)` (binary classification).
- Head B: predict magnitude `y` (regression on positive samples).
- Final prediction: `y_hat = p * magnitude`.

Implementation idea:
- Add a parallel sigmoid head for PFT1D zero/non-zero.
- Use BCE loss for the zero head, MSE/Huber for the magnitude head.

### 3) Tail-aware loss for heavy-tailed variables

Problem: maxima are under-predicted because MSE favors the mean.

Plan:
- For `cpool`, `deadstemc`, `deadcrootc`, use log-space loss:
  - train on `log1p(y)` or apply a Huber loss on `log1p(y)`.
- Keep standard loss for less heavy-tailed variables.

### 4) Rebalanced sampling for non-zero targets

Problem: non-zero samples are rare; model learns to predict near-zero.

Plan:
- Oversample non-zero samples for the target variables.
- Or use per-sample weights based on magnitude or zero/non-zero mask.

### 5) Variable-specific activation

Problem: a single activation (ReLU/abs/softplus) does not fit all variables.

Plan:
- Keep `abs` for highly sparse pools (reduces hard zeroing).
- Keep `relu` for large-range variables where `abs` inflates maxima.
- Apply activation per variable slice in the PFT1D head.

### New Option: Push the max (tail emphasis)

Add these three options specifically for recovering extreme values:

1) Use tail-aware + higher weight for heavy-tail variables
   - Keep log1p loss, but increase its contribution for
     `cpool`, `deadstemc`, `deadcrootc`, `livestemc`, `livecrootc`.

2) Add per-variable loss scaling or oversample high-value samples
   - Scale loss by variable or by target magnitude to emphasize the tail.
   - Or oversample rows where targets exceed a high-value threshold.

3) Switch to quantile or Huber loss for heavy-tail variables
   - Quantile loss targets upper quantiles directly.
   - Huber loss reduces sensitivity to extreme outliers while still
     fitting large values better than pure MSE.

## Evaluation checklist

- Compare R2 / relative RMSE / relative MAE against baseline.
- Compare zero fraction (GT vs pred) for each variable.
- Check max and p95 values to verify tail recovery.
- Inspect top-bad plots to confirm qualitative improvements.

## Suggested experiment order

1) Variable-specific sparsity weights (small change, easy to test).
2) Tail-aware loss for heavy-tailed variables.
3) Rebalanced sampling.
4) Variable-specific activation.
5) Zero-inflated two-head modeling (largest change).
