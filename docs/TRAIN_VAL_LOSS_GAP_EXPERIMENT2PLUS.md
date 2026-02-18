# Why the Train–Validation Loss Gap Appears in Experiment 2+

## What You’re Seeing

- **Experiment 2** (run_20260212_162802): train and validation loss **track each other** and end **close** (e.g. ~0.93 vs ~0.93). *(Experiment 2 was run **without** CNP ratio constraints.)*
- **Experiment 2+** (run_20260216_120353): **large gap** — train ~0.054, validation ~1.11.

So the gap is **specific to the 2+ setup**; experiment 2 had no ratio constraint and still had train ≈ val.

## Why Experiment 2 Had No Gap (No Ratio Constraint)

Experiment 2 uses **more balanced variable weights**: C, N, and P get similar weights (e.g. leafc=6, leafn=6, leafp=6; deadstemc=6, deadstemn=6, deadstemp=5; soil1c_vr=7, soil1n_vr=7, soil1p_vr=7). The loss is spread across many variables, so:

- Train and validation loss move in a similar range.
- Even if the model overfits somewhat, no single group of variables dominates the loss, so both curves end around ~0.93.

## Why Experiment 2+ Has a Big Gap

Experiment 2+ uses **C emphasis**: much higher weights on C variables and lower on N/P (e.g. leafc=9 vs leafn/leafp=5; deadstemc=9 vs deadstemn=5, deadstemp=4; soil1c_vr=9 vs soil1n_vr/soil1p_vr=5). So:

- The loss is **dominated by C variables**. When the model fits the **training** C targets well, the weighted MSE drops a lot → train loss ~0.05.
- On **validation**, the same C variables may be harder (different samples, different distribution) or the model has overfitted to training C patterns, so the weighted loss stays high → val loss ~1.11.
- Result: **large train–val gap** caused by **C emphasis** making the loss very sensitive to C fit on the training set, without the same gain on validation.

So the gap is due to **variable weighting (C emphasis)**, not the presence or absence of the CNP ratio constraint.

## What to Do

1. **Early stopping**  
   Stop when validation loss stops improving (e.g. patience 5–10). Avoid training many extra epochs once val has plateaued and only train keeps dropping.

2. **Soften C emphasis**  
   Reduce the spread between C and N/P weights (e.g. C weights a bit lower, N/P a bit higher) so the loss is less dominated by C and train/val behave more similarly.

3. **Stronger regularization**  
   Slightly increase weight decay or dropout to limit overfitting to the high-weight C variables.

4. **Optional: small ratio constraint as regularizer**  
   Enabling a **small** CNP ratio constraint weight (e.g. 0.2–0.5) can add a mild regularizer and sometimes bring train and val closer, but it is not the reason experiment 2 had no gap (experiment 2 was run without it).

5. **Accept the gap if validation metrics are acceptable**  
   If you use **derive N/P from C at inference**, what matters is validation **metrics** (e.g. R², RMSE on C and derived N/P). A train–val loss gap can be acceptable if those metrics and plots look good.

## Summary

- **Experiment 2:** No ratio constraint; **balanced** variable weights → loss spread across C/N/P → train and val both ~0.93.
- **Experiment 2+:** **C emphasis** (high C weights, lower N/P) → loss dominated by C → model fits train C very well (train ~0.05) but val doesn’t (val ~1.11) → gap.
- **Fix:** Early stopping, softer C weights, and/or stronger regularization; optionally a small ratio constraint as extra regularizer.

## Bugfix: Validation loss used unmasked PFT1D loss (fixed)

Previously, **training** applied the PFT presence mask when computing PFT1D loss (only present PFTs counted), while **validation** did not (all 16 PFTs counted). That made validation loss systematically higher and not comparable to training loss. Validation now uses the same masked PFT1D loss as training (`pft_loss_mask` in `validate_epoch`), so train and val loss are comparable. Re-run training to see the corrected validation curve.
