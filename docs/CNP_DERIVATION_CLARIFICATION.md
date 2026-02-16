# CNP Derivation: Why "Train C Only" Didn't Help (and What To Do Instead)

## What You Tried

You trained with **CNP_IO_updated9_reduced.txt** (C variables only, 39 variables) at  
`cnp_results/run_20260212_102026_updated9_reduced`.

**Result**: Performance did **not** improve; it got **worse** than experiment2 (full 71 variables).

## Why C-Only Training Performed Worse

### 1. Multi-task learning helps C

When the model is trained on **all** CNP variables (C, N, P):

- The **shared representation** (encoder/backbone) gets a stronger learning signal.
- N and P targets are **auxiliary tasks** that regularize and improve the same features used for C.
- The model implicitly uses N/P patterns to learn better C (e.g. via shared gradients and feature reuse).

When you **remove** N and P from the training targets:

- The model loses that auxiliary signal.
- C is trained in isolation, with a smaller effective “task set.”
- Your comparison (and `COMPARISON_REDUCED_VS_COMPLETE.md`) shows that C predictions get **worse** (e.g. leafc 68.8% → 43.8% good, deadstemc 62.5% → 56.2% good, soil1c_vr 30% → 20% good).

So: **training “C only” weakens the representation and hurts C quality.** You didn’t miss something in the setup; the setup itself (fewer targets) is the reason.

### 2. Your own comparison doc says so

From `docs/COMPARISON_REDUCED_VS_COMPLETE.md`:

- Reduced list: **49.8% good**, **28.9% bad**
- Complete list: **58.4% good**, **22.0% bad**

Conclusion there: *“The reduced variable list degrades performance because the model loses implicit CNP relationship learning.”*

So the behavior you see (C-only not improving, and actually degrading) is expected and already documented.

## What *does* work: full training + derivation at inference

The approach that **does** improve things is:

1. **Train with the full variable list** (as in experiment2, 71 variables).
2. **Do not** change the training setup to “C only.”
3. **After** inference, **replace** the model’s N and P predictions with **derived** N/P from C (using stoichiometric ratios).

That is exactly **Option 3** in your comparison doc:

- *“Train with complete variable list”*
- *“Derive N and P from C predictions using stoichiometric ratios”*

So the “approach that will be better” is **not** “train C only,” but **“train full, derive N/P at inference.”**

## Why this is better

| Aspect | Train C only (reduced list) | Train full + derive N/P at inference |
|--------|----------------------------|--------------------------------------|
| C prediction quality | **Worse** (less multi-task signal) | **Better** (same as experiment2) |
| N/P quality | N/A (you derive them) | **Defined by C** (derived from C) |
| CNP ratios | Perfect (by construction) | **Perfect** (by derivation) |
| Training setup | New reduced config | **Unchanged** (experiment2) |

So:

- **Training**: keep the **full** list so C (and the shared backbone) keep benefiting from N/P.
- **Inference**: run your derivation script so that **final** N/P are always from C, giving perfect ratios and consistent stoichiometry.

You already validated this: when you ran the derivation script **on experiment2 outputs** (full model), you got:

- **0%** CNP ratio errors (perfect stoichiometry),
- **R² = 0.9863** for derived deadstemn vs ground truth.

So the “approach that will be better” is: **same training as experiment2, plus derivation at inference.** Not “train C only.”

## Did you miss something?

- **In the idea**: Yes. The better approach is **“derive N/P at inference”**, not **“train only C.”**
- **In the runs**: No. Your reduced run correctly shows that C-only training doesn’t help and can hurt. Your comparison doc already explains why.

## Recommended workflow

1. **Training**  
   - Keep using the **full** variable list (e.g. experiment2 config).  
   - Do **not** switch to CNP_IO_updated9_reduced.txt for training.

2. **Inference**  
   - Run the model as usual (full 71 outputs).  
   - Run the derivation script on the predictions to overwrite N/P with values derived from C.  
   - Use these **derived** N/P (and original C) for analysis, restart files, etc.

3. **Optional**  
   - Integrate the derivation step into the inference pipeline so that any run automatically gets derived N/P and perfect ratios.

## Summary

- **“Train C only”** (reduced list) **does not** improve things; it worsens C and overall performance because multi-task learning is lost.
- The approach that **does** improve things is: **train full (experiment2), then derive N/P from C at inference.**
- You didn’t miss something in the C-only run; the right move is to keep full training and add derivation as a post-processing step, not to reduce the variable list for training.
