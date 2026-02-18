# Tropical Bad-Variable Observations and General Suggestions

## Scope
- Dataset: tropical-only subset (Latitude between -23.5 and 23.5).
- Source: Trendy_1 training data (21 PKL files).
- Samples: 4,030 tropical rows.
- Variables analyzed: `npool`, `ppool`, `cpool`, `totvegc`, `leafc/n/p`, `deadstemc/n`, `primp_vr`, `litr2c/n/p_vr`, `soil1c/n/p_vr`.

## Tropical Distribution Summary (aggregate across PFTs/layers)
Values below are computed over all PFTs (for 1D PFT variables) or layers (for 2D soil variables).

- `Y_npool`: nonzero 46.38%, max 151, median(nz) 10, q90 38.7, q95 53.4
- `Y_ppool`: nonzero 46.37%, max 8.16, median(nz) 1, q90 1.9, q95 3.24
- `Y_cpool`: nonzero 9.83%, max 8.57e3, median(nz) 748, q90 2.59e3, q95 3.87e3
- `Y_totvegc`: nonzero 45.83%, max 6.95e4, median(nz) 1.1, q90 4.48e3, q95 1.83e4
- `Y_leafc`: nonzero 21.47%, max 811, median(nz) 1, q90 335, q95 420
- `Y_leafn`: nonzero 21.47%, max 32.4, median(nz) 0.0333, q90 12.1, q95 16.2
- `Y_leafp`: nonzero 21.47%, max 2.16, median(nz) 0.0025, q90 0.816, q95 1.17
- `Y_deadstemc`: nonzero 31.92%, max 5.13e4, median(nz) 1, q90 3.79e3, q95 1.61e4
- `Y_deadstemn`: nonzero 31.92%, max 103, median(nz) 0.002, q90 7.59, q95 32.2
- `Y_primp_vr`: nonzero 66.67%, max 1.88e3, median(nz) 184, q90 1.26e3, q95 1.26e3
- `Y_litr2c_vr`: nonzero 3.87%, max 7.98e3, median(nz) 27.6, q90 478, q95 807
- `Y_litr2n_vr`: nonzero 3.87%, max 167, median(nz) 0.48, q90 6.26, q95 13.2
- `Y_litr2p_vr`: nonzero 3.96%, max 9.6, median(nz) 0.0271, q90 0.511, q95 0.888
- `Y_soil1c_vr`: nonzero 3.74%, max 779, median(nz) 6.86, q90 72.2, q95 108
- `Y_soil1n_vr`: nonzero 3.74%, max 64.9, median(nz) 0.572, q90 6.01, q95 8.97
- `Y_soil1p_vr`: nonzero 4.01%, max 2.16, median(nz) 0.0163, q90 0.193, q95 0.288

## Why These Variables Perform Poorly (Patterns)
1. **High sparsity**: soil1* and litr2* are ~4% non‑zero, so the model mostly sees zeros.
2. **Heavy‑tail distributions**: cpool, deadstemc, totvegc have huge max vs median, so typical losses underpredict extremes.
3. **Near‑constant regimes**: npool ~10 and ppool ~1 for many PFTs; the model learns the constant and misses tails.
4. **Mixed scales**: leafp/leafn are tiny vs leafc/cpool; shared heads can bias toward larger scales.

## General Suggestions (Applies to All These Variables)

### A) Target tail behavior only where it exists
- Apply tail‑aware loss **only** to tail‑heavy variables:
  - Strong tails: `cpool`, `deadstemc`, `totvegc`, `primp_vr`, `litr2c_vr`, `soil1c_vr`
  - Moderate tails: `npool`, `ppool`, `deadstemn`, `leafc`
- Use `log1p_huber` or `log1p_quantile` to push higher values without exploding gradients.

### B) Separate sparse vs dense handling
- For very sparse 2D soil variables (`litr2*`, `soil1*`), use **zero‑inflated loss**:
  - Loss = BCE/hinge for zero vs non‑zero + regression loss for non‑zero.
- Alternatively, add a **non‑zero mask head** and only regress where mask=1.

### C) Per‑variable or per‑group weighting, not global
- Avoid global weight increases (past runs degraded overall quality).
- Increase weights only for a **short list of tail variables** and keep others at baseline.

### D) Sampling strategy
- Oversample tropical rows where target values are in the top 10% (tail enrichment).
- This is less disruptive than large loss weights.

### E) Scaling / normalization improvements
- For variables with extreme max/median ratios (cpool, deadstemc, totvegc), consider:
  - log1p transform before scaling
  - per‑variable robust scaling (median/IQR)

### F) Optional architecture tweaks
- Add **per‑variable heads** for the most problematic groups:
  - One head for soil1/litr2 (sparse 2D)
  - One head for pool variables (cpool/npool/ppool)

## Concrete Next Step (Minimal Risk)
- Keep experiment2 as baseline.
- Add tail‑aware loss **only** for `cpool`, `deadstemc`, `totvegc`, `primp_vr`, `litr2c_vr`, `soil1c_vr`.
- Keep `npool` and `ppool` weights unchanged for now, but add tail‑aware for their known tail PFTs (see `npool_ppool_tropical_observations.md`).

