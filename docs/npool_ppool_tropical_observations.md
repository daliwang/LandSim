# Tropical NPOOL/PPOOL Observations and Suggestions

## Background values (10 for npool, 1 for ppool)

**Does this codebase use 10 as background for npool and 1 for ppool?**
- **No.** The training script and data loaders do **not** initialize or fill npool with 10 or ppool with 1. NaN/Inf in PFT1D data are replaced with **0** (`nan=0.0` in `data_loader_individual.py`); there is no special case for npool/ppool.
- **Where do 10 and 1 come from?** They come from the **training targets** (ELM/TRENDY output in the PKL files). For many PFTs, the **ground-truth** Y_npool and Y_ppool in the dataset are near-constant at ~10 and ~1. So the "background" is in the **source data** (likely ELM default or typical values for those pool variables), not from our initialization.

**Training dataset (ground truth) counts:** Run `scripts/analyze_npool_ppool_special_values.py` on your training PKL path. Example results on Trendy_1_data_CNP (20,975 grid cells, 16 PFTs):
- **npool == 10**: 80.7% of all (cell, PFT) pairs have this value; 32.8% of grid cells have *all* 16 PFTs equal to 10; 100% of cells have *at least one* PFT with npool == 10.
- **ppool == 1**: Same statistics (80.7% of pairs, 32.8% of cells all-1, 100% any).
- **Both**: 6,877 grid cells (32.8%) have all PFTs with npool==10 and ppool==1.
- Per-PFT counts and full report: `docs/npool_ppool_special_values_report.json`.

**Validation excluding special-value grid cells:** Run `scripts/validation_npool_ppool_exclude_special.py <run_dir>` to compute R² and RMSE for npool and ppool when excluding grid cells where *all* 16 PFTs have npool==10 and ppool==1. Example (run_20260226_114546_nofilter, test set 4,166 cells, 33.5% special):
- **NPOOL**: R² (all cells) ≈ 0.02 → R² (excl. special) ≈ **0.41**; RMSE 8.83 → 8.37.
- **PPOOL**: R² (all cells) ≈ -1.58 → R² (excl. special) ≈ **-0.51**; RMSE 0.87 → 0.82.
So on non-constant cells, npool prediction is moderate and ppool is still poor but much less bad than when including the constant cells.

**Per-PFT validation (recommended):** The special values (10 for npool, 1 for ppool) should be treated **per PFT**. For each PFT k, only grid cells where **PCT_NAT_PFT_k > 0** (that PFT is present) should be used when computing prediction quality for that PFT; other cells are filled with 10/1 and should not count. Run `scripts/validation_npool_ppool_per_pft.py <run_dir>` to get per-PFT R² and RMSE using this rule. Example (run_20260226_114546_nofilter): mean R² over PFTs (with ≥10 valid cells) is **~0.92** for npool and **~0.93** for ppool — prediction quality is good when evaluated only where each PFT is present.

## Scope
- Dataset: tropical-only subset (Latitude between -23.5 and 23.5).
- Source: Trendy_1 training data (21 PKL files).
- Samples: 4,030 tropical rows.
- Variables: `Y_npool`, `Y_ppool` (PFT1-16).

## Key Observations (Tropical-Only)

### 1) NPOOL is mostly near-constant with a few heavy-tail PFTs
- Many PFTs have median and upper quantiles near 10, indicating a near-constant target:
  - PFT02/03/05/07/08/09/11/12/16: median ~10, q90 ~10.
- A few PFTs show long-tailed distributions:
  - PFT04: max 67.6, q90 53.9
  - PFT06: max 60.5, q90 46.0
  - PFT10: max 106, q90 24.8
  - PFT13: max 88.9, q90 51.3
  - PFT14: max 151, q90 111
  - PFT15: max 87.7, q90 59.1

### 2) PPOOL is mostly near-constant at 1 with a few heavy-tail PFTs
- Many PFTs have median and upper quantiles near 1:
  - PFT02/03/05/07/08/09/11/12/16: median ~1, q90 ~1.
- Long-tailed PFTs:
  - PFT04: max 3.19, q90 2.47
  - PFT10: max 6.02, q90 1.53
  - PFT13: max 4.88, q90 3.16
  - PFT14: max 8.16, q90 6.27
  - PFT15: max 5.8, q90 4.14

### 3) Why performance is bad for NPOOL/PPOOL
- The targets are dominated by near-constant values (10 or 1), which makes the model learn a constant baseline.
- Tail PFTs are rare but have much larger values; standard losses underweight these rare large values.
- A shared head across all PFTs amplifies the dominance of constant PFTs and suppresses tail behavior.

## Suggestions to Improve NPOOL/PPOOL

### A) Tail-aware loss only for tail PFTs
- Apply tail-aware loss (log1p_huber or quantile) only to the tail PFTs:
  - NPOOL tail PFTs: 04, 06, 10, 13, 14, 15
  - PPOOL tail PFTs: 04, 10, 13, 14, 15
- Keep standard loss for constant PFTs to avoid destabilizing the baseline.

### B) PFT-specific weighting (targeted, not global)
- Increase weights only for the tail PFTs above.
- Avoid global increases; earlier runs showed overall regression when all weights were raised.

### C) Sampling strategy for tail PFTs
- Oversample tropical rows where NPOOL/PPOOL exceed the p90 threshold for tail PFTs.
- This increases exposure to large values without changing the loss for all PFTs.

### D) Optional: per-PFT head or conditional scaling
- Separate small head for NPOOL/PPOOL by PFT group:
  - Constant PFTs vs tail PFTs.
- This reduces gradient dominance from near-constant PFTs.

## Practical Next Step (Minimal Change)
- Keep experiment2 settings.
- Add tail-aware weights only for tail PFTs listed above.
- Do not increase weights for constant PFTs.

