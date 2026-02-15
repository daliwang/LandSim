# Tropical NPOOL/PPOOL Observations and Suggestions

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

