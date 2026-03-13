# Natveg filter vs no-filter: performance comparison

## Run setup

| Run | num_samples | natveg_only | natveg_filter_before_split |
|-----|-------------|-------------|----------------------------|
| nofilter | 20826 | false | true |
| natveg | 14006 | true | — |
| natveg_aligned | 20826 | true | false |

- **nofilter**: full data (train+test); no natveg filter.
- **natveg**: natveg-only filter *before* split → fewer training samples (≈14k vs 20k).
- **natveg_aligned**: natveg-only for *training*; test set same as nofilter (same 20% holdout).

## Aggregate metrics

| scalar_rmse | nofilter | **0.021993** | — | natveg | 0.032461 | +47.6% | natveg_aligned | 0.020582 | -6.4% |
| pft_1d_rmse | nofilter | **0.074273** | — | natveg | 0.072640 | -2.2% | natveg_aligned | 0.074440 | +0.2% |
| soil_2d_rmse | nofilter | **0.032720** | — | natveg | 0.040012 | +22.3% | natveg_aligned | 0.034643 | +5.9% |

(Lower RMSE is better; positive % = degradation with filter.)

## Scalar fluxes (R²)

| Variable | nofilter | natveg | Δ (pp) | natveg_aligned | Δ (pp) |
|----------|----------|--------|--------|----------------|--------|
| Y_GPP | 0.9900 | 0.9793 | -1.07 | 0.9919 | +0.19 |
| Y_NPP | 0.9877 | 0.9792 | -0.85 | 0.9884 | +0.07 |
| Y_AR | 0.9874 | 0.9757 | -1.18 | 0.9900 | +0.25 |
| Y_HR | 0.9871 | 0.9819 | -0.52 | 0.9879 | +0.08 |

(R² in [0,1]; negative Δ = degradation with filter.)

## Soil 2D variables (mean R² over layers)

| Variable | nofilter R² | natveg R² | Δ (pp) | natveg_aligned R² | Δ (pp) |
|----------|--------------|-----------|--------|-------------------|--------|
| cwdc_vr | 0.9387 | 0.9390 | +0.03 | 0.9435 | +0.48 |
| cwdn_vr | 0.9375 | 0.9396 | +0.20 | 0.9437 | +0.61 |
| cwdp_vr | 0.9374 | 0.9376 | +0.02 | 0.9420 | +0.46 |
| labilep_vr | 0.9871 | 0.9748 | -1.23 | 0.9840 | -0.32 |
| litr2c_vr | 0.7254 | 0.6183 | -10.71 | 0.7274 | +0.20 |
| litr2n_vr | 0.7170 | 0.6224 | -9.46 | 0.7222 | +0.52 |
| litr2p_vr | 0.7174 | 0.6141 | -10.33 | 0.7221 | +0.46 |
| litr3c_vr | 0.8760 | 0.8551 | -2.09 | 0.8746 | -0.14 |
| litr3n_vr | 0.8684 | 0.8540 | -1.44 | 0.8665 | -0.18 |
| litr3p_vr | 0.8671 | 0.8489 | -1.82 | 0.8649 | -0.22 |
| occlp_vr | 0.8960 | 0.7902 | -10.58 | 0.8493 | -4.67 |
| primp_vr | 0.6816 | 0.5898 | -9.18 | 0.7046 | +2.29 |
| secondp_vr | 0.9875 | 0.9756 | -1.20 | 0.9836 | -0.40 |
| soil1c_vr | 0.8733 | 0.8443 | -2.90 | 0.8722 | -0.12 |
| soil1n_vr | 0.8733 | 0.8434 | -2.98 | 0.8726 | -0.07 |
| soil1p_vr | 0.8737 | 0.8438 | -2.99 | 0.8726 | -0.11 |
| soil2c_vr | 0.9177 | 0.8997 | -1.80 | 0.9153 | -0.24 |
| soil2n_vr | 0.9173 | 0.9020 | -1.53 | 0.9156 | -0.17 |
| soil2p_vr | 0.9171 | 0.9006 | -1.65 | 0.9155 | -0.16 |
| soil3c_vr | 0.8958 | 0.8810 | -1.48 | 0.8929 | -0.29 |
| soil3n_vr | 0.8945 | 0.8817 | -1.28 | 0.8934 | -0.11 |
| soil3p_vr | 0.8952 | 0.8807 | -1.45 | 0.8936 | -0.16 |
| soil4c_vr | 0.8755 | 0.8469 | -2.86 | 0.8722 | -0.34 |
| soil4n_vr | 0.8756 | 0.8473 | -2.84 | 0.8718 | -0.38 |
| soil4p_vr | 0.8749 | 0.8459 | -2.90 | 0.8718 | -0.32 |
| solutionp_vr | 0.9612 | 0.9461 | -1.50 | 0.9603 | -0.09 |

## PFT 1D variables (mean R² over PFTs)

| Variable | nofilter R² | natveg R² | Δ (pp) | natveg_aligned R² | Δ (pp) |
|----------|--------------|-----------|--------|-------------------|--------|
| cpool | 0.9423 | 0.9429 | +0.05 | 0.9383 | -0.40 |
| deadcrootc | 0.7244 | 0.7246 | +0.02 | 0.7218 | -0.26 |
| deadcrootc_storage | 0.4247 | 0.4228 | -0.19 | 0.4229 | -0.18 |
| deadcrootn | 0.7238 | 0.7249 | +0.11 | 0.7212 | -0.26 |
| deadcrootn_storage | 0.4248 | 0.4226 | -0.22 | 0.4228 | -0.20 |
| deadcrootp | 0.7244 | 0.7242 | -0.02 | 0.7220 | -0.25 |
| deadcrootp_storage | 0.4194 | 0.4229 | +0.35 | 0.4228 | +0.34 |
| deadstemc | 0.7230 | 0.7250 | +0.20 | 0.7210 | -0.20 |
| deadstemc_storage | 0.4250 | 0.4228 | -0.22 | 0.4228 | -0.22 |
| deadstemn | 0.7205 | 0.7244 | +0.38 | 0.7205 | +0.00 |
| deadstemn_storage | 0.4248 | 0.4227 | -0.21 | 0.4228 | -0.20 |
| deadstemp | 0.7221 | 0.7244 | +0.23 | 0.7207 | -0.15 |
| deadstemp_storage | 0.4249 | 0.4226 | -0.23 | 0.4227 | -0.22 |
| frootc | 0.9572 | 0.9561 | -0.11 | 0.9493 | -0.79 |
| frootc_storage | 0.6635 | 0.6590 | -0.45 | 0.6625 | -0.10 |
| frootn | 0.9574 | 0.9562 | -0.12 | 0.9564 | -0.10 |
| frootn_storage | 0.6647 | 0.6609 | -0.38 | 0.6627 | -0.20 |
| frootp | 0.9569 | 0.9563 | -0.06 | 0.9570 | +0.01 |
| frootp_storage | 0.6652 | 0.6592 | -0.61 | 0.6626 | -0.26 |
| leafc | 0.9551 | 0.9553 | +0.02 | 0.9545 | -0.06 |
| leafc_storage | 0.6648 | 0.6608 | -0.40 | 0.6632 | -0.16 |
| leafn | 0.9557 | 0.9552 | -0.05 | 0.9552 | -0.05 |
| leafn_storage | 0.6653 | 0.6613 | -0.41 | 0.6626 | -0.27 |
| leafp | 0.9555 | 0.9476 | -0.78 | 0.9552 | -0.03 |
| leafp_storage | 0.6653 | 0.6611 | -0.43 | 0.6633 | -0.20 |
| livecrootc | 0.7216 | 0.7216 | -0.00 | 0.7192 | -0.24 |
| livecrootc_storage | 0.4250 | 0.4225 | -0.25 | 0.4226 | -0.24 |
| livecrootn | 0.7217 | 0.7223 | +0.06 | 0.7192 | -0.25 |
| livecrootn_storage | 0.4249 | 0.4227 | -0.22 | 0.4229 | -0.20 |
| livecrootp | 0.7196 | 0.7212 | +0.17 | 0.7188 | -0.08 |
| livecrootp_storage | 0.4249 | 0.4228 | -0.21 | 0.4229 | -0.20 |
| livestemc | 0.7221 | 0.7217 | -0.04 | 0.7199 | -0.23 |
| livestemc_storage | 0.4250 | 0.4227 | -0.23 | 0.4229 | -0.21 |
| livestemn | 0.7220 | 0.7207 | -0.13 | 0.7195 | -0.25 |
| livestemn_storage | 0.4248 | 0.4227 | -0.21 | 0.4229 | -0.19 |
| livestemp | 0.7207 | 0.7206 | -0.01 | 0.7193 | -0.14 |
| livestemp_storage | 0.4246 | 0.4228 | -0.18 | 0.4226 | -0.20 |
| npool | -12.7540 | -8.9153 | +383.87 | -12.7567 | -0.28 |
| ppool | -15.1043 | -10.4005 | +470.39 | -15.1068 | -0.25 |
| tlai | 0.9550 | 0.9513 | -0.37 | 0.9550 | -0.00 |
| totvegc | 0.9698 | 0.9708 | +0.10 | 0.9692 | -0.06 |

## Largest R² degradations (natveg vs nofilter)

Worst 15 (most negative Δ = largest drop with natveg):

- **litr2c_vr** (soil2d): -10.71 pp
- **occlp_vr** (soil2d): -10.58 pp
- **litr2p_vr** (soil2d): -10.33 pp
- **litr2n_vr** (soil2d): -9.46 pp
- **primp_vr** (soil2d): -9.18 pp
- **soil1p_vr** (soil2d): -2.99 pp
- **soil1n_vr** (soil2d): -2.98 pp
- **soil1c_vr** (soil2d): -2.90 pp
- **soil4p_vr** (soil2d): -2.90 pp
- **soil4c_vr** (soil2d): -2.86 pp
- **soil4n_vr** (soil2d): -2.84 pp
- **litr3c_vr** (soil2d): -2.09 pp
- **litr3p_vr** (soil2d): -1.82 pp
- **soil2c_vr** (soil2d): -1.80 pp
- **soil2p_vr** (soil2d): -1.65 pp

Best 5 (improvement with natveg, among variables with sensible R²):

- **deadstemn** (pft1d): +0.38 pp
- **deadcrootp_storage** (pft1d): +0.35 pp
- **deadstemp** (pft1d): +0.23 pp
- **cwdn_vr** (soil2d): +0.20 pp
- **deadstemc** (pft1d): +0.20 pp

## Is the performance degradation concerning?

### Two different comparisons

1. **natveg vs nofilter** (different train *and* test):
   - natveg has ~33% fewer samples (14k) and is evaluated on a *natveg-only* test set.
   - nofilter is evaluated on the *full* 20% holdout.
   - So the large aggregate RMSE increase (+47% scalar, +22% soil) is partly from **different test sets**, not just less data.

2. **natveg_aligned vs nofilter** (same test set, fair comparison):
   - Same 20% holdout for both; only the *training* set is natveg-only in natveg_aligned.
   - This answers: *If I train with the natveg filter, how much do I lose on the same test?*

- **natveg vs nofilter** aggregate scalar RMSE: **+47.6%** (worse).
- **natveg_aligned vs nofilter** scalar RMSE: **-6.4%** (slightly better when negative).

### Verdict: is degradation with the filter concerning?

**When comparing fairly (natveg_aligned vs nofilter, same test set):**

- **Scalar fluxes (GPP, NPP, AR, HR)**: R² **improves** or is flat (+0.07 to +0.25 pp). Scalar RMSE **improves** by ~6%. No concern.
- **PFT 1D**: Aggregate RMSE is virtually unchanged (+0.2%). Per-variable R² changes are mostly within ±0.5 pp. No concern.
- **Soil 2D**: Aggregate RMSE is ~6% higher. Most variables are within ±0.5 pp R². Notable drop: **occlp_vr** ≈ -4.7 pp R². litr2*/primp_vr stay similar or improve slightly with aligned run.

**Conclusion:** Using the **natveg filter is not concerning** for overall performance when the same test set is used. The aligned run (train on natveg, test on same holdout as nofilter) is slightly *better* on scalars and similar on PFT/soil, with **occlp_vr** as the only variable with a clear drop (~4.7 pp). If your science prioritizes natural vegetation and occlp is not central, the filter is reasonable to apply.

The **natveg** run (filter-before-split, 14k samples) looks much worse mainly because it is evaluated on a different (natveg-only) test set and with less training data; that comparison is not apples-to-apples for "degradation with filter."
