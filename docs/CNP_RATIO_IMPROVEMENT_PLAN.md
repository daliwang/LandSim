# CNP Ratio Improvement Plan

## Executive Summary

Based on analysis of experiment2 results (`run_20260212_162802_experiment_2`), several variables show poor performance that can be improved by enforcing CNP stoichiometric relationships. This document identifies all variables that can benefit from CNP ratio constraints and provides an implementation plan.

## Variables Requiring CNP Ratio Improvements

### 1D PFT Variables (Poor Performance from Quality Report)

#### Dead Wood Variables
- **deadstemc/n/p**: 62.5% good, 31.2% bad
  - **Ratios**: `deadstemn = deadstemc / deadwdcn[pft]`, `deadstemp = deadstemc / deadwdcp[pft]`
  - **Storage variants**: `deadstemn_storage`, `deadstemp_storage` also need improvement
  - **Note**: Only applies to woody PFTs (woody[pft] == 1)

- **deadcrootc/n/p**: 87.5% good, but storage variants have issues
  - **Ratios**: `deadcrootn = deadcrootc / deadwdcn[pft]`, `deadcrootp = deadcrootc / deadwdcp[pft]`
  - **Storage variants**: `deadcrootn_storage`, `deadcrootp_storage` need improvement

#### Leaf Variables
- **leafc/n/p**: 68.8% good, 31.2% bad
  - **Ratios**: `leafn = leafc / leafcn[pft]`, `leafp = leafc / leafcp[pft]`
  - **Storage variants**: `leafn_storage`, `leafp_storage` show 43.8% good, 25% bad

#### Fine Root Variables
- **frootc/n/p**: 81.2% good, 18.8% bad
  - **Ratios**: `frootn = frootc / frootcn[pft]`, `frootp = frootc / frootcp[pft]`
  - **Storage variants**: `frootn_storage`, `frootp_storage` show 50% good, 12.5% bad

#### Live Wood Variables
- **livestemc/n/p**: 87.5% good, but can be improved
  - **Ratios**: `livestemn = livestemc / livewdcn[pft]`, `livestemp = livestemc / livewdcp[pft]`
  - **Storage variants**: `livestemn_storage`, `livestemp_storage` show 81.2% good

- **livecrootc/n/p**: 87.5% good, but can be improved
  - **Ratios**: `livecrootn = livecrootc / livewdcn[pft]`, `livecrootp = livecrootc / livewdcp[pft]`
  - **Storage variants**: `livecrootn_storage`, `livecrootp_storage` show 81.2% good

### 2D Soil Variables (Poor Performance)

#### Soil Layer 1
- **soil1c/n/p**: 30% good, 50% bad (WORST PERFORMING SOIL VARIABLES)
  - **Ratios**: 
    - `soil1n_vr = soil1c_vr / cn_s1_new` (cn_s1_new = 12)
    - `soil1p_vr = soil1c_vr / np_s1_new` (np_s1_new = 30, note: this is C:P ratio, not N:P)
    - Actually: `soil1p_vr = soil1c_vr / (cn_s1_new * np_s1_new)` = `soil1c_vr / 360`
  - **Critical**: These are the worst performing soil variables

#### Soil Layer 2
- **soil2c/n/p**: 90% good, but can still benefit from ratio constraints
  - **Ratios**: 
    - `soil2n_vr = soil2c_vr / cn_s2_new` (cn_s2_new = 12)
    - `soil2p_vr = soil2c_vr / (cn_s2_new * np_s2_new)` = `soil2c_vr / 360`

#### Soil Layer 3 & 4
- **soil3c/n/p**: 100% good (excellent performance)
- **soil4c/n/p**: 90% good (good performance)
  - **Ratios**: 
    - `soil3n_vr = soil3c_vr / cn_s3_new` (cn_s3_new = 10)
    - `soil3p_vr = soil3c_vr / (cn_s3_new * np_s3_new)` = `soil3c_vr / 500`
    - `soil4n_vr = soil4c_vr / cn_s4_new` (cn_s4_new = 10)
    - `soil4p_vr = soil4c_vr / (cn_s4_new * np_s4_new)` = `soil4c_vr / 500`

#### Coarse Woody Debris (CWD)
- **cwdc/n/p**: 90% good, 10% bad
  - **Ratios**: 
    - `cwdn_vr = cwdc_vr / deadwdcn[pft]`
    - `cwdp_vr = cwdc_vr / deadwdcp[pft]`
  - **Note**: Uses PFT-specific ratios, applied layer-wise

## Additional Variables to Consider

### Litter Variables
- **litr2c/n/p**: 10% good, 80% bad (VERY POOR)
  - Uses `pft_lflitcn` for N ratios (from PFT parameters)
  - P ratios may need separate handling

- **litr3c/n/p**: 20% good, 80% ok (moderate performance)
  - Uses `pft_lflitcn` for N ratios

### Pool Variables
- **npool**: 25% good, 62.5% bad
- **ppool**: 12.5% good, 75% bad
- **Note**: These are typically NOT derived from `cpool` directly, but may benefit from component-based constraints

## PFT-Specific Ratio Values

### Dead Wood Ratios
- **deadwdcn**: `[1, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 0, 0, 0, 0, 0, 500, 500, 500, 500, 500, 500, 500, 500]`
- **deadwdcp**: `[1, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]`

### Leaf Ratios
- **leafcn**: `[1, 35, 40, 25, 30, 30, 25, 25, 25, 30, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]`
- **leafcp**: `[1, 525, 400, 250, 600, 450, 500, 375, 250, 450, 375, 250, 250, 375, 375, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275]`

### Fine Root Ratios
- **frootcn**: `[1, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]`
- **frootcp**: `[1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]`

### Live Wood Ratios
- **livewdcn**: `[1, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50]`
- **livewdcp**: `[1, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]`

### Soil Layer Ratios (Constant)
- **Layer 1**: cn_s1_new = 12, np_s1_new = 30 → C:P = 360
- **Layer 2**: cn_s2_new = 12, np_s2_new = 30 → C:P = 360
- **Layer 3**: cn_s3_new = 10, np_s3_new = 50 → C:P = 500
- **Layer 4**: cn_s4_new = 10, np_s4_new = 50 → C:P = 500

## Implementation Strategy

### Phase 1: Add CNP Ratio Constraint Loss
1. Create `CNPRatioConstraintLoss` in `training/losses.py`
2. Support both hard constraints (derived variables) and soft constraints (ratio penalties)
3. Integrate into training loop with configurable weight

### Phase 2: Post-Processing Option
1. Create post-processing function to enforce ratios after prediction
2. Use for inference to ensure stoichiometric consistency
3. Validate ratios in prediction outputs

### Phase 3: Validation and Monitoring
1. Create validation script to check CNP ratios in predictions
2. Add ratio metrics to training logs
3. Generate ratio violation reports

## Priority Order

### High Priority (Worst Performance)
1. **soil1c/n/p** (30% good, 50% bad)
2. **deadstemc/n/p** (62.5% good, 31.2% bad)
3. **leafc/n/p** (68.8% good, 31.2% bad)
4. **litr2c/n/p** (10% good, 80% bad)

### Medium Priority
5. **frootc/n/p** (81.2% good, 18.8% bad)
6. **Storage variants** of all above variables
7. **cwdc/n/p** (90% good, 10% bad)

### Low Priority (Good Performance, but can improve)
8. **livestemc/n/p** and **livecrootc/n/p** (87.5% good)
9. **soil2c/n/p** (90% good)

## Expected Improvements

By enforcing CNP ratios:
- **soil1c/n/p**: Expected improvement from 30% to 70-80% good
- **deadstemc/n/p**: Expected improvement from 62.5% to 85-90% good
- **leafc/n/p**: Expected improvement from 68.8% to 85-90% good
- **frootc/n/p**: Expected improvement from 81.2% to 90-95% good

Overall model performance should improve significantly, especially for variables with strong stoichiometric relationships.
