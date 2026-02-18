# CNP Stoichiometric Relationships

This document describes the CNP (Carbon-Nitrogen-Phosphorus) stoichiometric relationships used to derive variables from each other using PFT-specific ratios.

**Source File**: The actual ratio values are documented in [`model_variable_quantities.txt`](../model_variable_quantities.txt) in the project root directory.

## Overview

Many CNP variables can be derived from their carbon counterparts using PFT-specific C:N and C:P ratios. These ratios are stored as PFT parameters in the model inputs and are defined in `model_variable_quantities.txt`.

## PFT Parameters (C:N and C:P Ratios)

The following PFT parameters represent C:N and C:P mass ratios. All values are PFT-specific (25 PFTs, indexed 0-24):

### Dead Wood Ratios
- **`deadwdcn`** (Dead wood C:N ratio): Units = gC/gN
  - Values: `1, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 0, 0, 0, 0, 0, 500, 500, 500, 500, 500, 500, 500, 500`
  - Used for: `deadstemc → deadstemn`, `deadcrootc → deadcrootn`
  - **Important**: Only applies to woody PFTs (woody = 1). For non-woody PFTs (woody = 0), CNP values are forced to 0.

- **`deadwdcp`** (Dead wood C:P ratio): Units = gC/gP
  - Values: `1, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000`
  - Used for: `deadstemc → deadstemp`, `deadcrootc → deadcrootp`

### Leaf Ratios
- **`leafcn`** (Leaf C:N ratio): Units = gC/gN
  - Values: `1, 35, 40, 25, 30, 30, 25, 25, 25, 30, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25`
  - Used for: `leafc → leafn`

- **`leafcp`** (Leaf C:P ratio): Units = gC/gP
  - Values: `1, 525, 400, 250, 600, 450, 500, 375, 250, 450, 375, 250, 250, 375, 375, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275`
  - Used for: `leafc → leafp`

### Fine Root Ratios
- **`frootcn`** (Fine root C:N ratio): Units = gC/gN
  - Values: `1, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42`
  - Used for: `frootc → frootn`

- **`frootcp`** (Fine root C:P ratio): Units = gC/gP
  - Values: `1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000`
  - Used for: `frootc → frootp`

### Live Wood Ratios
- **`livewdcn`** (Live wood C:N ratio): Units = gC/gN
  - Values: `1, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50`
  - Used for: `livestemc → livestemn`, `livecrootc → livecrootn`

- **`livewdcp`** (Live wood C:P ratio): Units = gC/gP
  - Values: `1, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000`
  - Used for: `livestemc → livestemp`, `livecrootc → livecrootp`

### Soil Layer Ratios
- **Soil Layer 1**: `cn_s1_new = 12`, `np_s1_new = 30`
  - Used for: `soil1c_vr → soil1n_vr`, `soil1c_vr → soil1p_vr`

- **Soil Layer 2**: `cn_s2_new = 12`, `np_s2_new = 30`
  - Used for: `soil2c_vr → soil2n_vr`, `soil2c_vr → soil2p_vr`

- **Soil Layer 3**: `cn_s3_new = 10`, `np_s3_new = 50`
  - Used for: `soil3c_vr → soil3n_vr`, `soil3c_vr → soil3p_vr`

- **Soil Layer 4**: `cn_s4_new = 10`, `np_s4_new = 50`
  - Used for: `soil4c_vr → soil4n_vr`, `soil4c_vr → soil4p_vr`

## Conversion Formulas

### 1D PFT Variables

**Dead Wood:**
```python
# For woody PFTs only (woody[pft] == 1)
deadstemn = deadstemc / deadwdcn[pft]
deadcrootn = deadcrootc / deadwdcn[pft]
deadstemp = deadstemc / deadwdcp[pft]
deadcrootp = deadcrootc / deadwdcp[pft]

# For non-woody PFTs (woody[pft] == 0), force to 0
deadstemn = 0
deadcrootn = 0
deadstemp = 0
deadcrootp = 0
```

**Live Wood:**
```python
livestemn = livestemc / livewdcn[pft]
livecrootn = livecrootc / livewdcn[pft]
livestemp = livestemc / livewdcp[pft]
livecrootp = livecrootc / livewdcp[pft]
```

**Leaf:**
```python
leafn = leafc / leafcn[pft]
leafp = leafc / leafcp[pft]
```

**Fine Root:**
```python
frootn = frootc / frootcn[pft]
frootp = frootc / frootcp[pft]
```

### 2D Variables (Litter and Soil)

**Coarse Woody Debris (layer-wise):**
```python
cwdn_vr[layer] = cwdc_vr[layer] / deadwdcn[pft]
cwdp_vr[layer] = cwdc_vr[layer] / deadwdcp[pft]
```

**Litter (layer-wise, using `pft_lflitcn` from PFT parameters):**
```python
# Note: pft_lflitcn values come from PFT parameters, not model_variable_quantities.txt
litr1n_vr[layer] = litr1c_vr[layer] / pft_lflitcn[pft]
litr2n_vr[layer] = litr2c_vr[layer] / pft_lflitcn[pft]
litr3n_vr[layer] = litr3c_vr[layer] / pft_lflitcn[pft]
```

**Soil (layer-wise, constant ratios per soil layer):**
```python
# Soil Layer 1
soil1n_vr[layer] = soil1c_vr[layer] / cn_s1_new  # cn_s1_new = 12
soil1p_vr[layer] = soil1c_vr[layer] / np_s1_new  # np_s1_new = 30

# Soil Layer 2
soil2n_vr[layer] = soil2c_vr[layer] / cn_s2_new  # cn_s2_new = 12
soil2p_vr[layer] = soil2c_vr[layer] / np_s2_new  # np_s2_new = 30

# Soil Layer 3
soil3n_vr[layer] = soil3c_vr[layer] / cn_s3_new  # cn_s3_new = 10
soil3p_vr[layer] = soil3c_vr[layer] / np_s3_new  # np_s3_new = 50

# Soil Layer 4
soil4n_vr[layer] = soil4c_vr[layer] / cn_s4_new  # cn_s4_new = 10
soil4p_vr[layer] = soil4c_vr[layer] / np_s4_new  # np_s4_new = 50
```

## Important Notes

1. **PFT Indexing**: All PFT arrays are indexed 0-24 (25 PFTs total). The first value (index 0) is typically 1 or a special value.

2. **Woody PFT Filtering**: For dead wood variables, ratios only apply when `woody[pft] == 1`. For non-woody PFTs (`woody[pft] == 0`), the derived N and P values should be forced to 0.

3. **Storage Variables**: Storage variables (e.g., `deadstemc_storage`, `deadstemn_storage`) follow the same relationships as their non-storage counterparts.

4. **Zero Ratios**: When a ratio is 0 (e.g., `deadwdcn[pft] == 0`), it indicates that the PFT doesn't have that component, and derived values should be 0.

5. **Layer-wise Application**: For 2D variables (litter, soil), the conversion is applied layer-by-layer. Soil ratios are constant across all layers within each soil layer type.

6. **Pool Variables**: 
   - `npool` and `ppool` are typically **not** derived from `cpool` directly, as they represent different pools
   - These are usually predicted separately or derived from component variables

## Usage Example

If you have predictions for `deadstemc` and want to derive `deadstemn`:

```python
import numpy as np

# Load ratio values from model_variable_quantities.txt
# deadwdcn values for 25 PFTs
deadwdcn = np.array([1, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 
                     0, 0, 0, 0, 0, 500, 500, 500, 500, 500, 500, 500, 500])

# Assuming you have:
# - deadstemc predictions (shape: [samples, pfts])
# - woody flags (shape: [pfts]) indicating which PFTs are woody

# Derive deadstemn
deadstemn = np.zeros_like(deadstemc)
for pft_idx in range(deadstemc.shape[1]):
    if woody[pft_idx] == 1 and deadwdcn[pft_idx] > 0:
        deadstemn[:, pft_idx] = deadstemc[:, pft_idx] / deadwdcn[pft_idx]
    else:
        deadstemn[:, pft_idx] = 0  # Force to 0 for non-woody or zero-ratio PFTs
```

## Related Files

- **`model_variable_quantities.txt`**: Contains the actual ratio values used in conversions (see project root)
- **CNP_IO files** (e.g., `CNP_IO_updated9_cnpratio_reduced.txt`): List which variables are included in training
- **`config/training_config.py`**: Contains default PFT parameter lists and variable configurations
- **PFT parameters**: Loaded as model inputs during training (e.g., `pft_deadwdcn`, `pft_livewdcn`, `pft_leafcn`, `pft_frootcn`, `pft_lflitcn`)

## Reference

For the complete source of ratio values, see: [`model_variable_quantities.txt`](../model_variable_quantities.txt)
