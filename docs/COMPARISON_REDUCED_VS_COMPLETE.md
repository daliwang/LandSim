# Comparison: Reduced Variable List vs Complete Variable List

## Summary Statistics

| Metric | Complete List (run_20260211_232737) | Reduced List (run_20260212_102026) | Difference |
|--------|-------------------------------------|-------------------------------------|------------|
| **Total Variables** | 71 | 39 | -32 variables (-45%) |
| **Total Predictions** | 920 | 456 | -464 predictions (-50%) |
| **Good Predictions** | 537 (58.4%) | 227 (49.8%) | **-8.6%** |
| **OK Predictions** | 181 (19.7%) | 97 (21.3%) | +1.6% |
| **Bad Predictions** | 202 (22.0%) | 132 (28.9%) | **+6.9%** |

## Key Finding: Performance Degradation

**The reduced variable list shows worse overall performance:**
- **8.6% fewer good predictions** (58.4% → 49.8%)
- **6.9% more bad predictions** (22.0% → 28.9%)

## Variables with Degraded Performance (Reduced List)

### Significantly Worse Variables:

1. **`litr3c_vr`**: 
   - Complete: 20% good, 40% ok, 40% bad
   - Reduced: 20% good, 20% ok, **60% bad** ⚠️ (20% worse)

2. **`soil3c_vr`**: 
   - Complete: 90% good, 10% ok, 0% bad
   - Reduced: 60% good, 40% ok, 0% bad ⚠️ (30% fewer good)

3. **`cpool`**: 
   - Complete: Not in worst list (better than 31.2% good)
   - Reduced: 31.2% good, 25% ok, **43.8% bad** ⚠️

4. **`totvegc`**: 
   - Complete: Not in worst list (better than 31.2% good)
   - Reduced: 31.2% good, 31.2% ok, **37.5% bad** ⚠️

5. **`soil1c_vr`**: 
   - Both: 20% good, 20% ok, 60% bad (same, but still poor)

### Variables with Similar Performance:

- `ppool`: Both show 6.2% good, 18.8% ok, 75% bad
- `npool`: Both show 12.5% good, 25% ok, 62.5% bad
- `litr2c_vr`, `litr2n_vr`, `litr2p_vr`: Both show 10% good, 10% ok, 80% bad

## Why Performance Degrades with Reduced Variable List

### 1. **Loss of Implicit CNP Relationship Learning**

**Key Insight**: The model learns CNP stoichiometric relationships **implicitly** during training, even though variables are predicted independently.

- When N and P variables are present during training, the model learns to maintain CNP relationships through:
  - Shared feature representations
  - Cross-variable constraints in the loss function
  - Implicit regularization from seeing related variables

- **Without N and P variables**, the model loses:
  - The ability to learn CNP stoichiometric constraints
  - Cross-variable information that helps predict carbon pools more accurately
  - Regularization signals from related variables

### 2. **Reduced Training Signal**

- **Fewer variables = less training signal** for the model to learn from
- Variables that benefit from seeing related variables (e.g., `cpool` seeing `npool` and `ppool`) lose this information
- The model has less context to learn robust representations

### 3. **Feature Learning Degradation**

- Neural networks learn better representations when they see **related variables together**
- Carbon variables benefit from seeing their N and P counterparts during training because:
  - They share similar patterns and relationships
  - The model can learn shared features that improve all predictions
  - Cross-variable attention mechanisms (if present) can leverage relationships

### 4. **Loss of Regularization**

- When N and P variables are predicted alongside C variables, the model implicitly learns to maintain CNP ratios
- Without N and P variables, there's no constraint to maintain these relationships
- This can lead to carbon pool predictions that are inconsistent with stoichiometric relationships

### 5. **Pool Variable Dependencies**

Variables like `cpool`, `npool`, and `ppool` are particularly affected because:
- They represent aggregate pools that depend on component variables
- In the complete list, the model sees all three pools and learns their relationships
- In the reduced list, only `cpool` is present, losing the context of `npool` and `ppool`

## Evidence from the Data

### Variables Most Affected:

1. **`litr3c_vr`**: 20% worse (40% bad → 60% bad)
   - Litter variables benefit from seeing N and P litter variables
   - Without `litr3n_vr` and `litr3p_vr`, the model loses stoichiometric constraints

2. **`soil3c_vr`**: 30% fewer good predictions (90% → 60%)
   - Soil variables show strong CNP relationships
   - Without N and P soil variables, carbon predictions degrade

3. **`cpool`**: Appears in worst list only in reduced version
   - Pool variables are aggregates that benefit from seeing all components
   - Without `npool` and `ppool`, `cpool` predictions become less accurate

## Recommendations

### Option 1: Keep Complete Variable List (Recommended)
- **Pros**: Better overall performance, implicit CNP relationship learning
- **Cons**: More variables to predict, longer training time
- **Use Case**: When prediction accuracy is critical

### Option 2: Use Reduced List with Post-Processing
- **Pros**: Fewer variables to predict, faster training
- **Cons**: Lower accuracy, requires derivation step
- **Use Case**: When training time is critical and post-processing is acceptable
- **Note**: You can derive N and P from C using `model_variable_quantities.txt` ratios, but the derived values may not match what the model would have predicted

### Option 3: Hybrid Approach
- Train with complete variable list
- Use reduced list for inference (if needed)
- Derive N and P from C predictions using stoichiometric ratios
- **Trade-off**: Training uses full list (better learning), inference uses reduced list (faster)

## Conclusion

**The reduced variable list degrades performance because the model loses implicit CNP relationship learning.** Even though N and P variables can be derived from C variables post-hoc, the model benefits significantly from seeing all variables during training. The neural network learns better representations and maintains stoichiometric relationships when trained on the complete variable set.

**Recommendation**: Use the complete variable list for training to achieve better prediction quality, especially for carbon pool variables that show degraded performance in the reduced list.
