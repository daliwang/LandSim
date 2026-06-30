# E3SMv3_h0: model capacity vs grid size — do we need a larger transformer?

**Date:** 2026-06-30  
**Context:** E3SM training uses ~396k gridcells vs ~21k in Trendy. Question: should embedding size and transformer depth increase?  
**Related:** [REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md](REPORT_E3SM_SENSITIVE_P_ELMS_SCALE_R2_PHASE3.md) — the dominant E3SM issue is scale/R² mismatch, not capacity.

---

## 1. Short answer

**No — not because E3SM has more gridcells.** Gridcell count increases the **number of training samples**, not the **size of each forward pass**. More data with the same architecture usually improves generalization; it does not by itself require a wider transformer.

Current E3SM Phase 2 already achieves **R² ~0.98–0.99** on most tropical 5P variables. The unresolved problem is **`solutionp_vr` magnitude at sensitive sites**, not global underfitting. Scaling the model is unlikely to fix that without changing the **loss and evaluation metrics**.

---

## 2. Gridcells vs per-sample complexity

| | Trendy | E3SMv3_h0 |
|--|--------|-----------|
| Global gridcells (inference) | ~21k | ~396k |
| Tropical train samples | smaller | ~59k train + ~16k val |
| Per-sample I/O layout | CNP_IO (with scalars) | CNP_IO_e3smv3_h0 (no GPP/NPP/AR/HR) |
| Architecture (Phase 1/2 runs) | `token_dim=128`, 4 transformer layers, `lstm_hidden=64` | **Same** |

Each training step processes **one gridcell snapshot** with fixed input/output geometry. The transformer does not ingest the full grid at once.

---

## 3. Evidence that capacity is sufficient

From E3SM Phase 2 baseline (`run_20260624_092639`):

- Tropical holdout mean R²: `solutionp_vr` ~0.988, `occlp_vr` ~0.978, `labilep_vr` ~0.978, `secondp_vr` ~0.979  
- Amazon/Africa pooled R² on full tropical inference: ~0.996–0.997 on headline 3P  

If the model were **capacity-limited**, we would expect **high train and val loss**, low R² everywhere, and clear improvement when widening the net. That is not observed.

Remaining errors are **systematic scale bias** on `solutionp_vr` at sites (see sensitive-P report), not inability to fit patterns.

---

## 4. When model scaling *would* make sense

Consider larger encoders / transformer (`docs/model_scaling_guide.md`, e.g. `CNP_model_config_v01_with_large_embedding.txt`) only if:

1. **Input/output geometry grows** — e.g. full 18 soil columns × 10 layers instead of 1×10, or many new variables.  
2. **Underfitting** — train and val loss both stay high; R² plateaus low after long training.  
3. **New input modalities** — restored scalars, extra forcing, multi-mode heads needing more fusion capacity.

Scale **because of measured underfitting or I/O expansion**, not gridcell count.

---

## 5. Risks of scaling up now

- Longer training and inference  
- Higher risk of **overfitting high-weight variables** in normalized space  
- No guarantee of fixing **order-of-magnitude `solutionp` error** (a loss/metric problem)  
- Phase 2 fine-tuning and Phase 3 calibration become harder to compare across runs  

---

## 6. Recommendation

**Keep the current CNP architecture** for E3SM Phase 2 / Phase 3 until log-relative loss or multiplicative calibration is tested.

Priority order:

1. Baseline Phase 2 + **scale-first calibration / loss** for `solutionp_vr`  
2. E3SM-specific normalization review (quantile scalers vs min/max on tiny ranges)  
3. Model-scale experiment **only** if a log-relative retrain still shows clear underfitting  

---

## 7. Reference config locations

- Default model sizes: `config/training_config.py` → `get_cnp_combined_config()`  
- Saved in runs: `cnp_config.json` → `embed_dim`, `token_dim`, `transformer_layers`  
- Large-model template: `CNP_model_config_v01_with_large_embedding.txt`  
