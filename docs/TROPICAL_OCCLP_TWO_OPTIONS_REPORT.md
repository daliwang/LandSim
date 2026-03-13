# Tropical occlp_vr Improvement: Single Global vs Two-Model Strategy

This report compares two strategies for improving occlp_vr predictions in tropical regions (where low occlp_vr contributes to high NEE bias): **Option A** uses a single global model fine-tuned on tropical data; **Option B** uses a global model for non-tropical areas and a separate tropical model for tropical areas, with lat-based switching at inference.

---

## Context

- **Problem:** In tropical regions, the CNP model underpredicts occlp_vr, which contributes to high NEE in restart simulations.
- **Phase 1:** Global natveg model trained with occlp-focused weights (`training_config_experiment_3_global_natveg_improved_occlp.json`) improves occlp_vr vs the reference run but still underpredicts at tropical sites (e.g. Amazon point lon 303.75, lat −17.43).
- **Goal:** Further improve tropical occlp_vr either by fine-tuning the global model on tropical data or by training/maintaining a dedicated tropical model and using it only in the tropics at inference.

---

## Option A: Single Global Model (Fine-Tune on Tropical)

### Description

- **Training:** Phase 1 trains a global model with occlp emphasis. Phase 2 fine-tunes that same model (e.g. via `scripts/run_finetuning.py`) using tropical data and an occlp-focused configuration (e.g. `CNP_IO_finetune_tropics.txt` pointing to Phase 1 checkpoint).
- **Inference:** One model is used for all grid cells globally.

### Pros

- **Single model:** One checkpoint and one code path; easier to maintain, deploy, and version.
- **No boundary choice:** No need to define a tropical/non-tropical latitude cutoff or blending zone; behavior is continuous across the globe.
- **Stable extratropics in practice:** With a small learning rate and limited Phase 2 epochs, fine-tuning usually improves tropics without severely degrading extratropical skill.

### Cons

- **Possible extratropical drift:** Phase 2 optimizer only sees tropical batches; some forgetting or drift in non-tropical regions is possible (monitor via validation).

### When to Prefer

- Default choice when the main goal is to fix tropical occlp_vr without adding operational complexity.
- Use when you want one globally consistent product and simpler validation and release.

---

## Option B: Two Models (Global for Non-Tropical, Tropical for Tropical)

### Description

- **Training:** Phase 1 remains the “global” model (used as-is or lightly tuned for non-tropical focus). A separate “tropical” model is trained or fine-tuned **only on tropical data** (e.g. `--tropical-only` with `train_cnp_model.py`, or fine-tune on a tropical-only dataset).
- **Inference:** For each grid cell, if `|lat| <= threshold` (e.g. 23.5° or 30°) use the tropical model; otherwise use the global model. Optionally add a small blending zone to smooth the transition.

### Pros

- **Full tropical specialization:** The tropical model can be tuned aggressively for occlp_vr (and other P pools) without any compromise for extratropical performance.
- **No impact on global product:** The global model is unchanged for non-tropical areas; no risk of Phase 2 degrading mid/high latitudes.

### Cons

- **Two checkpoints and two code paths:** More to maintain, validate, and document.
- **Boundary handling:** Requires a defined latitude threshold (and possibly blending) and branching logic in the inference pipeline; risk of visible discontinuity at the boundary.

### When to Prefer

- After Option A, if validation shows **clear degradation** in non-tropical regions (e.g. occlp_vr or NEE in mid/high latitudes).
- When you explicitly want **two products:** e.g. a global map product and a tropical-focused product for regional studies.

---

## Recommendation

1. **Start with Option A:** Fine-tune the Phase 1 global model on tropical data using `CNP_IO_finetune_tropics.txt` and the procedure in the runbook. Validate both tropical and extratropical metrics (e.g. occlp_vr at the Amazon point and at a few non-tropical sites).
2. **Adopt Option B only if needed:** If extratropical performance drops unacceptably, keep Phase 1 as the global-only model and introduce a tropical-only model, with inference logic that uses the global model for non-tropical grid cells and the tropical model for tropical grid cells.

---

## Commands: Two-model restart (Option B) — global + tropical merge

If you use a **tropical-only fine-tuned model** and want one restart that uses the **global model** outside the tropics and the **tropical model** inside the tropics:

### 1. Build the global restart (Phase 1 model)

From the **Phase 1 run** directory (e.g. `cnp_results/run_20260303_145913_natveg_improved_occlp`):

```bash
# Global inference
python ../../scripts/run_inference_all.py --model model.pth --output-dir cnp_inference_entire_dataset --derive-np-from-c

# NetCDF for restart updater (run from run dir; script expects comparison_results under current or parent dir)
python ../../scripts/ai_predictions_to_netcdf.py

# Full restart from global model (no tropical filter)
python ../../scripts/ai_predictions_to_restart.py \
  --ai-predictions comparison_results/ai_predictions_for_plotting.nc \
  --restart-file /path/to/original_restart.nc \
  --output updated_restart_global.nc \
  --variable-list ../../CNP_IO_updated9_dev_dw.txt
```

### 2. Build tropical predictions NetCDF (fine-tuned model)

From the **fine-tune run** directory (e.g. `cnp_results/finetune_YYYYMMDD_HHMMSS`):

```bash
# Inference with tropical model (on full dataset so we have predictions for every gridcell)
python ../../scripts/run_inference_all.py --model model.pth --output-dir cnp_inference_entire_dataset --derive-np-from-c

python ../../scripts/ai_predictions_to_netcdf.py
```

### 3. Replace only tropical region in the global restart

From the **same fine-tune run** directory, use the **global** restart as input and the **tropical** predictions NetCDF; restrict updates to the tropical band with `--tropical-lat-range`:

```bash
python ../../scripts/ai_predictions_to_restart.py \
  --ai-predictions comparison_results/ai_predictions_for_plotting.nc \
  --restart-file /path/to/updated_restart_global.nc \
  --output updated_restart_global_with_tropical.nc \
  --variable-list ../../CNP_IO_updated9_dev_dw.txt \
  --tropical-lat-range -30,30
```

Result: `updated_restart_global_with_tropical.nc` has global-model values everywhere except gridcells with latitude in [−30, 30], which are filled from the tropical model.

---

## References

- Phase 1 config: `config/training_config_experiment_3_global_natveg_improved_occlp.json`
- Phase 2 / fine-tune config: `config/training_config_experiment_3_global_natveg_occlp_phase2_tropics.json` (for standalone Phase 2 training); fine-tune uses `CNP_IO_finetune_tropics.txt` and Phase 1 checkpoint.
- Runbook: `docs/CNP_pipeline_runbook.md` (fine-tune and global inference sections).
- Phase 1 run (example): `cnp_results/run_20260303_145913_natveg_improved_occlp`
- Reference run (pre–occlp focus): `cnp_results/run_20260228_214757_natveg_improved`
