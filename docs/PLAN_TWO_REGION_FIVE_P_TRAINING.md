# Plan: Two-Region, Five P–Variable–Only Model

This document outlines how to train a model focused on **two geographic regions** and **five soil P variables** only, and how to use it for prediction in those regions.

---

## 1. Target regions (lat/lon)

- **Region 1 – Amazon:**  
  Latitude **10°N to 30°S** (−30 to 10), Longitude **90°W to 30°W** (270° to 330° in 0–360°).  
- **Region 2 – Central Africa:**  
  Latitude **15°N to 15°S** (−15 to 15), Longitude **0° to 30°E** (0° to 30° in 0–360°).

So in (lat_min, lat_max, lon_min, lon_max) with lon in **0–360**:

- Amazon: `(-30, 10, 270, 330)`
- Africa:  `(-15, 15, 0, 30)`

---

## 2. Five P variables

- `labilep_vr`
- `occlp_vr`
- `solutionp_vr`
- `secondp_vr`
- `primp_vr`

---

## 3. Dataloader: restrict to these two regions

**Implemented:** The dataloader now supports **region boxes**.

- **Config:** `DataConfig.region_boxes`  
  - Type: `Optional[List[Tuple[float, float, float, float]]]`  
  - Each tuple: `(lat_min, lat_max, lon_min, lon_max)` with **longitude in 0–360°**.
- **Behavior:** In `data_loader_individual.preprocess_data()`, only rows whose (lat, lon) fall **inside at least one** box are kept. Lat/lon columns are resolved automatically (e.g. `Latitude`/`Longitude`).
- **Where to set:** Via training config JSON under `data_filtering_config.region_boxes`.

**Example for the two regions:**

```json
"data_filtering_config": {
  "region_boxes": [
    [-30, 10, 270, 330],
    [-15, 15, 0, 30]
  ],
  "natveg_only": true,
  "natveg_filter_before_split": false
}
```

- **Training (main script):** `train_cnp_model.py` applies `data_filtering_config` from the JSON (including `region_boxes`) when you pass `--training-config-json`.
- **Finetuning:** `run_finetuning_json.py` also applies `data_filtering_config.region_boxes` from the unified config.

So: **use the existing dataloader with a training config that sets `region_boxes`**; no extra code path is required.

---

## 4. “Five P only” training: two practical options

The current model has one scalar head, one PFT-1D head, and one soil2D (matrix) head. The matrix head size is `n_2d_vars * rows * cols`. So you can either keep the full variable set and focus loss on the 5 P vars, or reduce the variable list to 5 P only (smaller matrix head).

### Option A (recommended): Full variable list, 5 P–focused weights

- **Variable list:** Keep using the full CNP list (e.g. `CNP_IO_updated9_dev_dw.txt`) so the **model architecture is unchanged** (same heads, same number of 2D variables).
- **Data:** Use **only** the two regions above via `region_boxes`.
- **Loss focus:** In the training config JSON, set **`variable_weights.soil2d_weights`** so that the five P variables have **high weight** (e.g. 20–50) and all other soil2d variables **low or zero** (e.g. 0.1 or 0). The model will then prioritize fitting the 5 P variables in these two regions.
- **Pros:** No change to model construction or inference pipeline; reuse existing inference/restart scripts.  
- **Cons:** Model still has parameters for non-P outputs; training is a bit less “focused” than a true 5-P-only head.

**Example (excerpt) in training config JSON:**

```json
"variable_weights": {
  "soil2d_weights": {
    "labilep_vr": 40,
    "occlp_vr": 50,
    "solutionp_vr": 40,
    "secondp_vr": 40,
    "primp_vr": 30,
    "cwdc_vr": 0.1,
    "cwdn_vr": 0.1,
    "cwdp_vr": 0.1,
    "litr2c_vr": 0.1,
    "litr2n_vr": 0.1,
    "litr2p_vr": 0.1,
    "litr3c_vr": 0.1,
    "litr3n_vr": 0.1,
    "litr3p_vr": 0.1,
    "soil1c_vr": 0.1,
    "soil1n_vr": 0.1,
    "soil1p_vr": 0.1,
    "soil2c_vr": 0.1,
    "soil2n_vr": 0.1,
    "soil2p_vr": 0.1,
    "soil3c_vr": 0.1,
    "soil3n_vr": 0.1,
    "soil3p_vr": 0.1,
    "soil4c_vr": 0.1,
    "soil4n_vr": 0.1,
    "soil4p_vr": 0.1
  }
}
```

(Adjust names to match exactly your CNP_IO 2D list.)

### Option B: Five P variables only (smaller matrix head)

- **Variable list:** Create a **minimal** CNP_IO (or config) where the **only** 2D soil **outputs** are the five P variables. Inputs can remain the same (or be pruned to a minimal set); scalar/PFT-1D can be minimal (e.g. one variable each) if the code allows.
- **Model:** The matrix head becomes `5 * matrix_rows * matrix_cols`. This requires that the rest of the pipeline (data_info, trainer, inference) is built from this reduced list so that only 5 P columns are expected and written.
- **Pros:** Smaller head; training is only for the 5 P variables.  
- **Cons:** Requires a dedicated variable list and possibly small code paths to support “soil2d-only” or minimal scalar/PFT-1D; inference and restart scripts must use the same reduced list.

**Summary:** For “reasonably just predictions for these 5 P variables in these two regions,” **Option A** is the least intrusive and is supported with the new `region_boxes` and existing weight config.

---

## 5. Suggested workflow (Option A)

1. **Create a training config JSON** (e.g. `config/training_config_two_region_five_p.json`) that:
   - Sets `data_filtering_config.region_boxes` to `[[-30, 10, 270, 330], [-15, 15, 0, 30]]`.
   - Sets `data_filtering_config.natveg_only` (and optionally `natveg_filter_before_split`) as desired.
   - Sets `variable_weights.soil2d_weights` so that the five P variables have high weight and all other soil2d variables low/zero (as in the example above).
   - Keeps other sections (training_hyperparameters, reproducibility_config, etc.) aligned with your current phase2/tropical setup.

2. **Train:**
   ```bash
   python train_cnp_model.py \
     --training-config-json config/training_config_two_region_five_p.json \
     --variable-list CNP_IO_updated9_dev_dw.txt \
     --use-tva4km \
     --output-dir-suffix two_region_five_p
   ```
   The dataloader will load only grid cells in the two regions (and apply natveg if enabled); the loss will be dominated by the 5 P variables.

3. **Inference:**  
   Use the same variable list and model as today:
   - **Full-grid inference:** Run inference as usual; the model will output all variables (including the 5 P). For “focus” you can either use the outputs everywhere or only trust/use the 5 P predictions inside the two regions.
   - **Restart update:** Use `ai_predictions_to_restart.py` with `--variables-to-update` set to the five P names and, if available, a **spatial mask** so that only cells inside the two boxes are updated (see below).

4. **Optional – inference only in the two regions:**  
   If you want predictions **only** for the two regions (e.g. to avoid applying the regional model outside those areas):
   - Either run inference on the full grid and then **discard or mask** predictions outside the two boxes, or
   - Add an option to the inference script to **subset the dataloader** (e.g. by passing the same `region_boxes` or a list of (lat, lon) cells) so that only those grid cells are loaded and predicted. That would be a small extension of the current inference pipeline.

5. **Restart / CSV output:**  
   For writing restarts or CSVs “only in the two regions,” you need a spatial mask in the script that writes restarts/CSVs (e.g. `ai_predictions_to_restart.py`). Options:
   - Add a `--region-boxes` (or similar) argument that takes the same two boxes and only updates grid cells whose (lat, lon) fall in one of the boxes; or
   - Precompute a list of gridcell indices for the two regions and pass that as a mask.  
   This keeps the “five P only” and “two regions only” semantics consistent from training to application.

---

## 6. Checklist (last task to finish)

- [x] **Dataloader:** Add `region_boxes` to `DataConfig` and filter in `preprocess_data()` so only cells in the two regions are used (done).
- [x] **Config wiring:** Support `region_boxes` in `data_filtering_config` in `train_cnp_model.py` and `run_finetuning_json.py` (done).
- [x] **Training config:** Add `config/training_config_two_region_five_p.json` with the two boxes and 5 P–focused soil2d weights (done; see also `docs/IMPROVE_5P_PREDICTIONS_SUGGESTIONS.md`).
- [ ] **Train** a run with that config and your chosen variable list.
- [ ] **Validate** on the two regions (e.g. Amazon site + a few Africa cells) via existing validation/plot scripts.
- [ ] **Inference:** Decide whether to run full-grid and mask, or add a region-only inference option; implement restart/CSV masking for the two regions if you want updates only there.

---

## 7. Longitude convention

- **0–360°:** 90°W = 270°, 30°W = 330°, 0°E = 0°, 30°E = 30°.
- The dataloader normalizes longitude to 0–360 when applying `region_boxes`, so if your data use −180..180, they are converted before comparison.

---

## 8. Summary

- **Regions:** Amazon (lat −30–10, lon 270–330) and Central Africa (lat −15–15, lon 0–30); lon in 0–360.
- **Variables:** Train with strong focus on the five P variables (Option A: full list + weights; Option B: minimal list and smaller matrix head).
- **Dataloader:** Use `data_filtering_config.region_boxes` in your training config; no dataloader code change needed beyond what’s already implemented.
- **Next steps:** Add the JSON config, run training, then add optional inference/restart masking so that “five P only” and “two regions only” are applied end-to-end where you need them.
