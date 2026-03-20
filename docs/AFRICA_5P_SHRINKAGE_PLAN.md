## Africa 5P bias issue and shrinkage plan

### Diagnosis

- **Problem**: In central Africa, all 5P predictions (`labilep_vr`, `occlp_vr`, `solutionp_vr`, `secondp_vr`, `primp_vr`) are systematically too high in both the global and Phase2 tropical models.
- **Magnitude**:
  - `occlp_vr` ground truth max at key Africa sites is ~150, but predictions are ~400.
  - `solutionp_vr` ground truth max is ~0.00125, but predictions after bias correction can be ~0.03 (10–30× larger).
- **Fine-tuning result**: Africa-only 5P fine-tuning of Phase2 reduces loss but does **not** fix the magnitude issue at the Africa validation site (27.5E, 2.36N); predictions remain orders of magnitude above GT.
- **Bias/scale limitation**: Region-wise linear corrections (a * pred + b) cannot:
  - Pull down extreme Africa values into the GT range, **and**
  - Preserve Phase2’s good behavior in Amazon and other regions.

### Constraints

- Preserve **Amazon** 5P performance (already excellent after bias/scale).
- Do **not** change non-P variables.
- Keep existing Phase2 / Phase3 restart pipeline as much as possible.

### Proposed solution: Africa-only 5P shrinkage layer

1. **Keep existing pipeline as-is outside Africa**:
   - Global natveg_improved model → Phase1 restart.
   - Phase2 P-focused model → full-grid/tropical predictions.
   - Amazon + Africa bias/scale correction step for 5P (current `apply_5p_bias_scale_correction.py`).
   - `ai_predictions_to_restart.py` to write the final restart.

2. **Add an Africa-only shrinkage correction for 5P variables**:
   - Restrict to the **Africa region box**: lat ∈ [-15, 15], lon ∈ [0, 30].
   - For each 5P variable, use Africa GT and Phase2 predictions to compute a **shrinkage factor per layer**:
     - For cells with positive GT and prediction, compute `ratio = gt / pred` in the Africa box.
     - For each layer, take a robust statistic such as `median(ratio)` over Africa.
     - Clamp the factor for that layer to a safe range, e.g. `factor ∈ [0.0, 1.0]` with lower bound > 0.
   - Apply `corrected = pred * factor` **only in Africa**, leaving Amazon and other regions untouched.
   - Optionally clip corrected Africa values per-layer to an upper bound such as `p99(GT_Africa_layer)` to avoid any remaining outliers.

3. **Implement as a separate script** `scripts/apply_5p_africa_shrinkage.py`:
   - Inputs:
     - `--run-dir`: a `cnp_results` run that already has
       `cnp_inference_entire_dataset/cnp_predictions/soil_2d_ground_truth` and
       `soil_2d_predictions_5P_bias_corrected_phase2/`.
   - For each 5P variable:
     - Read `ground_truth_Y_{var}.csv` and `predictions_Y_{var}_bias_corrected.csv`.
     - Filter rows to the Africa box.
     - Compute per-layer shrinkage factor as described above.
     - Apply `pred * factor` for rows in Africa region, leave others unchanged.
     - Write corrected CSVs to a new subdir, e.g. `soil_2d_predictions_5P_bias_corrected_africa_shrinkage/`.
   - Save the shrinkage factors and basic diagnostics (per-layer factors, min/max GT and pred in Africa) to a JSON file under `run_dir/analysis/`.

4. **Use shrinkage output in restart creation**:
   - Modify the Phase3 workflow so that:
     1. Run `apply_5p_bias_scale_correction.py` (produces `soil_2d_predictions_5P_bias_corrected_phase2`).
     2. Run `apply_5p_africa_shrinkage.py` to produce Africa-shrunk 5P CSVs.
     3. Run `ai_predictions_to_netcdf.py` **pointing at the Africa-shrunk subdir** instead of the original bias-corrected subdir.
     4. Run `ai_predictions_to_restart.py` as before with `--tropical-lat-range=-30,30`.

5. **Validation plan**:
   - For selected Africa sites (including the problematic site near 28E, 0N and others), compare vertical profiles and MAE/RMSE for the 5P variables across:
     - Phase2 raw,
     - Phase2 + bias/scale,
     - Phase2 + bias/scale + Africa shrinkage.
   - Confirm:
     - Africa 5P magnitudes are closer to GT (within a factor ~2 instead of 10–100×).
     - Amazon plots are unchanged relative to the current Phase3 restart.

