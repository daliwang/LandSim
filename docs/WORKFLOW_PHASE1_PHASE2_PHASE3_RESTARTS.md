### 5. Site‑based 5P restart comparison (Amazon / Africa or any site)

Once `phase1_global`, `phase2_tropical`, and `phase3_tworegions` restarts are created,
you can **automatically generate single‑point restarts and 5P comparison plots** for
any longitude/latitude using:

- `scripts/generate_site_5p_restart_comparison.py`
- Documented in `docs/SITE_5P_RESTART_COMPARISON.md`

Example (Africa site at lon=28, lat=0):

```bash
python scripts/generate_site_5p_restart_comparison.py \
  --lon 28.0 \
  --lat 0.0 \
  --site-name africa_28_0
```

Example (Amazon validation site):

```bash
python scripts/generate_site_5p_restart_comparison.py \
  --lon 303.75 \
  --lat -17.434553 \
  --site-name amazon_303_17S
```

The script will:

- Extract single‑point restarts from the three phase runs at the requested site.
- Load Phase1 **ground‑truth 5P vertical profiles** at that site.
- Produce line plots comparing GT vs `phase1_global`, `phase2_tropical`,
  and `phase3_tworegions` for the 5 P variables.

## Workflow: phase1_global, phase2_tropical, phase3_tworegions restarts

This document summarizes how to create three restart files:

- **phase1_global**: global baseline restart (copy of the natveg_improved / Phase1 restart).
- **phase2_tropical**: tropical-only training with the **updated** `config/training_config_phase2_tropical_soilp_only.json` (emphasis on the five soil P variables). Produces a global restart with **raw** Phase2 5P in the tropical band \([-30°, 30°]\); extratropics unchanged from Phase1. The Phase2 run is the basis for Phase3 (e.g. `run_*_phase2_tropical_soilp_amazon_africa`).
- **phase3_tworegions**: **two-region (Amazon + Africa) bias correction and merge** of the 5 P variables. Apply bias/scale correction for Amazon and Africa separately, merge with `scripts/merge_5p_bias_corrected_amazon_africa.py`, then build a global restart from the Phase1 base with bias-corrected 5P in the tropics; extratropics unchanged from Phase1.

It is based on:

- `docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md`
- `docs/INSTRUCTIONS_TRENDY_1_AI_RESTART_CREATION.md`
- [REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md](./REPORT_PHASE3_5P_BIAS_CORRECTION_REVIEW.md) — bias/scale methodology, full-region objectives for `solutionp_vr` / `occlp_vr`, and improvement directions.
- [PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md](./PHASE3_TWOREGIONS_VS_GOOD_RUN_DIFFERENCE.md) — split Amazon/Africa correction + merge vs older single-config Phase3.
- Quantitative **Amazon / Africa 5P pred vs GT**: `scripts/compare_5p_gt_two_regions_inference.py` (`pred-eval`).

The recommended way to run all three phases is to use
`scripts/run_phase_restarts.sh`, described at the end of this document.

---

### 0. Common setup and model training for new users

For a **new user**, the full TRENDY‑1 AI restart workflow should start by
training **two CNP models**:

- A **global natveg_improved‑like model** (used for `phase1_global` and as the
  base restart).
- A **tropical Phase2 P‑focused model** (used for `phase2_tropical` and
  `phase3_tworegions`).

These two trainings are done with `train_cnp_model.py`, and then the rest of
the restart creation steps follow the procedure in
`docs/INSTRUCTIONS_TRENDY_1_AI_RESTART_CREATION.md` and in this document.

Assumptions below use the existing runs; if you train your own, replace them
with your paths:

- **Natveg run**: `cnp_results/run_20260228_214757_natveg_improved`
- **Base restart**:
  - `cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc`
- **Phase2 run**: tropical-only run with the **updated** Phase2 config (emphasis on 5 P variables), e.g. `cnp_results/run_20260313_224805_phase2_tropical_soilp_amazon_africa`. This is the basis run for Phase3.
- **Variable list file**: `CNP_IO_updated9_dev_dw.txt` in repo root.

If you are a new user, train your own models as in §0.1 and §0.2. Phase2 must use **tropical-only** training with `config/training_config_phase2_tropical_soilp_only.json` (the updated config with more emphasis on the five P variables).

#### 0.1 Train a natveg_improved‑like global model (for phase1_global)

From the repo root:

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

python train_cnp_model.py \
  --config config/training_config_experiment_3_global_natveg_improved.json \
  --run-dir cnp_results/run_YYYYMMDD_HHMMSS_natveg_improved_custom \
  --variable-list CNP_IO_updated9_dev_dw.txt
```

After training:

1. Follow `docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md` (or your
   existing ELM workflow) to produce a **global restart NetCDF** from this run
   (full‑domain inference → NetCDF → `ai_predictions_to_restart.py` as needed).
2. Use the resulting restart as your new **base restart** and set:

   ```bash
   export NATVEG_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_natveg_improved_custom"
   export BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_...your_file.nc"
   ```

#### 0.2 Train the Phase2 tropical model (updated config, emphasis on 5 P variables)

Phase2 uses **tropical-only** training with the **updated**
`config/training_config_phase2_tropical_soilp_only.json`, which emphasizes the
five soil P variables (labilep_vr, occlp_vr, solutionp_vr, secondp_vr, primp_vr).
The resulting run (e.g. `run_*_phase2_tropical_soilp_amazon_africa`) is the basis
for Phase3 two-region bias correction and merge.

**Option A – use the automation script:**

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

export CONFIG_TROPICAL=config/training_config_phase2_tropical_soilp_only.json
bash scripts/run_phase2_tropical.sh
```

The script writes to `run_*_phase2_tropical`. To match the basis run naming
(`run_*_phase2_tropical_soilp_amazon_africa`), use Option B.

**Option B – train manually with suffix `phase2_tropical_soilp_amazon_africa`:**

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

python train_cnp_model.py \
  --training-config-json config/training_config_phase2_tropical_soilp_only.json \
  --output-dir cnp_results \
  --output-dir-suffix phase2_tropical_soilp_amazon_africa \
  --variable-list CNP_IO_updated9_dev_dw.txt
```

Then set:

```bash
export PHASE2_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_phase2_tropical_soilp_amazon_africa"
```

The remaining inference, bias/scale, and restart steps are the same; only
`NATVEG_RUN_DIR`, `BASE_RESTART`, and `PHASE2_RUN_DIR` change to your run paths.

---

Create three run directories (example naming) and set run-specific variables.
**Existing users** can set `NATVEG_RUN_DIR`, `BASE_RESTART`, and `PHASE2_RUN_DIR` to
the example paths above (or to their own run paths) and skip training.

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

# For existing users: point to your natveg and Phase2 runs and base restart
# export NATVEG_RUN_DIR=cnp_results/run_20260228_214757_natveg_improved
# export BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc"
# export PHASE2_RUN_DIR=cnp_results/run_20260313_224805_phase2_tropical_soilp_amazon_africa   # tropical-only, updated Phase2 config (5 P emphasis)

TS=$(date +%Y%m%d_%H%M%S)

RUN1_DIR="cnp_results/run_${TS}_phase1_global"
RUN2_DIR="cnp_results/run_${TS}_phase2_tropical"
RUN3_DIR="cnp_results/run_${TS}_phase3_tworegions"

mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$RUN3_DIR"
```

---

### 1. phase1_global: global baseline restart (natveg_improved copy)

**Goal:** create a clearly labeled baseline restart identical to the natveg/Phase1
global restart. Use `BASE_RESTART` (from Phase1 or `NATVEG_RUN_DIR`).

```bash
cp "$BASE_RESTART" "$RUN1_DIR/updated_restart_phase1_global_natveg_improved.nc"
```

Optional README:

```bash
echo "phase1_global: copy of natveg_improved restart on $(date)" \
  > "$RUN1_DIR/README_phase1_global.txt"
```

---

### 2. phase2_tropical: tropical-only run (updated config) → restart with raw Phase2 5P

**Goal:** Use the **Phase2 run** trained with the updated
`config/training_config_phase2_tropical_soilp_only.json` (tropical-only, emphasis
on the five P variables). Overwrite 5P in the tropical band \([-30°, 30°]\) with
**raw Phase2 predictions**; extratropics stay as in the Phase1 base.

Set `PHASE2_RUN_DIR` to your Phase2 run (e.g.
`cnp_results/run_20260313_224805_phase2_tropical_soilp_amazon_africa`). The
commands below use `$PHASE2_RUN_DIR`.

#### 2.1 Tropical-only Phase2 inference (if not already done)

Use the checkpoint in your Phase2 run (`cnp_model.pt` or `cnp_predictions/model.pth`):

```bash
cd "$PHASE2_RUN_DIR"

python ../../scripts/run_inference_all.py \
  --model cnp_model.pt \
  --output-dir cnp_inference_tropical_only

cd -  # back to repo root
```

#### 2.2 Build NetCDF from tropical predictions

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$PHASE2_RUN_DIR/cnp_inference_tropical_only/cnp_predictions" \
  --output "$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt
```

#### 2.3 Create phase2_tropical restart (5P in tropics, raw Phase2)

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc" \
  --restart-file "$BASE_RESTART" \
  --output "$RUN2_DIR/updated_restart_phase2_tropical_5P_raw.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

Optional README:

```bash
echo "phase2_tropical: natveg base + raw Phase2 5P in tropics on $(date)" \
  > "$RUN2_DIR/README_phase2_tropical.txt"
```

---

### 3. phase3_tworegions: two-region bias correction and merge of 5 P variables

**Goal:** Apply **Amazon and Africa 5P bias/scale correction** separately, **merge**
the corrected 5 P variables with `scripts/merge_5p_bias_corrected_amazon_africa.py`,
then build a **global** restart from the Phase1 base with bias-corrected 5P in the
tropics. This is the recommended flow (Path 3B, using your Phase2 run).

**Path 3B** (recommended): Use the Phase2 run that already has (or will have)
full-grid inference; apply Amazon-only and Africa-only bias correction, merge,
then NetCDF and restart. Basis run: same as Phase2 (e.g.
`run_20260313_224805_phase2_tropical_soilp_amazon_africa`).

**Path 3A** (alternative): Single two-region correction with inference in a
dedicated RUN3_DIR; use `scripts/run_phase3_tworegions.sh` or run steps manually.

---

#### Path 3B (recommended): Amazon and Africa corrections separately, then merge

Use your **Phase2 run** (same as in §2, e.g.
`run_20260313_224805_phase2_tropical_soilp_amazon_africa`). It must contain
full-grid inference (`cnp_inference_entire_dataset`). Apply bias/scale for
Amazon only and Africa only, merge with
`scripts/merge_5p_bias_corrected_amazon_africa.py`, then build the NetCDF and
global restart. Region configs: `config/training_config_amazon_5p_box.json` and
`config/training_config_africa_5p_box.json`.

**3B.1 Ensure full-grid inference exists under Phase2 run**

If not already done, run full-grid inference into your Phase2 run:

```bash
python scripts/run_inference_all.py \
  --model "$PHASE2_RUN_DIR/cnp_model.pt" \
  --output-dir "$PHASE2_RUN_DIR/cnp_inference_entire_dataset" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --inference-full-grid
```

**3B.2 Apply 5P bias/scale for Amazon only**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$PHASE2_RUN_DIR" \
  --region-config-json config/training_config_amazon_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_amazon
```

**3B.3 Apply 5P bias/scale for Africa only**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$PHASE2_RUN_DIR" \
  --region-config-json config/training_config_africa_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_africa
```

**3B.4 Merge Amazon and Africa corrected 5P**

```bash
python scripts/merge_5p_bias_corrected_amazon_africa.py \
  --run-dir "$PHASE2_RUN_DIR"
```

This creates  
`$PHASE2_RUN_DIR/cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_5P_bias_corrected_amazon_africa/`.

**3B.5 Build global NetCDF and create global restart (Phase1 base + 5P in tropics)**

```bash
mkdir -p "$PHASE2_RUN_DIR/comparison_results"

python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$PHASE2_RUN_DIR/cnp_inference_entire_dataset/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_amazon_africa \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --output "$PHASE2_RUN_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa.nc"

python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$PHASE2_RUN_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa.nc" \
  --restart-file "$BASE_RESTART" \
  --output "$RUN3_DIR/updated_restart_global_5P_bias_corrected_amazon_africa.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

For Path 3B you can write the restart into `RUN3_DIR` (as above) or into
`$PHASE2_RUN_DIR`; adjust `--output` accordingly. The **base restart must be
the Phase1 global restart** (`BASE_RESTART`) so the result is a full global
restart with 5P updated only in the tropics (Amazon and Africa use
bias-corrected values; other tropics use raw Phase2 from the merged NetCDF).

---

#### Path 3A (alternative): Single two-region correction (inference in RUN3_DIR)

Use when you want a dedicated Phase3 run directory or when using
`scripts/run_phase3_tworegions.sh`. Full-grid inference is run into `RUN3_DIR`
and one bias/scale correction is applied for both regions.

**3A.1 Full-grid Phase2 inference into RUN3_DIR**

```bash
python scripts/run_inference_all.py \
  --model "$PHASE2_RUN_DIR/cnp_model.pt" \
  --output-dir "$RUN3_DIR/cnp_inference_entire_dataset" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --inference-full-grid
```

**3A.2 Apply 5P bias/scale correction (Amazon + Africa in one call)**

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_two_region_five_p.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_phase2
```

Outputs: corrected 5P CSVs under  
`$RUN3_DIR/cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_5P_bias_corrected_phase2/`  
and `$RUN3_DIR/analysis/bias_scale_params_5P_two_regions.json`.

**3A.3 Build global NetCDF and create phase3 restart**

```bash
mkdir -p "$RUN3_DIR/comparison_results"

python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$RUN3_DIR/cnp_inference_entire_dataset/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_phase2 \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --output "$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc"

python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc" \
  --restart-file "$BASE_RESTART" \
  --output "$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_tropical.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

---

Optional README for Phase3:

```bash
echo "phase3_tworegions: natveg base + Phase2 5P with Amazon+Africa bias/scale in tropics on $(date)" \
  > "$RUN3_DIR/README_phase3_tworegions.txt"
```

---

### 4. Automation scripts (run 1 → inspect → 2 → inspect → 3)

## make sure to export RESTART_TEMPLATE

# export RESTART_TEMPLATE=/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc


For new users, the recommended workflow is to run three scripts **sequentially**,
inspecting results between phases:

1. `scripts/run_phase1_global.sh`  
   - Trains the global natveg_improved‑like model (or reuses an existing one).  
   - Runs full‑grid inference and creates a **base restart** from an ELM
     restart template.  
   - Exports `NATVEG_RUN_DIR` and `BASE_RESTART` in its log output.

2. `scripts/run_phase2_tropical.sh`  
   - Trains the **tropical-only** Phase2 model with the updated
     `config/training_config_phase2_tropical_soilp_only.json` (emphasis on 5 P
     variables), or reuses an existing run (e.g.
     `run_*_phase2_tropical_soilp_amazon_africa`).  
   - Runs tropical‑only inference and creates the **phase2_tropical** restart
     with raw Phase2 5P in the tropics, using `BASE_RESTART` from phase 1.  
   - Exports `PHASE2_RUN_DIR` and prints the path to the tropical restart.

3. **Phase3 – two-region bias correction and merge:**  
   - **Recommended (Path 3B):** Run the Phase3 steps in §3 manually: full-grid
     inference in `PHASE2_RUN_DIR` (if needed), apply Amazon-only and
     Africa-only bias correction, run `scripts/merge_5p_bias_corrected_amazon_africa.py`,
     then build NetCDF and global restart from `BASE_RESTART`.  
   - **Alternative (Path 3A):** `scripts/run_phase3_tworegions.sh` runs
     full‑grid inference into a new RUN3_DIR, applies a single two‑region
     correction, and creates the phase3_tworegions restart.

Between scripts you can:

- Inspect logs, validation plots, and diagnostics for each model/run.
- Adjust configs or weights and re‑run the specific phase if needed.

Each script is self‑contained and safe to re‑run; they will reuse existing runs
when possible instead of retraining from scratch.

