## Workflow: phase1_global, phase2_tropical, phase3_tworegions restarts

This document summarizes how to create three restart files from existing
`natveg_improved` and `phase2_pvariable_focus` runs:

- **phase1_global**: copy of the `natveg_improved` restart (baseline).
- **phase2_tropical**: tropical restart with raw Phase2 5P in \([-30°, 30°]\).
- **phase3_tworegions**: tropical restart with Phase2 5P plus Amazon + Africa bias/scale correction.

It is based on:

- `docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md`
- `docs/INSTRUCTIONS_TRENDY_1_AI_RESTART_CREATION.md`

The recommended way to run all three phases is to use
`scripts/run_phase_restarts.sh`, described at the end of this document.

---

### 0. Common setup and how to train your own models

Assumptions (adjust paths if your run IDs differ):

- **Natveg run**: `cnp_results/run_20260228_214757_natveg_improved`
- **Base restart**:
  - `cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc`
- **Phase2 run**: `cnp_results/run_20260305_153217_phase2_pvariable_focus`
- **Variable list file**: `CNP_IO_updated9_dev_dw.txt` in repo root.

If you are a new user and want to **train your own models** instead of reusing
the existing runs, follow these high-level steps:

#### 0.1 Train your own natveg_improved-like global model (for phase1_global)

1. Choose or copy a global training config, e.g.:
   - `config/training_config_experiment_3_global_natveg_improved.json`
2. From the repo root, run a standard training script (adjust config path and
   output directory name as needed):

   ```bash
   python train_cnp_repeat.py \
     --config config/training_config_experiment_3_global_natveg_improved_occlp.json \
     --run-dir cnp_results/run_YYYYMMDD_HHMMSS_natveg_improved_custom
   ```

3. After training, produce a global restart file for your new run using the
   same pipeline that was used for `natveg_improved` (see
   `docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md` for details). In
   many cases this means:
   - Running full-domain inference for the model.
   - Converting predictions to NetCDF.
   - Using `ai_predictions_to_restart.py` (or your existing ELM workflow) to
     generate an updated restart NetCDF.
4. Use the resulting restart as your new **base restart** and update the
   variables in this document accordingly, for example:

   ```bash
   export NATVEG_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_natveg_improved_custom"
   export BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_...your_file.nc"
   ```

#### 0.2 Train your own phase2_tropical-like P-focused tropical model

1. Use the Phase2 two-region/P-focused configs as templates, for example:
   - `config/training_config_phase2_tropical_soilp_only.json`  
     (tropical-only, natveg-only, high weights on 5P).
2. Train your Phase2-style model:

   ```bash
   python train_cnp_repeat.py \
     --config config/training_config_phase2_tropical_soilp_only.json \
     --run-dir cnp_results/run_YYYYMMDD_HHMMSS_phase2_pvariable_focus_custom
   ```

3. This creates a run directory similar to
   `cnp_results/run_20260305_153217_phase2_pvariable_focus` but with your own
   timestamp and name. Use that directory as `PHASE2_RUN_DIR` in the commands
   below, for example:

   ```bash
   export PHASE2_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_phase2_pvariable_focus_custom"
   ```

4. The remaining inference, bias/scale, and restart steps in this document
   work the same way; only the run directory names change.

---

Create three run directories (example naming):

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

TS=$(date +%Y%m%d_%H%M%S)

RUN1_DIR="cnp_results/run_${TS}_phase1_global"
RUN2_DIR="cnp_results/run_${TS}_phase2_tropical"
RUN3_DIR="cnp_results/run_${TS}_phase3_tworegions"

mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$RUN3_DIR"
```

---

### 1. phase1_global: global baseline restart (natveg_improved copy)

**Goal:** create a clearly labeled baseline restart identical to `natveg_improved`.

```bash
cp cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc \
   "$RUN1_DIR/updated_restart_phase1_global_natveg_improved.nc"
```

Optional README:

```bash
echo "phase1_global: copy of natveg_improved restart on $(date)" \
  > "$RUN1_DIR/README_phase1_global.txt"
```

---

### 2. phase2_tropical: tropical restart from raw Phase2 predictions

**Goal:** overwrite all 5P variables in the tropical band \([-30°, 30°]\)
using **raw Phase2 predictions**.

#### 2.1 Tropical-only Phase2 inference (if not already done)

```bash
cd cnp_results/run_20260305_153217_phase2_pvariable_focus

python ../../scripts/run_inference_all.py \
  --model cnp_predictions/model.pth \
  --output-dir cnp_inference_tropical_only

cd -  # back to repo root
```

#### 2.2 Build NetCDF from tropical predictions

```bash
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_inference_tropical_only/cnp_predictions \
  --output cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_tropical_only.nc
```

#### 2.3 Create phase2_tropical restart (5P in tropics, raw Phase2)

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions cnp_results/run_20260305_153217_phase2_pvariable_focus/comparison_results/ai_predictions_tropical_only.nc \
  --restart-file cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc \
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

### 3. phase3_tworegions: tropical restart with two-region 5P bias/scale

**Goal:** use Phase2 predictions plus **Amazon + Africa 5P bias/scale correction**
and map those corrected 5P values into the tropical band of the natveg restart.

This follows **Path B** from
`WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md`, but writes outputs into
`RUN3_DIR`.

#### 3.1 Full-grid Phase2 inference into RUN3_DIR

```bash
python scripts/run_inference_all.py \
  --model cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_predictions/model.pth \
  --output-dir "$RUN3_DIR/cnp_inference_entire_dataset" \
  --inference-full-grid
```

#### 3.2 Apply 5P bias/scale correction (Amazon + Africa only)

```bash
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_two_region_five_p.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_phase2
```

Outputs:

- Corrected 5P CSVs under  
  `"$RUN3_DIR/cnp_inference_entire_dataset/cnp_predictions/soil_2d_predictions_5P_bias_corrected_phase2/"`
- Bias/scale parameters under  
  `"$RUN3_DIR/analysis/bias_scale_params_5P_two_regions.json"`

#### 3.3 Build global NetCDF from bias-corrected 5P predictions

```bash
mkdir -p "$RUN3_DIR/comparison_results"

python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$RUN3_DIR/cnp_inference_entire_dataset/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_phase2 \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --output "$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc"
```

#### 3.4 Create phase3_tworegions restart (bias-corrected 5P in tropics)

```bash
python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_phase2.nc" \
  --restart-file cnp_results/run_20260228_214757_natveg_improved/updated_restart_CNP_IO_updated9_dev_dw_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc \
  --output "$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_tropical.nc" \
  --variable-list CNP_IO_updated9_dev_dw.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

Optional README:

```bash
echo "phase3_tworegions: natveg base + Phase2 5P with Amazon+Africa bias/scale in tropics on $(date)" \
  > "$RUN3_DIR/README_phase3_tworegions.txt"
```

---

### 4. Automation script

The companion script `scripts/run_phase_restarts.sh` wraps the commands above.
By default it:

- Uses the natveg and Phase2 paths listed in this document.
- Creates three new run directories with a timestamp.
- Produces:
  - `updated_restart_phase1_global_natveg_improved.nc`
  - `updated_restart_phase2_tropical_5P_raw.nc`
  - `updated_restart_phase3_tworegions_5P_bias_corrected_tropical.nc`

See `scripts/run_phase_restarts.sh` for details and optional overrides.

