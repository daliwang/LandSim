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
- **Phase2 run**: `cnp_results/run_20260305_153217_phase2_pvariable_focus`
- **Variable list file**: `CNP_IO_updated9_dev_dw.txt` in repo root.

If you are a new user and want to **train your own pair of models** instead of
reusing these runs, follow these steps first.

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

#### 0.2 Train a phase2_tropical‑like P‑focused tropical model

From the repo root:

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

python train_cnp_model.py \
  --config config/training_config_phase2_tropical_soilp_only.json \
  --run-dir cnp_results/run_YYYYMMDD_HHMMSS_phase2_pvariable_focus_custom \
  --variable-list CNP_IO_updated9_dev_dw.txt
```

Then set:

```bash
export PHASE2_RUN_DIR="cnp_results/run_YYYYMMDD_HHMMSS_phase2_pvariable_focus_custom"
```

The remaining inference, bias/scale, and restart steps in this document are the
same; only `NATVEG_RUN_DIR`, `BASE_RESTART`, and `PHASE2_RUN_DIR` change to
point at your newly trained models.

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
   - Trains the tropical Phase2 P‑focused model (or reuses an existing one).  
   - Runs tropical‑only inference and creates the **phase2_tropical** restart
     with raw Phase2 5P in the tropics, using `BASE_RESTART` from phase 1.  
   - Exports `PHASE2_RUN_DIR` and prints the path to the tropical restart.

3. `scripts/run_phase3_tworegions.sh`  
   - Runs full‑grid inference with the Phase2 model.  
   - Applies two‑region 5P bias/scale correction (Amazon + Africa).  
   - Creates the **phase3_tworegions** tropical restart with bias‑corrected
     Phase2 5P, using the same `BASE_RESTART`.

Between scripts you can:

- Inspect logs, validation plots, and diagnostics for each model/run.
- Adjust configs or weights and re‑run the specific phase if needed.

Each script is self‑contained and safe to re‑run; they will reuse existing runs
when possible instead of retraining from scratch.

