# E3SMv3_h0: Phase 1 (global), Phase 2 (tropical), Phase 3 (two-region bias correction)

This runbook repeats the three-phase workflow used for the Trendy reference runs, using the training dataset at `/mnt/proj-shared/AI4BGC_7xw/TrainingData/E3SMv3_h0` (396 `training_data_batch_*.pkl` files).

Use git branch **`e3smv3case`** (includes E3SM-specific training fixes, commit `92473ed` or later).

**Trendy reference runs**

| Phase | Reference directory |
|-------|---------------------|
| Phase 1 global | `cnp_results/run_20260315_113900_phase1_global` |
| Phase 2 tropical | `cnp_results/run_20260315_175250_phase2_tropical` |
| Phase 3 two-region | `cnp_results/run_20260512_083650_phase3_tworegions_(africa_improvement)` |

**E3SMv3_h0 runs (Jun 2026 experiment)**

| Phase | Run directory | Notes |
|-------|---------------|-------|
| Phase 1 global | `cnp_results/run_20260623_172201_e3smv3_h0_phase1_global` | 120 epochs, ~292 min train; inference with `--derive-np-from-c --inference-batch-size 4096` |
| Phase 2 tropical | `cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical` | 150 epochs, ~141 min train; tropical inference ~105k cells |
| Phase 3 two-region | `cnp_results/run_20260624_153307_e3smv3_h0_phase3_tworegions` | ~54 min total; full-grid inference 395784 cells (batch 4096) → Amazon/Africa v3 → merge |

**Pipeline scripts and config**

| File | Purpose |
|------|---------|
| `CNP_IO_e3smv3_h0.txt` | Variable list for E3SMv3_h0 (see **dataset differences** below) |
| `config/e3smv3_h0_env.sh` | Data paths, configs, restart template, inference batch size |
| `scripts/run_e3smv3_h0_phase1_global.sh` | Phase 1 train → inference → base restart |
| `scripts/run_e3smv3_h0_phase2_tropical.sh` | Phase 2 train → tropical inference → phase2 restart |
| `scripts/run_e3smv3_h0_phase3_tworegions.sh` | Phase 3 full-grid inference → v3 bias correction → merge → restart |
| `scripts/run_e3smv3_h0_validation.sh` | Post-run validation and site comparisons |

Training configs match the Trendy runs:

- Phase 1: `config/training_config_experiment_3_global_natveg_improved.json` (120 epochs)
- Phase 2: `config/training_config_phase2_tropical_soilp_only.json` (150 epochs, tropical-only filter)

### E3SMv3_h0 vs Trendy variable list (`CNP_IO_e3smv3_h0.txt`)

The E3SM batches use the same PFT / soil / forcing layout as Trendy, but two groups differ:

| Variable | Trendy (`CNP_IO_updated9_dev_dw.txt`) | E3SMv3_h0 |
|----------|---------------------------------------|-----------|
| `landfrac` | present | **absent** — use `LANDFRAC_PFT` instead (48 surface vars) |
| `GPP`, `NPP`, `AR`, `HR` | present | **absent** — scalar section is empty (0 scalar inputs/outputs) |

Code on `e3smv3case` adapts training for zero scalar variables (`scalar_output_size` from the IO list, skip scalar loss/metrics, static group-embedding fix).

**Batch ordering:** files are named `training_data_batch_01.pkl` … `396.pkl`. Early batches (e.g. 01–03) are Antarctic ice with `PCT_NATVEG=0`; natveg training needs later batches. For a quick smoke test, symlink batches with land (e.g. 110–112) — see §0.1.

### Inference defaults (`config/e3smv3_h0_env.sh`)

| Variable | Default | Used by |
|----------|---------|---------|
| `INFERENCE_BATCH_SIZE` | `4096` | All inference steps (~396k or ~105k E3SM gridcells) |
| `PHASE1_INFERENCE_EXTRA_FLAGS` | `--derive-np-from-c --inference-batch-size 4096` | Phase 1 script |
| `INFERENCE_EXTRA_FLAGS` | `--no-derive-np-from-c --inference-batch-size 4096` | Documentation / manual runs |

Phase 2 and Phase 3 scripts pass **explicit** `--no-derive-np-from-c --inference-batch-size "${INFERENCE_BATCH_SIZE}"` to `run_inference_all.py` (avoids bash line-continuation bugs with unquoted `$INFERENCE_EXTRA_FLAGS` on its own line).

Override batch size if GPU OOM persists: `export INFERENCE_BATCH_SIZE=2048` before `source config/e3smv3_h0_env.sh`.

**Model checkpoint path:** training saves `cnp_predictions/model.pth`. `run_inference_all.py` resolves `cnp_model.pt` → `cnp_predictions/model.pth` automatically.

---

## 0. One-time setup

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC
git checkout e3smv3case
source config/e3smv3_h0_env.sh

# ELM restart template (default in config/e3smv3_h0_env.sh):
export RESTART_TEMPLATE=/mnt/proj-shared/AI4BGC_7xw/AI4BGC/ELM_data/E3SMV3_025/20240214.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.adsp.elm.r.0021-01-01-00000.nc

# Optional timestamp for Phase 3 run dir:
export TS=$(date +%Y%m%d_%H%M%S)

mkdir -p logs
chmod +x scripts/run_e3smv3_h0_*.sh
```

Verify the dataset:

```bash
ls "$E3SM_DATA_DIR"/training_data_batch_*.pkl | wc -l   # expect 396
```

### 0.1 Quick smoke test (3 batches, 5 epochs)

Before the full 396-batch run, verify the pipeline on three natveg batches (~38 s on GPU in a Jun 2026 test):

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC
source config/e3smv3_h0_env.sh

SMOKE_DIR=/tmp/e3smv3_h0_smoke_3natveg
mkdir -p "$SMOKE_DIR"
# Batches 110–112 have natveg land cells; batches 01–03 are polar ice only.
for bn in 110 111 112; do
  ln -sf "$E3SM_DATA_DIR/training_data_batch_${bn}.pkl" "$SMOKE_DIR/"
done

python train_cnp_model.py \
  --training-config-json config/training_config_experiment_3_global_natveg_improved.json \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --data-paths "$SMOKE_DIR" \
  --file-pattern training_data_batch_*.pkl \
  --epochs 5 \
  --output-dir cnp_results \
  --output-dir-suffix e3smv3_h0_smoketest_3files_5ep
```

Example result: `cnp_results/run_20260623_171716_e3smv3_h0_smoketest_3files_5ep` — train loss ~1.97 → 0.43 over 5 epochs (1798 train / 600 val samples).

Do **not** use `--max-files 3` on the full `E3SMv3_h0` directory alone; that selects batches 01–03 and yields zero natveg training rows.

---

## 1. Phase 1 — global model + base restart

**Goal:** Train global natveg model on E3SMv3_h0, run full-grid inference, write `updated_restart_base.nc` from the ELM restart template.

**Inference:** `--derive-np-from-c --inference-batch-size 4096` (required for ~396k gridcells).

### Option A — automation script

```bash
source config/e3smv3_h0_env.sh

# Full Phase 1 (training + inference + base restart):
nohup bash scripts/run_e3smv3_h0_phase1_global.sh \
  > logs/e3smv3_h0_phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Train only (no inference/restart):
# TRAIN_ONLY=1 bash scripts/run_e3smv3_h0_phase1_global.sh

# Skip training — resume inference + restart:
# export NATVEG_RUN_DIR=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase1_global
# bash scripts/run_e3smv3_h0_phase1_global.sh
```

### Option B — manual commands

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC
source config/e3smv3_h0_env.sh

# 1.1 Train (long — 396 batches)
python train_cnp_model.py \
  --training-config-json config/training_config_experiment_3_global_natveg_improved.json \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --data-paths /mnt/proj-shared/AI4BGC_7xw/TrainingData/E3SMv3_h0 \
  --file-pattern training_data_batch_*.pkl \
  --output-dir cnp_results \
  --output-dir-suffix e3smv3_h0_phase1_global

export NATVEG_RUN_DIR=$(ls -td cnp_results/run_*_e3smv3_h0_phase1_global | head -1)

# 1.2 Full-grid inference (batched; required for ~396k E3SM gridcells)
python scripts/run_inference_all.py \
  --model "$NATVEG_RUN_DIR/cnp_predictions/model.pth" \
  --output-dir "$NATVEG_RUN_DIR/cnp_inference_entire_dataset" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --inference-full-grid \
  --derive-np-from-c \
  --inference-batch-size 4096

# 1.3 NetCDF + base restart (requires RESTART_TEMPLATE)
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$NATVEG_RUN_DIR/cnp_inference_entire_dataset/cnp_predictions" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --output "$NATVEG_RUN_DIR/ai_predictions_global.nc"

python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$NATVEG_RUN_DIR/ai_predictions_global.nc" \
  --restart-file "$RESTART_TEMPLATE" \
  --output "$NATVEG_RUN_DIR/updated_restart_base.nc" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  "--tropical-lat-range=-90,90"

export BASE_RESTART="$NATVEG_RUN_DIR/updated_restart_base.nc"
```

**Outputs:** `cnp_predictions/model.pth`, `cnp_inference_entire_dataset/predictions.pkl`, `ai_predictions_global.nc`, `updated_restart_base.nc`.

**Success check:** log ends with `Phase 1 complete.` and `updated_restart_base.nc` exists (~53 GB).

---

## 2. Phase 2 — tropical P-focused model + raw 5P restart

**Goal:** Train tropical-only model; overwrite five soil P variables in \([-30°, 30°]\) on top of Phase 1 base restart.

**Inference:** `--no-derive-np-from-c --inference-batch-size 4096` (required — unbatched run OOM'd on ~105k tropical cells).

### Option A — automation script

```bash
source config/e3smv3_h0_env.sh
export BASE_RESTART=cnp_results/run_20260623_172201_e3smv3_h0_phase1_global/updated_restart_base.nc

nohup bash scripts/run_e3smv3_h0_phase2_tropical.sh \
  > logs/e3smv3_h0_phase2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Resume / skip options (script auto-detects):**

| Set | Skips |
|-----|-------|
| `PHASE2_RUN_DIR=...` | Training |
| Existing `cnp_inference_tropical_only/predictions.pkl` | Inference |
| Existing `updated_restart_phase2_tropical_5P_raw.nc` + NetCDF | NetCDF + restart |

```bash
# Example: skip training and inference, run NetCDF + restart only:
export PHASE2_RUN_DIR=cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical
export BASE_RESTART=cnp_results/run_20260623_172201_e3smv3_h0_phase1_global/updated_restart_base.nc
bash scripts/run_e3smv3_h0_phase2_tropical.sh
```

### Option B — manual commands

```bash
source config/e3smv3_h0_env.sh
export BASE_RESTART=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase1_global/updated_restart_base.nc

python train_cnp_model.py \
  --training-config-json config/training_config_phase2_tropical_soilp_only.json \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --data-paths /mnt/proj-shared/AI4BGC_7xw/TrainingData/E3SMv3_h0 \
  --file-pattern training_data_batch_*.pkl \
  --output-dir cnp_results \
  --output-dir-suffix e3smv3_h0_phase2_tropical

export PHASE2_RUN_DIR=$(ls -td cnp_results/run_*_e3smv3_h0_phase2_tropical | head -1)

python scripts/run_inference_all.py \
  --model "$PHASE2_RUN_DIR/cnp_predictions/model.pth" \
  --output-dir "$PHASE2_RUN_DIR/cnp_inference_tropical_only" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --no-derive-np-from-c \
  --inference-batch-size 4096

python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$PHASE2_RUN_DIR/cnp_inference_tropical_only/cnp_predictions" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --output "$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc"

python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$PHASE2_RUN_DIR/comparison_results/ai_predictions_tropical_only.nc" \
  --restart-file "$BASE_RESTART" \
  --output "$PHASE2_RUN_DIR/updated_restart_phase2_tropical_5P_raw.nc" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

**Outputs:** `comparison_results/ai_predictions_tropical_only.nc`, `updated_restart_phase2_tropical_5P_raw.nc`.

**Success check:** log ends with `Phase 2 complete.` and restart file exists (~53 GB).

---

## 3. Phase 3 — two-region bias correction (regional_fit_v3)

**Goal:** Match `run_20260512_083650_phase3_tworegions_(africa_improvement)`: Amazon v3 + Africa v3 + merge → global restart with bias-corrected 5P in tropics.

**Inference:** full-grid with `--no-derive-np-from-c --inference-batch-size 4096` (same ~396k grid as Phase 1).

```bash
source config/e3smv3_h0_env.sh
export BASE_RESTART=cnp_results/run_20260623_172201_e3smv3_h0_phase1_global/updated_restart_base.nc
export PHASE2_RUN_DIR=cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical

nohup bash scripts/run_e3smv3_h0_phase3_tworegions.sh \
  > logs/e3smv3_h0_phase3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

The Phase 3 script skips full-grid inference if `$RUN3_DIR/cnp_inference_entire_dataset/cnp_predictions` already exists.

Manual steps (same as `docs/RERUN_PHASE2_PHASE3_NO_CNP_FIX.md` §3):

```bash
export RUN3_DIR=cnp_results/run_${TS}_e3smv3_h0_phase3_tworegions
mkdir -p "$RUN3_DIR"

# 3.1 Full-grid inference (longest step; batched for ~396k E3SM gridcells)
python scripts/run_inference_all.py \
  --model "$PHASE2_RUN_DIR/cnp_predictions/model.pth" \
  --output-dir "$RUN3_DIR/cnp_inference_entire_dataset" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --inference-full-grid \
  --no-derive-np-from-c \
  --inference-batch-size 4096

# 3.2 Amazon v3 bias correction
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_amazon_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_amazon_v3 \
  --regional-fit-v3

# 3.3 Africa v3 bias correction
python scripts/apply_5p_bias_scale_correction.py \
  --run-dir "$RUN3_DIR" \
  --region-config-json config/training_config_africa_5p_box.json \
  --output-subdir soil_2d_predictions_5P_bias_corrected_africa_v3 \
  --regional-fit-v3

# 3.4 Merge
python scripts/merge_5p_bias_corrected_amazon_africa.py \
  --run-dir "$RUN3_DIR" \
  --amazon-subdir soil_2d_predictions_5P_bias_corrected_amazon_v3 \
  --africa-subdir soil_2d_predictions_5P_bias_corrected_africa_v3 \
  --output-subdir soil_2d_predictions_5P_bias_corrected_amazon_africa_v3 \
  --output-filename-suffix _bias_corrected_amazon_africa_v3

# 3.5 NetCDF + final restart
python scripts/ai_predictions_to_netcdf.py \
  --ai-predictions "$RUN3_DIR/cnp_inference_entire_dataset/cnp_predictions/" \
  --soil-2d-bias-corrected-subdir soil_2d_predictions_5P_bias_corrected_amazon_africa_v3 \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --output "$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa_v3.nc"

python scripts/ai_predictions_to_restart.py \
  --ai-predictions "$RUN3_DIR/comparison_results/ai_predictions_5P_bias_corrected_amazon_africa_v3.nc" \
  --restart-file "$BASE_RESTART" \
  --output "$RUN3_DIR/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc" \
  --variable-list CNP_IO_e3smv3_h0.txt \
  --variables-to-update labilep_vr,occlp_vr,solutionp_vr,secondp_vr,primp_vr \
  "--tropical-lat-range=-30,30"
```

**Final restart (Jun 2026 run):**  
`cnp_results/run_20260624_153307_e3smv3_h0_phase3_tworegions/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc` (~53 GB)

Log: `logs/e3smv3_h0_phase3_20260624_153307.log`

---

## 4. Validation

After all three phases complete:

```bash
source config/e3smv3_h0_env.sh
export NATVEG_RUN_DIR=cnp_results/run_20260623_172201_e3smv3_h0_phase1_global
export PHASE2_RUN_DIR=cnp_results/run_20260624_092639_e3smv3_h0_phase2_tropical
export RUN3_DIR=cnp_results/run_20260624_153307_e3smv3_h0_phase3_tworegions

bash scripts/run_e3smv3_h0_validation.sh
```

Individual checks:

| Check | Command |
|-------|---------|
| Training loss / metrics | Inspect `$NATVEG_RUN_DIR/cnp_training_*.log`, `cnp_training_losses.csv` |
| NPOOL/PPOOL per PFT | `python scripts/validation_npool_ppool_per_pft.py $NATVEG_RUN_DIR` |
| 5P pred vs GT (Amazon/Africa) | `python scripts/compare_5p_gt_two_regions_inference.py pred-eval --runs-json $RUN3_DIR/analysis/5p_pred_eval_runs.json` |
| Site vertical profiles | `python scripts/generate_site_5p_restart_comparison.py --lon 28 --lat 0 --phase3-run-dir $RUN3_DIR` |
| Bias parameters | `$RUN3_DIR/analysis/bias_scale_params_5P_regional_fit_v3_*.json` |

---

## 5. Suggested launch order

Run phases **sequentially**; inspect each log before starting the next.

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC
source config/e3smv3_h0_env.sh
mkdir -p logs

# Step 1: Phase 1
nohup bash scripts/run_e3smv3_h0_phase1_global.sh \
  > logs/e3smv3_h0_phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Wait for "Phase 1 complete."
export NATVEG_RUN_DIR=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase1_global
export BASE_RESTART=$NATVEG_RUN_DIR/updated_restart_base.nc

# Step 2: Phase 2 (after Phase 1 complete)
nohup bash scripts/run_e3smv3_h0_phase2_tropical.sh \
  > logs/e3smv3_h0_phase2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Wait for "Phase 2 complete."
export PHASE2_RUN_DIR=cnp_results/run_YYYYMMDD_HHMMSS_e3smv3_h0_phase2_tropical

# Step 3: Phase 3 (after Phase 2 complete)
nohup bash scripts/run_e3smv3_h0_phase3_tworegions.sh \
  > logs/e3smv3_h0_phase3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
export RUN3_DIR=cnp_results/run_20260624_153307_e3smv3_h0_phase3_tworegions

# Step 4: validation
bash scripts/run_e3smv3_h0_validation.sh
```

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `canUse32BitIndexMath ... false` during Phase 1 inference | Full ~396k grid in one GPU forward pass | Add `--inference-batch-size 4096` (default in scripts) |
| `CUDA out of memory` during Phase 2 inference | ~105k tropical cells unbatched | Add `--inference-batch-size 4096`; try `2048` if needed |
| `able-list: command not found` in Phase 2 script | Bash misparsed multiline `python` command when `$INFERENCE_EXTRA_FLAGS` was on its own line | Fixed in `run_e3smv3_h0_phase2_tropical.sh` — uses explicit flags |
| Zero natveg training samples in smoke test | `--max-files 3` picks polar batches 01–03 | Symlink land batches (110–112); see §0.1 |
| Training works but scalar loss crashes | E3SM has no `GPP`/`NPP`/`AR`/`HR` | Use `e3smv3case` branch + `CNP_IO_e3smv3_h0.txt` |
| Inference succeeded but restart step missing | Script exited early (`set -e`) | Re-run script with `PHASE2_RUN_DIR` set; inference auto-skipped |

**Monitoring logs:**

```bash
tail -f logs/e3smv3_h0_phase1_*.log
grep -E 'Phase [123] complete|ERROR|failed' logs/e3smv3_h0_phase*.log
```

---

## Notes

- **Experiment status (Jun 2026):** All three phases completed on E3SMv3_h0. Final ELM restart for tropical 5P bias correction: `run_20260624_153307_e3smv3_h0_phase3_tworegions/updated_restart_phase3_tworegions_5P_bias_corrected_amazon_africa_v3_tropical.nc`.
- **Training time:** 396 batches × ~1000 gridcells each. Phase 1 train ~5 h, Phase 2 train ~2.5 h (Jun 2026 GPU node). Use `nohup` or a job scheduler.
- **ELM restart:** `RESTART_TEMPLATE` defaults to `ELM_data/E3SMV3_025/20240214.lndr025_trigrid_top_bgc.IcoswISC30E3r5.chrysalis.adsp.elm.r.0021-01-01-00000.nc` (not committed — large file).
- **Phase 1 vs Phase 2/3 inference:** Phase 1 uses `--derive-np-from-c`; Phase 2/3 use `--no-derive-np-from-c` (see `docs/RERUN_PHASE2_PHASE3_NO_CNP_FIX.md`).
- **Scalar fluxes:** E3SMv3_h0 has no scalar flux columns; model trains PFT 1D + soil 2D only.
- **Phase 3 alternative:** For selective v3 (only `primp_vr` corrected), see `scripts/run_phase3_path_a_selective_v3.sh` and `docs/PHASE3_PATH_A_SELECTIVE_V3.md`.

**See also:** `docs/WORKFLOW_PHASE1_PHASE2_PHASE3_RESTARTS.md`, `docs/WORKFLOW_5P_TWO_REGIONS_BIAS_SCALE_AND_RESTARTS.md`
