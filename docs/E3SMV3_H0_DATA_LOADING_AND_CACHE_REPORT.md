# E3SMv3_h0 data loading performance and preprocessed tensor cache

**Date:** 2026-06-25  
**Branch:** `e3smv3case`  
**Dataset:** `/mnt/proj-shared/AI4BGC_7xw/TrainingData/E3SMv3_h0` (396 × `training_data_batch_*.pkl`, 1000 gridcells each)

## Summary

Phase 1 global training on the full E3SM grid spent roughly **40 minutes** in data preparation before epoch 1 (pickle load ~19 min, preprocess ~10 min, normalize ~10 min). Training itself dominated wall time (~247 min for 120 epochs at batch 128).

This change set adds:

1. **Parallel pickle loading** (`LOAD_WORKERS`, default 8 on shared storage)
2. **Pipeline training batch size 1024** for E3SM scripts (`TRAINING_BATCH_SIZE`; global default in `train_cnp_model.py` remains 128)
3. **Opt-in preprocessed tensor cache** to skip load + preprocess + normalize + split on reruns

## Baseline timing (Phase 1, `run_20260623_172201`)

From `cnp_results/run_20260623_172201_e3smv3_h0_phase1_global/cnp_training_20260623_172201.log`:

| Stage | Duration | % of ~290 min total |
|-------|----------|---------------------|
| Pickle load (396 files, sequential) | 19.1 min | 6.6% |
| Preprocess | 9.9 min | 3.4% |
| Normalize (individual scalers) | 9.9 min | 3.4% |
| Split / tensor prep | 2.2 min | 0.8% |
| Training (120 epochs) | 247.2 min | 85.3% |

Pickle load was skewed: median ~1.3 s/file, but a few files took 60–150 s (shared-filesystem latency).

## Implementation

### Parallel pickle I/O

- **Module:** `data/data_loader_individual.py`
- **Config:** `DataConfig.load_workers` (`None` = auto, up to 8 threads on multi-file datasets)
- **Env:** `LOAD_WORKERS` (default `8` in `config/e3smv3_h0_env.sh`)
- **Behavior:** Thread pool reads pickles in parallel; concat order matches sorted file list

### E3SM training batch size

- **Env:** `TRAINING_BATCH_SIZE=1024` in `config/e3smv3_h0_env.sh`
- **Scripts:** Phase 1/2 pass `--batch-size "${TRAINING_BATCH_SIZE}"` to `train_cnp_model.py`
- **Rationale:** H100/H200 have headroom; batch 128 under-utilizes GPU (~1630 steps/epoch vs ~204 at 1024)

### Preprocessed tensor cache

- **Module:** `data/preprocessed_cache.py`
- **What is cached:** train/val tensors after normalize + split, fitted scalers, `data_info`
- **What is not cached:** raw pickle I/O on first build (still required once)
- **Invalidation:** SHA-256 fingerprint over data file mtimes/sizes, variable lists, filters (`natveg_only`, `tropical_only`, …), `train_split`, `random_state`, normalization method, `pft_presence_threshold`
- **Files per entry** (under `PREPROCESSED_CACHE_DIR`):
  - `{fingerprint}.pt` — tensors
  - `{fingerprint}.scalers.pkl` — scalers
  - `{fingerprint}.meta.json` — manifest (human-readable)

**Default cache directory:** `$E3SM_DATA_DIR/.preprocessed_cache`  
(i.e. `/mnt/proj-shared/AI4BGC_7xw/TrainingData/E3SMv3_h0/.preprocessed_cache`)

Phase 1 (global) and Phase 2 (tropical) produce **different fingerprints** because `tropical_only` and training JSON configs differ.

## Usage

### Build cache (no training)

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC
source config/e3smv3_h0_env.sh

# Phase 1 global only
bash scripts/build_e3smv3_preprocessed_cache.sh global

# Phase 2 tropical only
bash scripts/build_e3smv3_preprocessed_cache.sh tropical

# Both (sequential)
bash scripts/build_e3smv3_preprocessed_cache.sh both
```

Equivalent manual flags:

```bash
python train_cnp_model.py \
  --training-config-json "$CONFIG_GLOBAL" \
  --variable-list "$VARIABLE_LIST" \
  --data-paths "$DATA_PATHS" \
  --file-pattern "$FILE_PATTERN" \
  --use-preprocessed-cache \
  --preprocessed-cache-dir "$PREPROCESSED_CACHE_DIR" \
  --rebuild-preprocessed-cache \
  --preprocessed-cache-only \
  --output-dir cnp_results \
  --output-dir-suffix cache_build_phase1_global
```

### Use cache during training

```bash
export USE_PREPROCESSED_CACHE=1
source config/e3smv3_h0_env.sh
bash scripts/run_e3smv3_h0_phase1_global.sh
```

CLI alternative: `--use-preprocessed-cache` on `train_cnp_model.py`.

Force rebuild: `export REBUILD_PREPROCESSED_CACHE=1` or `--rebuild-preprocessed-cache`.

### Environment variables (`config/e3smv3_h0_env.sh`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOAD_WORKERS` | `8` | Parallel pickle read threads |
| `TRAINING_BATCH_SIZE` | `1024` | E3SM Phase 1/2 training batch size |
| `USE_PREPROCESSED_CACHE` | `0` | Opt-in cache for training |
| `PREPROCESSED_CACHE_DIR` | `$E3SM_DATA_DIR/.preprocessed_cache` | Cache storage |
| `REBUILD_PREPROCESSED_CACHE` | unset | Set to `1` to force refresh |

## Cache build run (2026-06-25)

A full cache build was started with:

```bash
nohup bash scripts/build_e3smv3_preprocessed_cache.sh both \
  > logs/e3smv3_cache_build_20260625_133408.log 2>&1 &
```

- **Phase 1 fingerprint (global):** `41d7cf8d98da…`
- **Log:** `logs/e3smv3_cache_build_20260625_133408.log`
- **Expected runtime:** ~40–80 min total (global + tropical), depending on shared-storage load

Verify completion:

```bash
grep -E "Cache build complete|Preprocessed cache ready" logs/e3smv3_cache_build_20260625_133408.log
ls -lh "$PREPROCESSED_CACHE_DIR"
```

After success, enable `USE_PREPROCESSED_CACHE=1` for subsequent Phase 1/2 training runs.

## Files changed

| File | Role |
|------|------|
| `data/preprocessed_cache.py` | Fingerprint, save/load |
| `data/data_loader_individual.py` | Parallel file loading |
| `train_cnp_model.py` | Cache integration, CLI flags |
| `config/training_config.py` | `load_workers` on `DataConfig` |
| `config/e3smv3_h0_env.sh` | Env defaults |
| `scripts/build_e3smv3_preprocessed_cache.sh` | Cache-only build helper |
| `scripts/run_e3smv3_h0_phase1_global.sh` | Training batch + cache env logging |
| `scripts/run_e3smv3_h0_phase2_tropical.sh` | Same |
| `scripts/run_inference_all.py` | `LOAD_WORKERS` for inference data load |

## Expected impact on reruns

| Optimization | First run | Repeat runs (same data/config) |
|--------------|-----------|-------------------------------|
| Parallel load (`LOAD_WORKERS=8`) | Moderate speedup vs sequential | Same |
| `TRAINING_BATCH_SIZE=1024` | Faster epochs (fewer steps) | Same |
| `USE_PREPROCESSED_CACHE=1` | Full pipeline + cache write | **Skip ~20–40 min** data prep |
