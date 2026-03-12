# LandSim (model_tokenization_dev): Software Description and Overview

This overview describes the **model_tokenization_dev** branch, which introduces a **tokenization-based** encoder–backbone–decoder architecture in place of the earlier LSTM + separate encoders design.

---

## Two-paragraph summary (for short academic papers, ~200 words)

LandSim (v0.1, formerly AI4BGC) is an open-source deep learning framework for modeling terrestrial ecosystem dynamics with a focus on the Carbon–Nitrogen–Phosphorus (CNP) cycle in the E3SM Land Model (ELM). The **model_tokenization_dev** branch implements a **token-based** architecture: heterogeneous ELM inputs (time series, static surface and PFT attributes, and 1D/2D state variables) are first mapped into **identity-aware tokens** (variable and group embeddings), then fused by a shared Transformer backbone. A central capability is ingesting AI-predicted states into ELM restart files, enabling stable, accelerated spin-up and scenario experiments without replacing the physical model. LandSim supports 1° and 0.5° resolutions and is designed for reuse in large-domain and km-scale settings (e.g. kmELM, kiloCraft), with configurable variable lists (CNP_IO) and optional mixed-precision training.

The workflow comprises configuration, training on enhanced CNP datasets with per-variable or group normalization, validation and prediction-quality reporting, full-dataset inference, NetCDF export, comparison with ELM outputs, restart-file ingestion of PFT 1D and soil 2D variables, and restart verification. The core model is **two-stream tokenization plus Transformer**: (1) **ForcingTemporalEncoder**—time series is patched (grouped Conv1d), then augmented with variable ID, time position (sin-cos), and lat/lon embeddings to produce dynamic tokens; (2) **StaticVariableEncoder**—static inputs are grouped (single- or multi-variable per token), projected to a common embedding dimension, and augmented with variable and group ID embeddings to produce static tokens. Dynamic and static token sequences are concatenated, passed through a Transformer encoder, then **global mean pooling** yields a single vector fed to task-specific heads (scalars, PFT 1D, soil 2D, optional water). Training uses a composite loss (task-weighted MSE and optional physics or learnable weights), MinMax normalization, and optional automatic mixed precision.

---

## Short description

**LandSim** (v0.1, formerly AI4BGC) is an open-source deep learning framework for modeling terrestrial ecosystem dynamics with a focus on the **Carbon–Nitrogen–Phosphorus (CNP) cycle** in the E3SM Land Model (ELM). On the **model_tokenization_dev** branch, the model uses a **tokenization-based** design: a **ForcingTemporalEncoder** turns time-series forcings into patches with variable and time position embeddings; a **StaticVariableEncoder** turns static, PFT, scalar, and 2D inputs into tokens with variable and group identity embeddings. These token sequences are concatenated and processed by a **Transformer encoder**; a **global mean pool** over the token dimension then feeds **multi-task heads** (scalars, PFT 1D, soil 2D, optional water). The pipeline remains config-driven (CNP_IO, optional model-config overrides), with training, validation, inference, NetCDF export, comparison, and **ELM restart ingestion and verification** unchanged. LandSim supports 1° and 0.5° resolutions and is reusable for large-domain and km-scale (e.g. kmELM, kiloCraft) experiments.

---

## Workflow overview

End-to-end flow (CNP pipeline) is unchanged from the main branch:

```
1. Configuration   → 2. Training  → 3. Validation  → 4. Inference  → 5. NetCDF export  → 6. Comparison  → 7. Restart ingestion  → 8. Restart verification
```

| Stage | Description |
|-------|-------------|
| **1. Configuration** | Define variable lists in a `CNP_IO_*.txt` file. Optional `--model-config` overrides architecture (e.g. `embed_dim`, `patch_size`, `transformer_layers`, `transformer_heads`). |
| **2. Training** | `train_cnp_model.py` loads enhanced CNP datasets, normalizes with per-variable or group scalers, trains the tokenization-based CNP model, saves model, scalers, and metrics under `cnp_results/run_YYYYMMDD_HHMMSS/`. |
| **3. Validation** | Scripts such as `cnp_result_validationplot.py` and `generate_prediction_quality_report.py` evaluate test-set predictions (RMSE, R², quality categories). |
| **4. Inference** | `run_inference_all.py` runs the trained model on the full dataset and denormalizes with saved scalers; optional `fix_pft1d_nans.py` for post-processing. |
| **5. NetCDF export** | `ai_predictions_to_netcdf.py` aggregates predictions to a single NetCDF aligned to the ELM grid. |
| **6. Comparison** | `ai_model_comparison_plot.py` produces tri-panel plots (AI vs ELM vs difference). |
| **7. Restart ingestion** | `ai_predictions_to_restart.py` writes AI predictions into ELM restart NetCDF (PFT 1D and soil 2D variables). |
| **8. Restart verification** | `ai_restart_comparison.py` compares the AI-updated restart with the original. |

**Training data generation** (under `scripts/training_data_generation/`): `construct_forcing_20years.py` builds forcing NetCDFs; `enhanced_training_dataset.py` produces forcing-only, full enhanced ecosystem, or initial-condition-only datasets in batch pickle format.

---

## Architecture overview (tokenization-based)

The **model_tokenization_dev** branch replaces the previous LSTM + separate encoders + concatenation with a **two-stream tokenization + Transformer** design in `models/cnp_combined_model.py`:

### Stream 1: ForcingTemporalEncoder (dynamic variables)

- **Input:** Time series `[B, T, V]` (e.g. 6 forcing variables over 240 months) and lat/lon `[B, 2]`.
- **Patching:** Grouped 1D convolution (per-variable) with configurable `patch_size` (e.g. 60 months) → `[B, V, N_patches, embed_dim]`.
- **Embeddings:** Variable ID embedding (sin-cos over variable index), time position embedding (sin-cos over patch index), and a small MLP on lat/lon added to each patch.
- **Output:** Flattened token sequence `[B, V × N_patches, embed_dim]` (dynamic tokens).

### Stream 2: StaticVariableEncoder (static variables)

- **Input:** Flattened static vector (surface, PFT params, scalars, water, PFT 1D, soil 2D) with a pre-defined **group structure** (`input_group_indices`, `group_ids`).
- **Token generation:** Each group entry is either (a) a **single variable** → 1×1 Conv (per-variable projection to `embed_dim`) or (b) a **multi-variable group** (e.g. PCT_NAT_PFT, PCT_CLAY, PCT_SAND, or a full PFT-parameter vector) → Linear to `embed_dim`. Output order matches the group list.
- **Embeddings:** Variable position embedding (sin-cos) and **group embedding** (learned, one per group ID) added to each token.
- **Output:** Static token sequence `[B, N_static_tokens, embed_dim]`.

### Backbone and heads

- **Concatenation:** `all_tokens = [dyn_tokens; sta_tokens]` → `[B, N_dyn + N_sta, embed_dim]`.
- **Transformer:** Standard `nn.TransformerEncoder` (configurable `embed_dim`, `transformer_heads`, `transformer_layers`, `batch_first=True`, `norm_first=True`) over the token sequence.
- **Pooling:** Global mean over the token dimension → `[B, embed_dim]`.
- **Heads:** Separate MLPs from the pooled vector to scalar targets (ReLU), soil 2D (Softplus), PFT 1D (configurable activations per variable), and optional water. Task-weighted MSE (and optional physics or learnable weights) form the composite loss.

### Config (ModelConfig)

- **Tokenization / backbone:** `embed_dim`, `patch_size`, `transformer_layers`, `transformer_heads`, `dropout_p`.
- **Data-dependent:** `num_pfts`, `vector_length`, `matrix_rows`, `matrix_cols`; static structure is derived from `data_info` (static_columns, pft_param_columns, etc.) and the grouping logic in `_configure_static_structure()`.

---

## Repository layout (abridged)

```
LandSim/
├── config/              # training_config.py (ModelConfig: embed_dim, patch_size, transformer_*, etc.)
├── training/            # Training loop and trainer
├── models/              # cnp_combined_model.py (ForcingTemporalEncoder, StaticVariableEncoder, CNPCombinedModel)
│                        # combined_model.py (wrapper / FlexibleCombinedModel)
├── data/                # Data loaders and scaler management
├── scripts/             # CNP pipeline: validation, inference, NetCDF, comparison, restart ingestion/verification
│                        # training_data_generation/
├── train_model.py       # Quick-start demo training
├── train_cnp_model.py   # Full CNP training entry point
├── requirements.txt
├── README.md
└── docs/                # CNP runbook, prediction quality, workflow details
```

For step-by-step commands and run directory structure, see `docs/CNP_pipeline_runbook.md` and the main `README.md`.
