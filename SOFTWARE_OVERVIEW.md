# LandSim: Software Description and Overview

## Two-paragraph summary (for short academic papers, ~200 words)

LandSim (v0.1, formerly AI4BGC) is an open-source deep learning framework for modeling terrestrial ecosystem dynamics with a focus on the Carbon–Nitrogen–Phosphorus (CNP) cycle in the E3SM Land Model (ELM). The software provides a modular, config-driven pipeline to train physics-aware models on heterogeneous ELM data—time series (e.g. atmospheric forcings), static surface and plant functional type (PFT) attributes, and 1D/2D state variables—and to run inference at scale. A central capability is ingesting AI-predicted states into ELM restart files, enabling stable, accelerated spin-up and scenario experiments without replacing the physical model. LandSim supports 1° and 0.5° resolutions and is designed for reuse in large-domain and km-scale settings (e.g. kmELM, kiloCraft), with configurable variable lists (CNP_IO) and optional mixed-precision training.

The workflow comprises configuration, training on enhanced CNP datasets with per-variable or group normalization, validation and prediction-quality reporting, full-dataset inference, NetCDF export, comparison with ELM outputs, restart-file ingestion of PFT 1D and soil 2D variables, and restart verification. The core model is a multi-modal encoder–fusion–decoder: an LSTM for time series, fully connected encoders for static/surface and PFT parameters, and CNNs for depth-resolved soil 2D inputs; a Transformer encoder fuses these representations; and separate multi-task heads predict scalars (e.g. GPP, NPP), PFT 1D (e.g. LAI), and soil 2D targets (e.g. carbon, nitrogen, phosphorus by layer). Training uses a composite loss (task-weighted MSE and optional physics or learnable weights), MinMax normalization, and optional automatic mixed precision.

---

## Short description

**LandSim** (v0.1, formerly AI4BGC) is an open-source deep learning framework for modeling terrestrial ecosystem dynamics with a focus on the **Carbon–Nitrogen–Phosphorus (CNP) cycle** in the E3SM Land Model (ELM). It provides a modular, config-driven pipeline to train physics-aware AI models on heterogeneous ELM data (time series, static surface and PFT attributes, and 1D/2D state variables), run inference at scale, export predictions to NetCDF, and **ingest AI-predicted states into ELM restart files** for stable, accelerated spin-up and scenario runs. LandSim supports 1° and 0.5° (e.g. TRENDY) resolutions, optional mixed-precision training, and is designed to be reusable for large-domain and km-scale (e.g. kmELM / kiloCraft) experiments.

---

## Workflow overview

End-to-end flow (CNP pipeline):

```
1. Configuration   → 2. Training  → 3. Validation  → 4. Inference  → 5. NetCDF export  → 6. Comparison  → 7. Restart ingestion  → 8. Restart verification
```

| Stage | Description |
|-------|-------------|
| **1. Configuration** | Define variable lists in a `CNP_IO_*.txt` file (inputs and targets: scalars, PFT 1D, soil 2D, time series, static, PFT params). Optional `--model-config` overrides architecture (e.g. compact vs 27M-parameter). |
| **2. Training** | `train_cnp_model.py` loads enhanced CNP pickle datasets, normalizes with per-variable or group scalers, trains the CNP combined model (LSTM + encoders + Transformer fusion + task heads), saves model, scalers, and training metrics under `cnp_results/run_YYYYMMDD_HHMMSS/`. |
| **3. Validation** | Scripts such as `cnp_result_validationplot.py` and `generate_prediction_quality_report.py` evaluate test-set predictions (RMSE, R², quality categories) and optional PFT/soil checks. |
| **4. Inference** | `run_inference_all.py` runs the trained model on the full dataset, denormalizes with saved scalers; optional `fix_pft1d_nans.py` for post-processing. |
| **5. NetCDF export** | `ai_predictions_to_netcdf.py` aggregates predictions to a single NetCDF with coordinates aligned to the ELM grid. |
| **6. Comparison** | `ai_model_comparison_plot.py` produces tri-panel plots (AI vs ELM vs difference) for selected variables and layers. |
| **7. Restart ingestion** | `ai_predictions_to_restart.py` writes AI predictions into ELM restart NetCDF (PFT 1D and soil 2D variables), preserving file structure and attributes. |
| **8. Restart verification** | `ai_restart_comparison.py` compares the AI-updated restart with the original (e.g. four-panel plots and metrics). |

**Training data generation** (separate sub-pipeline under `scripts/training_data_generation/`):

- **Forcing:** `construct_forcing_20years.py` builds forcing NetCDF files (e.g. FLDS, FSDS, PSRF, QBOT, PRECTmms, TBOT) from raw monthly data.
- **Datasets:** `enhanced_training_dataset.py` can produce (1) forcing-only, (2) full enhanced ecosystem (history + restart + surface aligned), or (3) initial-condition-only datasets in batch pickle format for training.

---

## Architecture overview

LandSim’s main **CNP combined model** (`models/cnp_combined_model.py`) is a multi-modal encoder–fusion–decoder stack:

- **Time series:** LSTM over 6 forcing variables (e.g. 20 years of monthly data).
- **Static / surface:** Fully connected (FC) encoders for surface properties, PFT coverage, soil texture, etc.
- **PFT parameters:** FC or optionally 2D CNN over 44 PFT parameters × number of PFTs.
- **Scalars / PFT 1D / water:** FC encoders for scalar inputs and 1D PFT-state inputs.
- **Soil 2D:** CNN over depth-resolved 2D variables (e.g. soil carbon, nitrogen, phosphorus by layer).
- **Fusion:** Transformer encoder over concatenated encoder outputs (with optional positional/modality identity) to capture cross-variable interactions.
- **Output heads:** Separate multi-task perceptrons for scalar targets (e.g. GPP, NPP, AR, HR), PFT 1D targets (e.g. LAI), and soil 2D targets (e.g. cwdc_vr, cwdn_vr, cwdp_vr by layer).

Training uses a composite loss (task-weighted MSE and optional physics or learnable loss weights), with MinMax normalization and optional mixed precision (AMP). Configurable options include token dimension, dropout, and encoder sizes via `ModelConfig` (e.g. `CNP_model_config_v01.txt`, `CNP_model_config_27M.txt`). Data loading is handled by modules in `data/` (e.g. `data_loader_smart.py`, `individual_scaler_manager.py`) with support for variable lists from CNP_IO and per-group or per-variable scalers for normalization and denormalization.

---

## Repository layout (abridged)

```
LandSim/
├── config/              # Training and model configs (e.g. variable weights, ModelConfig)
├── training/            # Training loop and trainer
├── models/              # combined_model.py, cnp_combined_model.py
├── data/                # Data loaders and scaler management
├── scripts/             # CNP pipeline: validation, inference, NetCDF export, comparison,
│                        # restart ingestion/verification; training_data_generation/
├── train_model.py       # Quick-start demo training
├── train_cnp_model.py   # Full CNP training entry point
├── requirements.txt
├── README.md
└── docs/                # CNP runbook, prediction quality, workflow details
```

For step-by-step commands and run directory structure, see `docs/CNP_pipeline_runbook.md` and the main `README.md`.

---

## Comparison with model_tokenization_dev branch

For the **tokenization-based** architecture (ForcingTemporalEncoder, StaticVariableEncoder, identity-aware tokens, Transformer backbone, global mean pooling), see **`SOFTWARE_OVERVIEW_model_tokenization_dev.md`**.
