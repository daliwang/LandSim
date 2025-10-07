# Dynamic Model Configuration via --model-config

## Summary
- Introduces dynamic model architecture overrides using a simple text config file supplied via `--model-config`.
- Enables rapid experimentation without code edits: adjust LSTM/FC sizes, CNN channels, transformer depth/heads/dim, dropout, and output geometry.
- Works for both training and inference.

## Motivation
- Previously, architecture changes required code edits and error‑prone synchronization between training and inference.
- This feature standardizes overrides and improves reproducibility by checking in the exact config files used in runs.

## What’s Included
- Training: `train_cnp_model.py` accepts `--model-config CNP_model_config_*.txt` and applies recognized keys onto `ModelConfig`.
- Inference: `scripts/run_inference_all.py` also takes `--model-config` (optional) while still loading the exact variable lists and scalers from the training run (`cnp_config.json`).
- Parser: tolerant to unknown keys; supports `key = value`, booleans, comma lists, and Python lists.

## Example Configs (checked in)
- `CNP_model_config_v01.txt` – compact architecture for quick runs
- `CNP_model_config_27M.txt` – larger architecture approximating ~27M parameters

## Usage

### Train with compact config
```bash
python train_cnp_model.py \
  --variable-list CNP_IO_LiterP.txt \
  --model-config CNP_model_config_v01.txt \
  --epochs 5 --batch-size 128 --learning-rate 1e-4
```

### Train with 27M-like config
```bash
python train_cnp_model.py \
  --variable-list CNP_IO_list1.txt \
  --model-config CNP_model_config_27M.txt \
  --epochs 300 --batch-size 128 --learning-rate 1e-4
```

### Inference (optional override)
```bash
python scripts/run_inference_all.py \
  --model cnp_results/run_YYYYmmdd_HHMMSS/cnp_predictions/model.pth \
  --data-paths /path/to/Trendy_1_data_CNP \
  --file-pattern 'enhanced_*1_training_data_batch_*.pkl' \
  --output-dir cnp_inference_entire_dataset \
  --use-training-config \
  --model-config CNP_model_config_v01.txt
```

Notes:
- Inference prioritizes the exact training variable lists and scalers from `cnp_config.json` to avoid mismatches. The `--model-config` only influences architecture sizing; variable selection still comes from training config when `--use-training-config` is set.

## Supported Keys (subset)
- Encoders: `lstm_hidden_size`, `static_fc_size`, `pft_1d_fc_size`, `scalar_fc_size`, `water_fc_size`
- Soil 2D CNN: `conv_channels`, `conv_kernel_size`, `conv_padding`
- PFT parameters: `use_cnn_for_pft_param`, `pft_param_cnn_channels`, `pft_param_cnn_kernel_size`, `pft_param_cnn_padding`, `num_pfts`, `pft_param_size`
- Transformer: `num_tokens`, `token_dim`, `transformer_layers`, `transformer_heads`
- Global: `dropout_p`
- Outputs: `vector_length`, `matrix_rows`, `matrix_cols`

Unknown keys are ignored safely.

## Compatibility & Guidance
- Fully backward-compatible; `--model-config` is optional.
- When PFT parameter inputs are 3D tensors, prefer `use_cnn_for_pft_param = true` to ensure feature dimensions align across encoders.

## Results & Impact
- Dynamic architecture toggling without code edits; reproducible runs via committed config files.
- No runtime overhead beyond one-time parsing.

## Files of Interest
- `config/training_config.py`: applies overrides from text file in `get_cnp_combined_config()`
- `train_cnp_model.py`: CLI `--model-config` wiring
- `scripts/run_inference_all.py`: optional `--model-config` support during inference
- `README.md`: user-facing docs and examples


