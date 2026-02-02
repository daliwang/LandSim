### Zero-aware loss and CLI controls for zero-inflated targets

This change adds a zero-aware loss you can enable via CLI to explicitly penalize wrong predictions involving zeros—especially false negatives—while keeping the default MSE path unchanged.

### What changed
- New module: `training/losses.py` with `ZeroAwareLoss`.
- Trainer loads loss from config:
  - Default: `MSE`
  - Optional: `ZeroAwareLoss` with tunable penalties for zero-related mistakes.
- New zero-related metrics are emitted during evaluation (saved alongside existing metrics).

### Why
Many targets are zero. We want to:
- Strongly penalize false negatives: predict ~0 when truth > 0
- Penalize false positives: predict > 0 when truth = 0

### How ZeroAwareLoss works
Overall loss:
- L = BaseLoss(ŷ, y) + α · FP(ŷ, y) + β · FN(ŷ, y)
- Base loss: MSE, MAE, or SmoothL1 (configurable).
- FP term: penalizes predictions above zero when truth is zero.
- FN term: penalizes predictions below a relative margin τ · |y| when truth is positive. Set β » α to emphasize under-prediction.

Non-negativity: The model already uses non-negative heads for most outputs (ReLU/Softplus), which complements the zero-aware terms.

### CLI flags
Add these to `train_cnp_model.py` runs:
- `--loss {mse, zero_aware}`: switch between standard MSE and zero-aware.
- `--base-loss {mse, mae, smoothl1}`: base part inside zero-aware.
- `--zero-fp-weight FLOAT` (α): weight for false positives (y=0, ŷ>0). Default 1.0
- `--zero-fn-weight FLOAT` (β): weight for false negatives (y>0, ŷ too small). Default 8.0
- `--zero-margin FLOAT` (τ): relative margin for FN penalty, default 0.1
- `--fp-power FLOAT` (p): exponent for FP term, default 2.0
- `--fn-power FLOAT` (q): exponent for FN term, default 1.0
- `--zero-eps FLOAT`: numerical epsilon for defining zero in loss space, default 1e-8
- `--zero-eval-epsilon FLOAT`: threshold for FP@0/FN@0 metrics, default 1e-6

These are also stored in the training config so they can be programmatically controlled.

### Example commands
Default MSE:

```bash
python train_cnp_model.py --loss mse
```

Zero-aware with stronger FN penalty:

```bash
python train_cnp_model.py \
  --loss zero_aware \
  --base-loss smoothl1 \
  --zero-fp-weight 10 \
  --zero-fn-weight 50.0 \
  --zero-margin 0.1 \
  --fp-power 2.0 \
  --fn-power 1.0
```

Tip: Increase `--zero-fn-weight` if you still see too many false negatives at zero; if you see many false positives on zeros, increase `--zero-fp-weight`.  (changed from 1.5, 10.0)

### New metrics
Reported in the metrics file, per head:
- Scalars: `scalar_fp_at0`, `scalar_fn_at0`
- PFT1D: `pft1d_fp_at0`, `pft1d_fn_at0`
- Soil2D: `soil2d_fp_at0`, `soil2d_fn_at0`

They are fractions of total elements (0.0–1.0). Zero threshold used is `--zero-eval-epsilon`.

### Sensible defaults
- Start with: `--loss zero_aware --base-loss smoothl1 --zero-fn-weight 10 --zero-fp-weight 1 --zero-margin 0.1`
- If under-prediction persists (high FN@0), raise `--zero-fn-weight`.
- If too many false positives on zeros, raise `--zero-fp-weight`.

### Backward compatibility
- If you do nothing, training uses standard MSE (exactly as before).
- The zero-aware pathway is opt-in via `--loss zero_aware`.

### Files touched
- `training/losses.py`: adds `ZeroAwareLoss`
- `training/trainer.py`: configurable criterion + FP/FN@0 metrics
- `config/training_config.py`: new loss-related fields
- `train_cnp_model.py`: new CLI flags mapped into the training config

### FAQ
- Q: Does this change model architecture?
  - A: No. This is strictly a loss-and-metrics change.
- Q: Can I revert to the old behavior?
  - A: Yes, use `--loss mse` (the default).

example usage: 
 python train_cnp_model.py --loss zero_aware --base-loss smoothl1 --zero-fn-weight 10 --zero-fp-weight 1.5 --zero-margin 0.1 --variable-list CNP_IO_updated9_dev_dw.txt --use-trendy1 --epochs 100 > std.train.log 2>&1 &