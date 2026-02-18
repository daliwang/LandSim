# Suggestions to Further Improve CNP Model Performance

Based on comparing **run_20260212_154931_updated9_training_config** and **run_20260212_162802_experiment_2**, the following levers can help.

---

## 1. Config / loss (quick to try)

### 1.1 Still failing: `primp_vr` (100% bad)
- **primp_vr** does not use litter weights (those apply only to `litr*_vr`). It only uses `soil2d_weights` and `tail_aware_weights`.
- **Suggestions:**
  - Set **primp_vr** weight to **10–12** in both `soil2d_weights` and `tail_aware_weights` (experiment_2 used 9).
  - Ensure **primp_vr** is in `tail_aware_weights` so it gets the tail-aware loss (log1p), which helps heavy-tailed targets.

### 1.2 Litter layer: `litr2c_vr`, `litr2n_vr`, `litr2p_vr` (80% bad)
- The trainer applies **litter_c_loss_weight**, **litter_n_loss_weight**, **litter_p_loss_weight** to all litr*_vr variables (in addition to per-variable weights).
- **Suggestion:** Increase global litter weights so the model pays more attention to all litter variables:
  ```bash
  --litter-c-loss-weight 2.0 --litter-n-loss-weight 2.0 --litter-p-loss-weight 2.0
  ```
  Or add these to a unified config if supported. This multiplies with your existing litr2* variable weights.

### 1.3 Tail-aware loss type
- Current: `log1p_mse`.
- **Suggestion:** Try **`log1p_huber`** for tail-aware variables (and set `tail_aware_huber_delta`, e.g. 0.5–1.0). Huber is less sensitive to large residuals and can stabilize training on difficult PFTs/layers. Optionally try **`log1p_quantile`** (e.g. tau=0.9) to focus on the upper tail.

### 1.4 Pools: `ppool` (75%), `npool` (62.5%)
- Already heavily weighted. In addition:
  - Keep them in **tail_aware_weights** with high weight (e.g. 8–10).
  - Consider **pft1d_activation_overrides** (e.g. `"abs"` or similar) only if the variable is non-negative and the model can output negative; otherwise leave as is.

---

## 2. Training schedule

- **More epochs:** Try **150** epochs (e.g. `--epoch 150`). Loss curves often keep improving past 100.
- **Learning rate:** If loss is still decreasing at the end, try a slightly lower LR (e.g. 5e-5) for the last 20–30 epochs (would require a small code change or a two-run approach: train 100 epochs, then resume with lower LR).
- **Two-phase training (advanced):** Phase 1: train as now (e.g. 80 epochs). Phase 2: freeze most of the model and train only the head(s) or only the worst variables (e.g. primp_vr, litr2*_vr, ppool, npool) with a small learning rate for 20–30 epochs. Would require a script that loads the phase-1 checkpoint and applies different loss weights / frozen params.

---

## 3. Data and normalization

- **primp_vr:** Check distribution (histogram, min/max, zeros). If it is very skewed or has many zeros, consider:
  - A variable-specific scaling (e.g. log1p or sqrt) in the dataloader or loss, or
  - Ensuring the scaler for this variable is fit robustly (e.g. robust scale or clip extremes before scaling).
- **litr2*_vr:** Similarly, check whether layer 1–8 have very different scales or sparsity; per-layer or per-variable scaling might help.
- **PFT imbalance:** Bad predictions are often on specific PFTs (e.g. 2, 3, 7, 8, 9, 11, 12). If some PFTs are rare in the training set, consider oversampling those PFTs or weighting samples by inverse PFT frequency.

---

## 4. Architecture (longer-term)

- **Soil2D / mineral P:** If primp_vr stays 100% bad after config changes, consider a small dedicated branch or head for “mineral P” (primp_vr) with a few extra layers, so the model can learn a different representation for that variable.
- **Shared vs separate:** Ensure soil2d variables share enough capacity; if the same encoder is used for all soil layers, increasing width/depth slightly might help the worst variables without hurting the rest.

---

## 5. What to try next (concrete)

1. **Experiment 3 config:** Use **`config/training_config_experiment_3.json`** (primp_vr 11, tail_aware `log1p_huber`, `huber_delta` 1.0). Optionally add litter loss weights via CLI:
   ```bash
   python train_cnp_model.py ... --unified-config config/training_config_experiment_3.json \
     --litter-c-loss-weight 2.0 --litter-n-loss-weight 2.0 --litter-p-loss-weight 2.0
   ```
2. **Epochs:** Run with **--epoch 150**.
3. **Inspect data:** For **primp_vr** and **litr2*_vr**, plot histograms and per-layer stats; adjust scaling or clipping if needed.
4. **Reproducibility:** Fix seed and run experiment_2 config twice; if good % varies a lot, consider more epochs or a small learning-rate decay at the end.

These steps are ordered from “quick config change” to “data/architecture”; doing 1–2 first is the most efficient.
