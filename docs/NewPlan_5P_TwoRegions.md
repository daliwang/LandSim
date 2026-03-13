## New plan: 5P, two-region focus

You are now in a very specialized regime:

- **Regions**: Amazon + (sub‑Saharan) Africa only.
- **Targets**: Only the 5 P variables (e.g. occlp, labilep, secondp, primp, plus the remaining P variable).
- **Base models**:
  - The best AI base model for P so far is `phase2_pvariable_focus` (built on `natveg_improved`).
  - Even this is still not good enough for Amazon + Africa.
  - For the rest of the globe, you will keep using the original `natveg_improved` run as-is.

Given this, it is not efficient to keep trying small variations of “same architecture + heavier P weights” on the full global model. A better strategy is:

1. **Post‑process / calibrate the existing best model for these regions** (fast, guaranteed improvement).
2. **Optionally add a dedicated small regional 5P model** (residual or stand‑alone) trained only on the two-region data.

Below are concrete options, roughly ordered by impact vs effort.

---

## 1. Fast win: bias/scale correction on top of phase2 predictions

Use `phase2` (global) or your best current run as a baseline, then statistically correct its systematic errors in Amazon + Africa. This does **not** require retraining the large CNP model.

- **For each P variable, layer, and optionally region**:
  - Take ground truth vs `phase2` predictions over all Amazon + Africa test cells.
  - Fit a simple linear correction per `(variable × layer)` or `(variable × region × layer)`:

  \[
  \text{GT} \approx a \cdot \text{pred} + b
  \]

- **At inference time (for these regions only)**, apply:

  \[
  \hat{Y}_{\text{corrected}} = a \cdot \hat{Y}_{\text{phase2}} + b
  \]

- **Implementation notes**:
  - You already have CSVs with predictions and GT, so estimating \(a, b\) is straightforward (e.g. per-variable linear regression).
  - Pros:
    - Very cheap and quick to implement.
    - Deterministic improvement if errors are mostly bias/scale (which seems to be the case for occlp, labilep, secondp, primp).
  - Usage:
    - When generating restarts for Amazon + Africa, apply this correction.
    - Elsewhere, continue to use raw `natveg_improved` (or your preferred global baseline).

---

## 2. Residual regional 5P model (recommended “real” model change)

Instead of re‑training the big CNP net, train a small residual model **only on Amazon + Africa data**.

- **Inputs**:
  - Original inputs (climate, soil, etc.).
  - `phase2` 5P predictions at each cell/layer.

- **Target**:

  \[
  \text{Residual} = \text{GT} - \text{phase2\_pred}
  \]

  for each of the 5 P variables (per layer).

- **Output**:
  - Residual for each \(5\text{P} \times \text{layer}\).
  - Final prediction:

    \[
    \hat{Y}_{\text{final}} = \hat{Y}_{\text{phase2}} + \hat{Y}_{\text{residual}}
    \]

- **Training**:
  - Train this small model only on two‑region training cells.
  - Loss focuses purely on the 5 P variables (no other outputs).

- **Why this helps**:
  - `phase2` already captures global structure and physical relationships.
  - The residual model only needs to learn **regional corrections** (bias, shape) for high‑P tropics.
  - Capacity is dedicated entirely to these 5 variables in these two regions.

- **Implementation options**:
  - A standalone PyTorch script that reads:
    - Per‑cell 5P GT / prediction CSVs, plus features.
  - Or a variant of your current architecture with:
    - Just the soil‑2D head and 5 outputs.
    - `region_boxes` to select Amazon + Africa.
    - Precomputed `phase2` predictions fed in as additional inputs.

---

## 3. True 5P‑only regional model from scratch

Your previous two‑region model still used the full 25 soil‑2D outputs, just with different weights. For this very targeted use‑case, a more radical but cleaner option is a dedicated **5P‑only** model:

- **Architecture / IO**:
  - Change the IO list so only the 5 P variables are outputs (matrix head size \(= 5 \times \text{layers}\)).

- **Data**:
  - Train only on two‑region data using `region_boxes`.

- **Training setup**:
  - Use individual normalization and strong 5P loss weights.
  - Allow more epochs and/or slightly higher LR since the task is smaller.

- **Advantages**:
  - Smaller head, more effective capacity per P variable.
  - No competition from non‑P soil variables in the loss.

- **Trade‑offs**:
  - Larger pipeline change:
    - IO list changes.
    - Model sizing needs to be updated.
    - Inference / restart‑generation scripts must be updated to use the 5P‑only head.
  - This is the “dedicated specialist” model if the residual approach still isn’t sufficient.

---

## 4. Loss and normalization tweaks for the two regions

Given the very different distributions in Amazon + Africa vs the rest of the world (you already see GT ratios \(\sim 2.5\text{–}4.8\)), you can tune the loss and scaling specifically for regional / residual models:

- **Loss choices**:
  - Prefer \(L_1\) / MAE or a less tail‑heavy variant than `log1p_huber`, so you do not over‑focus extreme P outliers.
  - Optionally a **quantile loss** (e.g. 0.5 or 0.7) if you want asymmetric penalties for under/over‑prediction.

- **Normalization**:
  - Keep individual normalization per variable.
  - Optionally re‑scale targets so typical two‑region ranges map to \(\mathcal{O}(1)\).
  - Your current individual scalers likely already help; only tweak if diagnostics show saturation or strong skew.

These tweaks are most relevant for options **2** and **3** (residual or 5P‑only regional models).

---

## 5. Practical recommendation / priority order

Given limited time and the ROI curve:

1. **Immediately**:
   - Implement **bias/scale correction** (option 1) on top of `phase2` for Amazon + Africa.
   - This should give noticeably better 5 P performance at those sites with minimal new code.
2. **Next, if you want a true ML improvement**:
   - Implement a small **residual two‑region 5P model** (option 2) and evaluate it at Amazon and several African sites.
3. **Only if the above is still insufficient**:
   - Move to a full **5P‑only regional model** from scratch (option 3), plus the loss/normalization tweaks in option 4.

