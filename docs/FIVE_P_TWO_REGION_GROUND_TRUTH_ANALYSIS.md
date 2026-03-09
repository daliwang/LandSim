# 5 P Ground Truth: Two Regions vs Rest of World

**Goal:** Get the 5 P predictions (labilep_vr, occlp_vr, solutionp_vr, secondp_vr, primp_vr) in the two target regions (Amazon + Central Africa) as close as possible to ground truth. One strategy is to train only on data from these two regions, **if** the 5 P variables there have different distributions than in other regions.

---

## Is it true that 5 P in these two regions are different?

**Yes.** Ground-truth analysis on the training data (Trendy_1_data_CNP) shows that the 5 P variables in **Amazon + Central Africa** have clearly different (and much higher) distributions than in the **rest of the world**.

### Results (sum over first 10 soil layers, same units as training)

| Variable        | Two regions (Amazon + Africa) | Other (rest of world) | Ratio (two_region / other) |
|----------------|--------------------------------|------------------------|-----------------------------|
| Y_labilep_vr   | mean 636,  p50 605             | mean 232,  p50 0       | **2.74**                    |
| Y_occlp_vr     | mean 3305, p50 3511            | mean 1351, p50 0       | **2.45**                    |
| Y_solutionp_vr | mean 1.12, p50 0.02            | mean 0.24, p50 0       | **4.65**                    |
| Y_secondp_vr   | mean 12847, p50 10335          | mean 4849, p50 0       | **2.65**                    |
| Y_primp_vr     | mean 82.7, p50 44.6            | mean 17.1, p50 0       | **4.84**                    |

- **Two regions:** 1,758 grid cells (Amazon box: lat −30–10, lon 270–330; Africa box: lat −15–15, lon 0–30).
- **Other:** 9,242 grid cells from the same batches (lat outside the two boxes or lon outside).
- In “other” regions, many cells have **zero or very low** P (p50 = 0 for all five variables). In the two regions, medians are **non-zero and large** (e.g. labilep_vr p50 ≈ 605, occlp_vr p50 ≈ 3511).

So the 5 P variables in the two target regions are **not** representative of the global distribution: they are systematically higher and have a different shape (fewer zeros, higher medians and means). Training a global model on all data would be dominated by the many low-P cells elsewhere and can underfit or mis-scale the high-P tropical behavior in Amazon and Africa.

---

## Recommendation

1. **Training only on the two regions is well justified.** The ground truth shows that 5 P in Amazon + Africa are different from the rest of the world (ratios of means ~2.4–4.8). Using only data from these regions lets the model focus on the relevant distribution and should help get 5 P predictions there closer to ground truth.
2. **You are already doing this** with **`config/training_config_two_region_five_p.json`**, which sets `data_filtering_config.region_boxes` to the Amazon and Africa boxes so the dataloader keeps only cells in those two regions. The run **run_20260308_211537_two_region_five_p** used this config and is a large improvement over the non–region-focused run.
3. To **improve further** in the two regions, you can:
   - Keep two-region-only data (region_boxes) and optionally **increase 5 P weights** or **epochs**.
   - **Finetune** from the current two-region run (or from phase2_pvariable_focus) with the same region_boxes and strong 5 P weights.
   - Re-run the ground-truth analysis on **all** training files (or on the exact files used in production) to confirm the same conclusion; the script supports `--max-files` and `--data-dir`.

---

## How to reproduce the analysis

From the repo root:

```bash
python scripts/analyze_5p_ground_truth_by_region.py \
  --data-dir /path/to/Trendy_1_data_CNP \
  --max-files 11 \
  --output docs/analysis_5p_by_region.txt \
  --output-csv docs/analysis_5p_by_region.csv
```

- Use `--max-files` so that the loaded batches include both two-region and other cells (e.g. batches 7–11 for Trendy_1_data_CNP). For a full run, use a large `--max-files` or load all batches.
- Results are written to `--output` (text report) and optionally `--output-csv` (per-variable stats).

See **`docs/analysis_5p_by_region.txt`** for the exact numbers and conclusion from the last run.
