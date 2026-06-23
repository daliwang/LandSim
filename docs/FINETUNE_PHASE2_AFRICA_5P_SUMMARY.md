# Summary and reason for fine-tuning Phase2 on Africa only (5P-focused)

## Reason for fine-tuning

- **Africa 5P predictions are far too high** from both the global and the tropical (Phase2) models. Examples:
  - **occlp_vr:** ground truth max ~150 at Africa sites; predictions ~400.
  - **solutionp_vr:** ground truth max ~0.00125; predictions after bias correction ~0.03 (order-of-magnitude too high).
- **Bias/scale correction alone cannot fix this:** the underlying model bias in Africa is too large; a linear correction either fails to bring values into a plausible range or overfits and degrades elsewhere.
- **We do not want to change other parts or variables:** Amazon and the rest of the tropics are working well with Phase2 + bias/scale. The goal is to improve **only** Africa 5P without altering Amazon or non-P variables.

Therefore we **fine-tune the Phase2 model on Africa-only data**, with loss focused on the five P variables, so that the model learns Africa-specific 5P structure. The resulting checkpoint can then be used for Africa-region inference and its 5P predictions spliced into the restart in the Africa box only.

## Summary

| Item | Description |
|------|-------------|
| **Base model** | Phase2 P-focused: `cnp_results/run_20260305_153217_phase2_pvariable_focus/cnp_predictions/model.pth` |
| **Fine-tune data** | Africa region only: lat ∈ [-15, 15], lon ∈ [0, 30] (single region box). |
| **Loss focus** | Five P variables: labilep_vr, occlp_vr, solutionp_vr, secondp_vr, primp_vr (high weights; other variables down-weighted). |
| **Config** | `config/finetune_phase2_africa_5p_only.json` |
| **Output** | Timestamped run under `cnp_results/` with suffix `phase2_africa_5p`; checkpoint e.g. `finetuned_phase2_africa_5p.pth`. |
| **Use after training** | Run inference with the Africa fine-tuned model in the Africa box only; merge those 5P values into the restart (Phase2 + bias/scale elsewhere). |

## Two-region model not used

The two-region 5P model (`run_20260308_211537_two_region_five_p`) was evaluated and performs **worse** than Phase2 at the Amazon site. We do not use it for production. See `docs/AMAZON_5P_COMPARISON_TWO_REGION_VS_PHASE2_REPORT.md`.

## How to run the fine-tuning

From the repo root:

```bash
cd /mnt/proj-shared/AI4BGC_7xw/AI4BGC

python scripts/run_finetuning_json.py \
  --config-json config/finetune_phase2_africa_5p_only.json \
  --output-dir-suffix phase2_africa_5p
```
