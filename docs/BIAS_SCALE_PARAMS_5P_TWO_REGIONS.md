## Bias/scale correction parameters for 5P, two-region focus

This document records the linear bias/scale corrections learned by `scripts/apply_5p_bias_scale_correction.py` for the `natveg_improved` run, restricted to the **Amazon** and **Africa** region boxes defined in the two-region configuration.

For each soil-2D P variable, region, and layer, the script fits:

\[
\text{GT} \approx a \cdot \text{pred} + b
\]

and then applies, at inference time:

\[
\hat{Y}_\text{corrected} = a_{\text{var, layer, region}} \cdot \hat{Y}_\text{pred} + b_{\text{var, layer, region}}
\]

The full set of coefficients is saved in:

- `docs/bias_scale_params_5P_two_regions_natveg_improved.json`

That JSON is the canonical source for any downstream scripts that need to reuse the same correction outside the original calibration step.

### High-level ranges

Across all 5 P variables, 10 layers, and both regions:

- **Slopes \(a\)**: roughly from **-0.53** to **1.25**, with most values between **0.9** and **1.1**.
- **Offsets \(b\)**:
  - Typically in the range **\(-20\) to \(+40\)** for `labilep_vr`, `occlp_vr`, `solutionp_vr`, and mid/deep layers of `secondp_vr` and `primp_vr`.
  - Larger positive offsets (up to \(\sim 192\)) appear for top-layer `secondp_vr` in Africa.
  - `solutionp_vr` has very small \(b\) (on the order of \(10^{-3}\) after the scaling trick).

Per-variable patterns (Amazon + Africa, all layers):

- **`labilep_vr`**:
  - \(a \sim 0.93\text{–}1.09\)
  - \(b \sim -0.95\text{–}10.0\)

- **`occlp_vr`**:
  - \(a \sim 0.97\text{–}1.00\)
  - \(b \sim 14\text{–}34\)
  - Interpretation: primarily an additive upward shift in occluded P.

- **`solutionp_vr`**:
  - \(a \sim 1.06\text{–}1.25\)
  - \(b \sim -0.014\text{–}0.001\)
  - Effectively a multiplicative rescale, with negligible intercept.

- **`secondp_vr`**:
  - \(a \sim 0.92\text{–}1.09\)
  - \(b\) from about \(-18\) to \(\sim 192\), with the largest \(b\) in African top layers.

- **`primp_vr`**:
  - Top 1–2 layers: \(a \sim 0.87\text{–}1.11\), \(b \sim 4.7\text{–}41\).
  - Deeper layers: some **negative slopes** with small positive offsets, implying strong shape corrections that are then clamped to non-negative in the application script.

### How to reuse these parameters

- For any grid cell inside the **Amazon** or **Africa** region boxes, even without local ground truth, apply the corresponding \((a, b)\) from the JSON for that variable, layer, and region.
- For cells outside those boxes, applying these parameters is an extrapolation; it may still be useful in similar tropical conditions but should be treated as an assumption.

