# Post-hoc probability-position audit — WDBC v1.1

**Status:** Exploratory follow-up to the frozen v1.0 repeated-CV experiment. **Date:** 2026-11-23. **Original protocol unchanged.** No model refitting, hyperparameter changes, or reinterpretation as a clinical risk distribution.

## Rationale and mathematical limits

For Logistic Regression, `p=sigmoid(z)`, so `dp/dz=p(1-p)`. For small shifts on the logit scale near a representative value, the probability width is approximately `p(1-p)` multiplied by the corresponding logit-scale width. Thus large probability widths near `p=0.5` can arise from the probability scale even in the absence of special case-level mechanisms. This is a *local approximation*, not an identity for arbitrary P10/P90 ranges. It does not establish the mechanism for Random Forest, where no logistic link is used.

**Proxy:** `g_i=median(p_ir) × [1−median(p_ir)]`. This uses the *same* repeated predictions that also define `W_i`; association between `g` and `W` must not be interpreted causally or as an externally validated explanation.

## Recomputed empirical results

All values below were independently recalculated from the public LR and RF `artifacts/results/*_individual_stability.csv` files (569 matched observations). Spearman uses average ranks for ties.

| Diagnostic | Result |
|---|---:|
| Spearman(W_LR, W_RF) | 0.727522 |
| Spearman(W_LR, g_LR) | 0.993923 |
| Spearman(W_RF, g_RF) | 0.990166 |
| Pearson(W_LR, g_LR) | 0.801393 |
| Pearson(W_RF, g_RF) | 0.919340 |
| Pearson(residual_rank_LR, residual_rank_RF) | −0.022667 |

For each model separately, `residual_rank_model` is the ordinary least-squares residual from `rank(W_model) ~ intercept + rank(g_model)`. The final value is a **Pearson correlation between those residual vectors**, not a partial Spearman correlation with a common single covariate. It is a quick descriptive adjustment, sensitive to functional form and shared measurement structure. It is **not** a hypothesis test and does not establish conditional independence.

### Subset sensitivity: both models' median probabilities inside the stated interval

| Both medians in | N | Spearman(W_LR, W_RF) |
|---|---:|---:|
| [0.01, 0.99] | 149 | 0.548921 |
| [0.05, 0.95] | 72 | 0.192649 |
| [0.10, 0.90] | 54 | 0.208919 |
| [0.20, 0.80] | 31 | 0.075000 |

These are **post-hoc selected subsets**. Restricting ranges reduces sample size and may induce selection effects. Do not compare these coefficients as if they came from independent populations or interpret them as evidence of zero association in a target population.

### Candidates with more width than a local position-neighborhood baseline

Sort the 569 observations by `g` for each model (ties broken by ID), then subtract the **centered rolling median of W across 51 adjacent observations**. A positive difference means larger empirical `W` than this descriptive local median. Repeat with windows of 31 and 71 to diagnose arbitrary bandwidth choices. This is not a calibrated residual, confidence interval, statistical outlier test or biological property.

- LR: `WDBC_038`, `WDBC_068`, `WDBC_190` and `WDBC_146` are the leading positive-excess cases under all three windows.
- RF: `WDBC_191`, `WDBC_038` and `WDBC_255` are the leading positive-excess cases under all three windows.
- `WDBC_038` appears near the top in both methods. This is an observation worthy of further scrutiny, not a claim that its biological characteristics cause redevelopment sensitivity.

The reproducible complete top-10 results for each model, along with window sensitivity, are produced by `scripts/analyze_probability_position_v11.py`. Sorting and rolling-window rules are exactly specified in the script. **No patient-level explanation can be inferred** from these rankings.

## Inference and interpretation

1. Aggregate AUC/Brier distributions remain narrow under the *specified* repeated-CV redevelopment experiment; some individual `W` values are much larger than the median.
2. The rankings of widths almost coincide with the rankings of `g` within each model. This is a major alternative explanation for the raw cross-model width concordance.
3. The raw 0.728 cross-model Spearman should not be described as evidence of model-independent, intrinsic "unstable observations." The examined rank-adjustment is near zero, although this is method-dependent.
4. The residual/local-excess investigations are exploratory **hypothesis-generating** diagnostics. Without independent datasets or a design able to isolate geometry, local support and training-composition effects, the underlying causes remain unidentified.
5. The rank correlations of approximately 0.99 do **not** imply a coefficient of determination of 0.99 or that 99% of width variance has a causal explanation.

## Reproduction and scope constraints

```bash
python scripts/analyze_probability_position_v11.py
python -m unittest discover -s tests -v
```

The script consumes only two already-published observation-summary CSVs and creates JSON, CSV, PNG and PDF outputs in `analysis/generated_v11/`. It does not rerun the 8,000-tree RF experiment or modify original artifacts. SciPy, pandas, NumPy and Matplotlib are already declared (directly or via SciPy) in the environment; see the existing `environment.yml`.

**Not undertaken in v1.1:** bootstrap, new datasets, external validation, feature attribution, probability calibration, inferential p-values, preregistration claims, the republishing of any absent audit ZIP, or alteration of the original notebook and legacy figures.

## Sources

1. Riley, R. D., & Collins, G. S. (2023). *Stability of clinical prediction models developed using statistical or machine learning methods*. Biometrical Journal 65:2200302. https://doi.org/10.1002/bimj.202200302.
2. Marx, C., Calmon, F., & Ustun, B. (2020). *Predictive Multiplicity in Classification*. ICML/PMLR 119:6765–6774. https://proceedings.mlr.press/v119/marx20a.html. This addresses multiple competing models rather than the same repeated-development protocol.
3. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.
