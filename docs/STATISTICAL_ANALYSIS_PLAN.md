# Statistical Analysis Plan

This document is the default confirmatory SAP. A dated copy, target manifest hash, query-bank hash, analysis commit, and randomization salt must be frozen before test traces are unblinded.

## 1. Experimental and observational units

- **Independent unit:** training lineage.
- **Nested units:** checkpoint, intervention, prompt, output token, sampled completion, and parameter coordinate.
- **Repeated inverse-training runs:** algorithmic replicates, not new target samples.
- **Public model family:** the independent unit for leave-one-family-out external validity.

The primary dataset is reduced to one prespecified summary per lineage and condition or analyzed with lineage random effects. Treating prompts, parameters, or checkpoints as independent observations is prohibited.

## 2. Co-primary hypotheses and familywise error

The confirmatory family contains four tests:

1. full logits versus sampled tokens on functional-KL AUC;
2. active versus random queries on functional-KL AUC;
3. active versus random queries on orbit-aligned-NRMSE AUC;
4. inverse model versus strongest preregistered baseline on controlled-edit MRR.

Two-sided lineage-paired tests are used unless a directional alternative was preregistered before unblinding. Holm's step-down procedure controls familywise \(\alpha=0.05\). Exact adjusted p-values and simultaneous interpretation are reported.

Secondary endpoints use Benjamini–Hochberg FDR at 0.05 within named endpoint families. Exploratory analyses report unadjusted p-values only alongside an explicit exploratory label.

## 3. Primary estimand

For method/interface \(m\), lineage \(l\), and log-budget grid \(b\), define a bounded or transformed score \(S_{lmb}\). The primary query-efficiency summary is trapezoidal AUC over the fixed \(\log_2\)-budget grid, normalized to `[0,1]`:

\[
A_{lm}=\frac{1}{B_{\max}-B_{\min}}
\int_{B_{\min}}^{B_{\max}} S_{lm}(B)\,dB.
\]

The primary contrast is

\[
\Delta=E_l[A_{l,m_1}-A_{l,m_0}],
\]

where all checkpoints/edits in a lineage are averaged using fixed equal weights. For error metrics, signs are reversed so positive \(\Delta\) always favors the proposed method.

## 4. Sample size and power

The normal paired approximation requires approximately:

| Standardized lineage-paired effect | Independent lineages for 80% power, two-sided 0.05 |
|---:|---:|
| 0.20 | 197 |
| 0.25 | 126 |
| 0.30 | 88 |
| 0.35 | 65 |
| 0.40 | 50 |
| 0.50 | 32 |

The confirmatory Tier-1/Tier-2 target is **at least 144 evaluable test lineages** for the pooled primary contrasts, providing margin over the 126-lineage approximation for \(d=0.25\). Stratum-specific claims require at least 65 evaluable lineages for \(d=0.35\), or are explicitly labeled underpowered/exploratory.

Repeated checkpoints do not replace independent lineages. Before launch, `b2w stats power` simulates power under the pilot-estimated intraclass correlation and actual missingness pattern. The greater of the analytic and clustered-simulation requirements is used, inflated by 10% for technical failure.

Pilot lineages used to estimate variance are never included in the confirmatory test set.

## 5. Primary inference

### 5.1 Paired lineage-cluster bootstrap

Resample lineages with replacement 10,000 times. Preserve all nested observations for a sampled lineage, recompute the prespecified lineage summary, and obtain a percentile 95% confidence interval for \(\Delta\).

### 5.2 Paired randomization test

As a robustness test, independently flip the sign of each lineage-paired difference. Use 100,000 Monte Carlo permutations or exact enumeration when feasible. Add one to numerator and denominator when estimating the Monte Carlo p-value.

### 5.3 Mixed-effects model

The prespecified repeated-measures model is, after the endpoint-specific transform:

```text
score ~ method * channel * log2_budget + tier + architecture_stratum
      + method:held_out_axis
      + (1 + method + log2_budget | lineage_id)
```

If the maximal random-effects structure does not converge, simplify in this order: remove random correlations; remove random method slope; retain at least a lineage intercept. The simplification path and convergence diagnostics are reported. Mixed-effects estimates supplement rather than replace the bootstrap primary result.

## 6. Endpoint transforms

- KL and NRMSE: analyze `-log(metric + epsilon)`; present raw geometric means and differences as well.
- Accuracy/AUROC/coverage: lineage means; use logit transform for models when values are not exactly 0/1.
- Reciprocal rank: lineage mean on raw bounded scale plus ordinal/top-k sensitivity analyses.
- Query threshold: interval-censored survival analysis if many methods never reach threshold; otherwise report restricted mean log-budget.
- Calibration: absolute coverage error at 50%, 80%, 90%, and 95%; one aggregate expected calibration error is prespecified.

`epsilon` is fixed from numerical precision on validation data and recorded in the frozen analysis config.

## 7. Covariates and stratification

Primary randomization/paired analyses need no covariate adjustment. Precision models include only prespecified variables:

- architecture stratum;
- parameter-count bin;
- task/corpus;
- intervention category;
- checkpoint bin;
- trace channel;
- query budget;
- held-out axis.

No stepwise selection is allowed. Post-hoc interactions are exploratory.

## 8. Missing data and technical failures

A target is excluded only for one of these preregistered reasons:

- checkpoint checksum mismatch or unreadable archive;
- target model cannot reproduce its saved validation checksum within tolerance;
- trace collection fails before any method sees that target;
- corrupt query-bank entry;
- hardware failure documented by scheduler logs.

Model failure, NaN prediction, OOM caused by the tested method, failure to reach a threshold, or poor reconstruction is an outcome, not an exclusion. Assign the prespecified worst finite score where an endpoint requires a number, and separately report failure rate.

Primary analysis is intention-to-evaluate over all technically valid target lineages. A complete-case sensitivity analysis is secondary.

## 9. Outliers

No performance observation is removed because it is extreme. Invalid numeric values are handled by the failure rule. Robustness analyses report median paired differences, 20% trimmed means, and Huber mixed models where available.

## 10. Multiple checkpoints, edits, and prompts

Lineage summaries use a frozen weighting scheme:

- checkpoints: equal weight over selected early/middle/final bins;
- edit categories: equal category weight, then equal weight within category;
- prompts: aggregate to the target×budget endpoint before lineage inference;
- repeated generation samples: estimate one response distribution, not independent observations.

Alternative weightings are sensitivity analyses.

## 11. Model selection and stopping

- Hyperparameters are selected on training/validation lineages only.
- A fixed maximum training budget and validation-based early stopping rule are declared before test access.
- The confirmatory test is run once per frozen method.
- Test performance is not monitored during training.
- Adding compute after seeing a confirmatory result requires a new registered replication, not continuation of the same experiment.

## 12. Negative controls

A confirmatory run is invalidated and investigated if any control exceeds its frozen tolerance:

- shuffled target-weight labels outperform the population mean;
- manifest-order or path-only classifier predicts test identity;
- permuting target order changes predictions;
- duplicate/near-duplicate audit finds cross-split leakage;
- function-preserving head/neuron permutations change target outputs beyond numerical tolerance;
- inverse-model performance is unchanged when responses are replaced by unrelated responses.

## 13. Generalization reporting

For each held-out axis report:

1. in-distribution score;
2. OOD score;
3. absolute difference;
4. OOD/ID ratio on a direction-consistent scale;
5. lineage-cluster confidence interval;
6. number of independent lineages/families;
7. failure rate.

A pooled estimate is accompanied by forest plots or tabular stratum estimates. Heterogeneity is not hidden by the pooled mean.

## 14. Bayesian posterior diagnostics

The inverse model's learned posterior is evaluated frequentistically on held-out targets:

- empirical interval coverage;
- standardized residual mean and variance;
- NLL and continuous ranked probability score;
- calibration by tensor role/layer and by OOD stratum;
- sharpness conditional on coverage.

Post-hoc temperature scaling may be fit on validation lineages and must be applied unchanged to test lineages. Both pre- and post-calibration values are reported.

## 15. Reproducible output

The frozen analysis emits:

- tidy per-target results following `ResultRecord`;
- lineage-aggregated table;
- all confidence intervals and raw p-values;
- multiplicity-adjusted p-values;
- mixed-model coefficient table and convergence warnings;
- exclusions/failures table;
- environment, Git commit, manifest/query hashes, and config snapshot;
- plots generated only from the tidy result table.

No manually transcribed values enter the paper.
