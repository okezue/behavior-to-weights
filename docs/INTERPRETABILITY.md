# Interpreting the Inverse Model

The inverse model is itself a scientific object. The central interpretability question is not only “what weights were predicted?” but “what behavioral evidence caused the prediction?”

## 1. Paired causal contrasts

Create target pairs differing in exactly one intervention. For each pair:

1. collect responses to the same query bank;
2. compare inferred latent states and parameter posteriors;
3. patch selected target responses from edited to base transcript and vice versa;
4. identify inverse-model activations that mediate edit classification/localization;
5. validate by ablating or patching those inverse activations.

Ground-truth edit masks make these analyses objectively scorable.

## 2. Transcript patching

`transcript_patch_effect` replaces selected query/response pairs from target A with target B and measures the change in predicted coordinates. Aggregate effects by:

- query source;
- prompt token/position;
- target layer/tensor role;
- inferred circuit/edit;
- observation channel;
- query budget.

Patching is preferable to correlation because it is an explicit intervention on model input.

## 3. Behavior-to-weight attribution

For diagnostic batches, compute

\[
J_{a,i,f}=\frac{\partial \hat\theta_a}{\partial o_{i,f}},
\]

where `a` is a weight address, `i` a query, and `f` an observation feature. The repository includes a full Jacobian helper for small cases and integrated gradients for selected coordinates.

Large studies should use:

- vector-Jacobian products for tensor-role summaries;
- randomized Jacobian sketches;
- integrated gradients with multiple baselines;
- SmoothGrad-like response perturbations;
- causal response ablation to confirm saliency.

Attribution is reported only when intervention-based checks agree in direction.

## 4. Minimal behavioral witnesses

For each correctly inferred property/circuit, solve a sparse subset problem:

1. begin with the full query set;
2. greedily remove queries while retaining posterior odds or localization rank;
3. verify the subset on unseen lineages;
4. compare against random subsets of equal size;
5. test whether the witness transfers across architectures.

This may reveal compact diagnostic languages invented by the active policy.

## 5. Latent representation analysis

Probe the trace latent for:

- architecture and parameter count;
- training step;
- seed/lineage;
- corpus/task;
- optimizer;
- intervention type/location;
- target loss and behavioral propensities.

Use nested cross-validation by lineage. Linear probe success is descriptive; causal claims require latent intervention or decoder-mediated tests.

### Disentanglement tests

- train a probe on one factor while holding correlated factors balanced;
- orthogonalize or adversarially remove one factor and measure others;
- test latent arithmetic for composed edits;
- train on edit A and B separately, evaluate A+B composition;
- compare representational similarity across inverse-model random seeds.

## 6. Intervention transfer

Mechanistic recovery is strongest when an inferred edit can be applied to a clean parent and reproduce the target behavior.

For predicted delta \(\widehat{\Delta}\):

1. apply it to the parent checkpoint;
2. evaluate the edited and target models on held-out diagnostic and natural prompts;
3. compare to norm-matched random, wrong-layer, and nearest-train-edit controls;
4. report treatment-effect correlation and causal mediation, not only parameter overlap.

## 7. Observable parameter subspace

Estimate empirical output Fisher or Jacobian Gram matrices under each query distribution. Compare:

- effective rank and eigenvalue decay;
- overlap between high-observability directions and known edit subspaces;
- posterior variance versus Fisher eigenvalue;
- functional effect of equal-norm perturbations in high- and low-observability directions;
- changes across token, top-k, and full-logit interfaces.

Exact zero directions should include declared parameter symmetries and inactive components. Apparent recovery along a theoretically invisible direction is a leakage alarm.

## 8. Avoiding interpretability circularity

Do not select “interesting” circuits based on test performance and then claim confirmatory localization on the same targets. Split intervention categories or lineages into discovery and confirmation sets. Freeze attribution thresholds and visualization choices where they affect quantitative claims.
