# Tier-0 analytic identifiability

864000 systems, 864000 unique, channels=['probabilities', 'samples']

## Identifiability threshold (full-rank rate)

Recovery of the full canonical parameter needs design rank = input_dim+1.

| query_strategy | input_dim | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| basis | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| basis | 4 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| basis | 8 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| basis | 16 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| gaussian | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gaussian | 4 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gaussian | 8 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| gaussian | 16 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

## Below vs at/above the query threshold

below (q<d+1): full_rank=0.000 observable_fraction=0.519 orbit_nrmse=0.897 fwd_kl=0.8341

at/above (q>=d+1): full_rank=1.000 observable_fraction=1.000 orbit_nrmse=4.273e-01 fwd_kl=8.84e-02

## Exact recovery (probabilities channel, full rank)

exact_recovery_rate=1.0000 mean_orbit_nrmse=3.44e-16 mean_fwd_kl=1.47e-20

rank-deficient probabilities: orbit_nrmse_median=0.722 fwd_kl_median=0.5545

## Sampling noise (samples channel, full rank)

| sample_count | orbit_nrmse | fwd_kl | systems |
| --- | --- | --- | --- |
| 1 | 0.899 | 0.5997 | 84000 |
| 8 | 0.6755 | 0.3287 | 84000 |
| 32 | 0.4992 | 0.2059 | 84000 |
| 128 | 0.3474 | 0.1284 | 84000 |
| 512 | 0.2364 | 0.08075 | 84000 |

all-N: nrmse~N^-0.22  kl~N^-0.32

large-N (>=32): nrmse~N^-0.27  kl~N^-0.34  (theory: -0.5 and -1.0)

## Class count effect (at full rank)

| observation_channel | class_count | orbit_nrmse | fwd_kl | exact |
| --- | --- | --- | --- | --- |
| probabilities | 3 | 3.374e-16 | 5.699e-21 | 1 |
| probabilities | 5 | 3.437e-16 | 2.224e-21 | 1 |
| probabilities | 10 | 3.516e-16 | 3.613e-20 | 1 |
| samples | 3 | 0.4452 | 0.1386 | 0 |
| samples | 5 | 0.5259 | 0.2436 | 0 |
| samples | 10 | 0.6234 | 0.4238 | 0 |

## Query design: gaussian vs basis

| query_strategy | full_rank | orbit_nrmse | fwd_kl |
| --- | --- | --- | --- |
| basis | 0.5833 | 0.6557 | 0.3189 |
| gaussian | 0.5833 | 0.7031 | 0.3278 |

## Conditioning

corr(log cond, log orbit_nrmse) at full rank, samples = 0.308

## Findings

1. Identifiability is a sharp step: the canonical parameter is recoverable iff query_count>=input_dim+1. Below it full-rank rate is 0 and only 52% of parameter directions are observed; at/above it is exactly 1.0 for both gaussian and basis designs.

2. With exact probabilities and a full-rank design, recovery is algebraically exact (nrmse 3e-16, KL 1e-20) for every class count.

3. Below threshold the model also fails to generalize: held-out forward KL stays high (median 0.83), so matching behavior on too few queries recovers neither weights nor function.

4. Finite sampling converges slower than ideal Monte Carlo: orbit_nrmse~N^-0.27 and KL~N^-0.34 versus the -0.5 and -1.0 a variance-only model predicts. The Dirichlet floor biases log-odds of near-deterministic rows, so sampled black-box access is bias-limited, not variance-limited.

5. Sample complexity grows with class count: at full rank the sampled-channel forward KL rises from 0.14 to 0.42 across class counts [3, 5, 10], so large-vocabulary targets are far harder to extract from samples than from logits.
