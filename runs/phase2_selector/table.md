# Phase-2: in-model metrics vs closed-loop (selector study)

17 control-family (run, policy) pairs; ground truth = banked 3-route 3x3 (all) and dev-10 (subset).

| model | latent match | divergence | H-step MSE | MAE | NLL | 3x3 compl | 3x3 score | dev-10 compl |
|---|---|---|---|---|---|---|---|---|
| gen2_10x_s1 BC (control) | 0.8174 | 0.0078 | 0.07229 | 0.1714 | 0.07 | 87.2 | 25.92 | - |
| gen3_clean BC (control) | 0.8003 | 0.0077 | 0.06840 | 0.1679 | 0.25 | 82.7 | 25.36 | 75.3 |
| gen2_10x_s2 BC (control) | 0.8248 | 0.0086 | 0.06960 | 0.1654 | 0.16 | 72.5 | 26.71 | - |
| gen2_10x BC (control) | 0.8276 | 0.0070 | 0.07086 | 0.1676 | 0.16 | 65.8 | 22.12 | 70.1 |
| kl01 multi (control) | 0.8341 | 0.0027 | 0.07323 | 0.2141 | 0.10 | 64.6 | 6.38 | 53.5 |
| gen3_wp BC (wp-action) | 0.8114 | 0.0046 | 0.06829 | 0.0202 | -12.60 | 58.4 | 11.99 | 60.3 |
| gen2_20x multi (control) | 0.8287 | 0.0099 | 0.07015 | 0.1703 | 2.14 | 57.6 | 6.32 | - |
| kl01_20x multi (control) | 0.7590 | 0.0427 | 0.07833 | 0.2119 | 8.69 | 56.5 | 12.94 | - |
| kl01_5x multi (control) | 0.8046 | 0.0104 | 0.07365 | 0.1969 | 2.28 | 56.3 | 17.11 | 63.7 |
| gen2_10x_s2 multi (control) | 0.8253 | 0.0081 | 0.06945 | 0.1649 | 0.50 | 54.3 | 15.48 | - |
| kl015 multi (control) | 0.8343 | 0.0025 | 0.07324 | 0.2136 | 0.03 | 54.2 | 6.93 | - |
| gen2_10x_s1 multi (control) | 0.8173 | 0.0080 | 0.07229 | 0.1803 | 0.21 | 49.2 | 9.97 | - |
| gen2_10x multi (control) | 0.8282 | 0.0064 | 0.07078 | 0.1735 | 0.22 | 48.1 | 16.80 | 64.2 |
| gen2_10x single (control) | 0.8279 | 0.0067 | 0.07076 | 0.1755 | 0.28 | 47.5 | 11.22 | - |
| kl02 multi (control) | 0.8343 | 0.0024 | 0.07324 | 0.2155 | 0.02 | 46.9 | 4.45 | - |
| gen2 multi (control) | 0.8310 | 0.0046 | 0.07027 | 0.1822 | -0.17 | 46.3 | 14.72 | - |
| kl01k16 multi (control) | 0.8341 | 0.0027 | 0.07322 | 0.2154 | 0.11 | 39.4 | 3.81 | - |
| v5kl01 multi (control) | 0.8416 | 0.0023 | 0.06668 | 0.2097 | 0.05 | 27.5 | 9.19 | - |
| gen3_wp multi (wp-action) | 0.8068 | 0.0092 | 0.06848 | 0.0309 | -12.32 | 25.1 | 7.34 | - |

## Spearman rank correlations (control family)

| metric | vs 3x3 completion | vs 3x3 score | vs dev-10 completion |
|---|---|---|---|
| latent_match | -0.60 (n=17) | -0.68 (n=17) | -0.70 (n=5) |
| divergence | +0.56 (n=17) | +0.49 (n=17) | +0.40 (n=5) |
| hstep_obs_mse | -0.03 (n=17) | -0.31 (n=17) | -0.80 (n=5) |
| action_mae | -0.55 (n=17) | -0.73 (n=17) | -0.90 (n=5) |
| action_nll | +0.32 (n=17) | +0.23 (n=17) | +0.30 (n=5) |

## Within-objective slices (vs 3x3 completion)

| metric | multi-only | bc-only |
|---|---|---|
| latent_match | -0.47 (n=12) | -0.80 (n=4) |
| divergence | +0.58 (n=12) | +0.40 (n=4) |
| hstep_obs_mse | +0.34 (n=12) | +0.20 (n=4) |
| action_mae | -0.20 (n=12) | +0.80 (n=4) |
| action_nll | +0.59 (n=12) | -0.40 (n=4) |
