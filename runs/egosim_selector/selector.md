# G1: egosim-as-selector validation

12 banked wp-family models, 192 val windows x 40 steps, burn-in 16; batteries: clean / launch / divergent (score = battery-mean late-half reward)

| model | sim score | pos err@H | col rate | dev-10 |
|---|---|---|---|---|
| wp_action_bc | 0.228 | 5.81 | 0.276 | 16.37/69.0 |
| wph_cap | 0.103 | 10.21 | 0.408 | 20.40/68.7 |
| wph_lw | 0.100 | 10.71 | 0.382 | 19.72/78.4 |
| wph_lw2 | 0.089 | 10.91 | 0.436 | 15.64/75.2 |
| wph_bc_s1 | 0.084 | 11.05 | 0.464 | 25.86/73.0 |
| wph_bc_s0 | 0.083 | 11.52 | 0.451 | 30.49/83.2 |
| wph_bc_s2 | 0.072 | 12.05 | 0.438 | 28.45/70.6 |
| dwp_k03nd | 0.044 | 14.21 | 0.438 | 24.07/80.8 |
| dwp_es3k | 0.030 | 14.99 | 0.470 | 13.31/71.1 |
| dwp_k10nd | 0.028 | 15.28 | 0.517 | 18.08/60.5 |
| dwp_k03 | 0.017 | 16.65 | 0.569 | 19.49/70.8 |
| dwp_v1 | 0.007 | 16.68 | 0.625 | 3.46/50.4 |

Spearman (higher-is-better orientation):
- spearman_sim_score_vs_d10_score: +0.259
- spearman_sim_score_vs_d10_completion: +0.245
- spearman_sim_reward_vs_d10_score: +0.259
- spearman_sim_reward_vs_d10_completion: +0.245
- spearman_pos_err_final_vs_d10_score: +0.259
- spearman_pos_err_final_vs_d10_completion: +0.245
- spearman_collision_rate_vs_d10_score: +0.224
- spearman_collision_rate_vs_d10_completion: +0.308

**G1 VERDICT: FAIL** (gate: sim_score vs dev-10 score >= +0.4; v0.1 latent metric was -0.60 on the same question)
