# G1: egosim-as-selector validation

12 banked wp-family models, 192 val windows x 80 steps, burn-in 16; batteries: clean / launch / divergent (score = battery-mean late-half reward)

| model | sim score | pos err@H | col rate | dev-10 |
|---|---|---|---|---|
| wp_action_bc | 0.027 | 17.80 | 0.550 | 16.37/69.0 |
| wph_lw | 0.014 | 21.44 | 0.547 | 19.72/78.4 |
| wph_cap | 0.012 | 22.23 | 0.594 | 20.40/68.7 |
| wph_bc_s2 | 0.011 | 22.87 | 0.595 | 28.45/70.6 |
| wph_lw2 | 0.011 | 21.90 | 0.571 | 15.64/75.2 |
| wph_bc_s0 | 0.010 | 22.87 | 0.585 | 30.49/83.2 |
| wph_bc_s1 | 0.007 | 24.09 | 0.693 | 25.86/73.0 |
| dwp_k03nd | 0.005 | 27.20 | 0.575 | 24.07/80.8 |
| dwp_k10nd | 0.005 | 30.44 | 0.649 | 18.08/60.5 |
| dwp_k03 | 0.004 | 31.24 | 0.681 | 19.49/70.8 |
| dwp_es3k | 0.003 | 28.18 | 0.611 | 13.31/71.1 |
| dwp_v1 | 0.001 | 29.82 | 0.776 | 3.46/50.4 |

Spearman (higher-is-better orientation):
- spearman_sim_score_vs_d10_score: +0.371
- spearman_sim_score_vs_d10_completion: +0.182
- spearman_sim_reward_vs_d10_score: +0.189
- spearman_sim_reward_vs_d10_completion: +0.315
- spearman_pos_err_final_vs_d10_score: +0.168
- spearman_pos_err_final_vs_d10_completion: +0.252
- spearman_collision_rate_vs_d10_score: +0.133
- spearman_collision_rate_vs_d10_completion: +0.497

**G1 VERDICT: FAIL** (gate: sim_score vs dev-10 score >= +0.4; v0.1 latent metric was -0.60 on the same question)
