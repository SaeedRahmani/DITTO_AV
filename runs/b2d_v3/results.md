# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 15

WM teacher-forced recon MSE: 0.00995 | expert-replay latent match 0.8439 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.031 | 0.2169 | 0.8420 | 0.07307 |
| ditto_single | 0.016 | 0.2192 | 0.8414 | 0.07345 |
| ditto_multi | 0.025 | 0.2176 | 0.8414 | 0.07338 |
