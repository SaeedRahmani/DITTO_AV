# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 15

WM teacher-forced recon MSE: 0.01097 | expert-replay latent match 0.7998 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.028 | 0.2159 | 0.7975 | 0.09250 |
| ditto_single | 0.009 | 0.2209 | 0.7970 | 0.09245 |
| ditto_multi | 0.009 | 0.2202 | 0.7968 | 0.09273 |
