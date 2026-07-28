# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 15

WM teacher-forced recon MSE: 0.00982 | expert-replay latent match 0.8368 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.044 | 0.2109 | 0.8344 | 0.07348 |
| ditto_single | 0.013 | 0.2161 | 0.8347 | 0.07337 |
| ditto_multi | 0.030 | 0.2136 | 0.8343 | 0.07324 |
