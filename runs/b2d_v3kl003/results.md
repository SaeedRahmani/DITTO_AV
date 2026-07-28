# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 15

WM teacher-forced recon MSE: 0.00982 | expert-replay latent match 0.8368 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.044 | 0.2109 | 0.8344 | 0.07348 |
| ditto_single | 0.337 | 0.2266 | 0.8343 | 0.07340 |
| ditto_multi | 0.434 | 0.2233 | 0.8332 | 0.07331 |
