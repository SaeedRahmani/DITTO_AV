# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 15

WM teacher-forced recon MSE: 0.00982 | expert-replay latent match 0.8368 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.044 | 0.2109 | 0.8344 | 0.07348 |
| ditto_single | 0.097 | 0.2177 | 0.8346 | 0.07336 |
| ditto_multi | 0.111 | 0.2154 | 0.8341 | 0.07322 |
