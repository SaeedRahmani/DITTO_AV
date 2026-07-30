# Bench2Drive open-loop results

val: 166 clips, 39251 frames | horizon 15

WM teacher-forced recon MSE: 0.00476 | expert-replay latent match 0.8346 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | 0.162 | 0.1676 | 0.8276 | 0.07086 |
| ditto_single | 0.280 | 0.1755 | 0.8279 | 0.07076 |
| ditto_multi | 0.216 | 0.1735 | 0.8282 | 0.07078 |
