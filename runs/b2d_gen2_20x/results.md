# Bench2Drive open-loop results

val: 166 clips, 39251 frames | horizon 15

WM teacher-forced recon MSE: 0.00460 | expert-replay latent match 0.8386 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | 1.633 | 0.1623 | 0.8286 | 0.07023 |
| ditto_single | 2.296 | 0.1689 | 0.8291 | 0.07003 |
| ditto_multi | 2.139 | 0.1703 | 0.8287 | 0.07015 |
