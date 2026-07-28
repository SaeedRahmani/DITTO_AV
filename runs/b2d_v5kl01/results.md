# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 15

WM teacher-forced recon MSE: 0.00963 | expert-replay latent match 0.8438 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.020 | 0.2070 | 0.8419 | 0.06647 |
| ditto_single | 0.068 | 0.2122 | 0.8415 | 0.06672 |
| ditto_multi | 0.049 | 0.2097 | 0.8416 | 0.06668 |
