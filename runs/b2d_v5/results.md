# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 15

WM teacher-forced recon MSE: 0.00963 | expert-replay latent match 0.8438 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.020 | 0.2070 | 0.8419 | 0.06647 |
| ditto_single | -0.008 | 0.2065 | 0.8419 | 0.06666 |
| ditto_multi | 0.007 | 0.2056 | 0.8416 | 0.06671 |
