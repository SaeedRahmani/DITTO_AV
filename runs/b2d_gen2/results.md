# Bench2Drive open-loop results

val: 166 clips, 39251 frames | horizon 15

WM teacher-forced recon MSE: 0.00567 | expert-replay latent match 0.8356 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.267 | 0.1758 | 0.8303 | 0.07033 |
| ditto_single | -0.158 | 0.1822 | 0.8308 | 0.07023 |
| ditto_multi | -0.173 | 0.1822 | 0.8310 | 0.07027 |
