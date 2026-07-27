# Bench2Drive open-loop results

val: 49 clips, 10321 frames | horizon 5

WM teacher-forced recon MSE: 0.01003 | expert-replay latent match 0.9310 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | -0.019 | 0.2163 | 0.9301 | 0.05031 |
| ditto_single | 0.031 | 0.2214 | 0.9301 | 0.04991 |
| ditto_multi | 0.036 | 0.2225 | 0.9298 | 0.05009 |
