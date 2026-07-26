# Bench2Drive open-loop results

val: 10 clips, 1475 frames | horizon 15

WM teacher-forced recon MSE: 0.02156 | expert-replay latent match 0.7300 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | 7.570 | 0.2707 | 0.6664 | 0.13078 |
| ditto_single | 7.821 | 0.2441 | 0.6790 | 0.12881 |
| ditto_multi | 7.813 | 0.2437 | 0.6775 | 0.12936 |
