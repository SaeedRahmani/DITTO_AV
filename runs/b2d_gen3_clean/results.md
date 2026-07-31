# Bench2Drive open-loop results

val: 166 clips, 39251 frames | horizon 15

WM teacher-forced recon MSE: 0.00425 | expert-replay latent match 0.8080 (dynamics ceiling)

| policy | action NLL | action MAE | latent match | H-step obs MSE |
|---|---|---|---|---|
| bc | 0.248 | 0.1679 | 0.8003 | 0.06840 |
| ditto_single | 0.225 | 0.1742 | 0.8011 | 0.06826 |
| ditto_multi | 0.302 | 0.1709 | 0.8013 | 0.06807 |
