# Phase 1 results (highway-env)

## Main comparison (3 seeds)

### in_distribution

| policy | return | collision rate | mean speed |
|---|---|---|---|
| expert | 21.56 ± 0.00 | 0.00 ± 0.00 | 19.85 ± 0.00 |
| random | 7.26 ± 0.00 | 0.82 ± 0.00 | 22.47 ± 0.00 |
| bc | 12.88 ± 0.44 | 0.57 ± 0.05 | 21.23 ± 0.42 |
| ditto_single | 14.53 ± 1.06 | 0.44 ± 0.05 | 20.87 ± 0.34 |
| ditto_multi | 15.39 ± 2.13 | 0.41 ± 0.12 | 20.24 ± 0.94 |

### shifted

| policy | return | collision rate | mean speed |
|---|---|---|---|
| expert | 20.53 ± 0.00 | 0.04 ± 0.00 | 18.02 ± 0.00 |
| random | 5.96 ± 0.00 | 0.92 ± 0.00 | 21.69 ± 0.00 |
| bc | 9.17 ± 0.72 | 0.71 ± 0.06 | 20.82 ± 0.47 |
| ditto_single | 10.04 ± 0.75 | 0.65 ± 0.02 | 20.53 ± 0.37 |
| ditto_multi | 11.68 ± 2.83 | 0.58 ± 0.13 | 19.70 ± 1.60 |

## Improved config (K=16, H=5), ditto_multi vs same baselines (3 seeds)

### in_distribution

| policy | return | collision rate | mean speed |
|---|---|---|---|
| expert | 21.56 ± 0.00 | 0.00 ± 0.00 | 19.85 ± 0.00 |
| random | 7.26 ± 0.00 | 0.82 ± 0.00 | 22.47 ± 0.00 |
| bc | 13.87 ± 0.73 | 0.49 ± 0.05 | 21.34 ± 0.54 |
| ditto_single | 18.04 ± 1.45 | 0.21 ± 0.10 | 19.23 ± 0.33 |
| ditto_multi | 20.08 ± 0.88 | 0.10 ± 0.06 | 17.86 ± 1.01 |

### shifted

| policy | return | collision rate | mean speed |
|---|---|---|---|
| expert | 20.53 ± 0.00 | 0.04 ± 0.00 | 18.02 ± 0.00 |
| random | 5.96 ± 0.00 | 0.92 ± 0.00 | 21.69 ± 0.00 |
| bc | 10.48 ± 2.15 | 0.63 ± 0.10 | 20.69 ± 1.03 |
| ditto_single | 13.60 ± 1.61 | 0.48 ± 0.07 | 18.81 ± 0.82 |
| ditto_multi | 18.27 ± 1.47 | 0.21 ± 0.11 | 16.37 ± 0.86 |

## K (retrieval modes, ditto_multi)

| setting | return (ID) | collisions (ID) | return (shift) | collisions (shift) |
|---|---|---|---|---|
| K=1 | 16.66 | 0.34 | 11.37 | 0.60 |
| K=2 | 16.61 | 0.28 | 10.47 | 0.62 |
| K=4 | 11.66 | 0.62 | 9.89 | 0.62 |
| K=8 (main) | 17.50 | 0.30 | 12.65 | 0.52 |
| K=16 | 19.99 | 0.14 | 17.72 | 0.20 |

## Contrastive negatives (ditto_multi)

| setting | return (ID) | collisions (ID) | return (shift) | collisions (shift) |
|---|---|---|---|---|
| M=0 (raw) | 10.50 | 0.70 | 7.15 | 0.88 |
| M=4 | 14.99 | 0.42 | 11.60 | 0.58 |
| M=16 (main) | 17.50 | 0.30 | 12.65 | 0.52 |
| M=32 | 15.52 | 0.40 | 12.99 | 0.50 |

## Imagination horizon (ditto_multi)

| setting | return (ID) | collisions (ID) | return (shift) | collisions (shift) |
|---|---|---|---|---|
| H=5 | 19.12 | 0.16 | 18.10 | 0.22 |
| H=10 | 18.47 | 0.22 | 15.24 | 0.38 |
| H=15 (main) | 17.50 | 0.30 | 12.65 | 0.52 |

## Expert data scale (ditto_multi)

| setting | return (ID) | collisions (ID) | return (shift) | collisions (shift) |
|---|---|---|---|---|
| 75 eps | 15.00 | 0.44 | 8.35 | 0.72 |
| 150 eps | 11.15 | 0.62 | 11.28 | 0.60 |
| 300 eps (main) | 17.50 | 0.30 | 12.65 | 0.52 |

## Expert style ratio (ditto_multi)

| setting | return (ID) | collisions (ID) | return (shift) | collisions (shift) |
|---|---|---|---|---|
| 25/75 | 19.87 | 0.14 | 17.71 | 0.22 |
| 50/50 (main) | 17.50 | 0.30 | 12.65 | 0.52 |
