# Conditional multimodality: empirical checks

run: /scratch/srahmani/ditto_av/outputs/phase1/main_seed0

## A. Paired expert rollouts from identical initial states

- pairs: 40 (same env reset seed, both styles)
- diverged within episode: 57%
- median first divergence step: 0.0

Same initial traffic state, different demonstrated continuation —
the expert policy is conditionally multimodal by construction.

## B. Latent-bank retrieval crosses styles

- queries: 2000, K=16, H=5
- mean start cosine of retrieved windows: 0.979
- cross-style fraction of retrieved windows: 0.36
- expert action disagreement at matched step: 0.33
- mean end-state similarity of retrieved windows: 0.910

High start similarity + substantial cross-style retrieval and
action disagreement + lower end similarity = near-identical states
with divergent expert continuations in the actual training data.
