# DITTO-AV: Offline Imitation Learning with World Models for Autonomous Driving

This repository adapts **DITTO** ([DeMoss et al., 2023](https://arxiv.org/abs/2302.03086),
offline imitation learning inside a learned world model) to **autonomous
driving**, with a new multimodal latent-matching objective.

**Core idea.** Train an RSSM world model on offline driving logs. Then learn a
policy *fully offline* by unrolling it inside the world model from expert
start states and rewarding latent closeness to expert trajectories — on-policy
imitation without a simulator, which corrects the covariate shift that breaks
behavior cloning.

**What's new vs. DITTO (the paper contribution).**

1. **Multimodal nearest-mode matching** (`reward_mode: multi`). Driving is
   multimodal: from the same blocked-lane state, one expert overtakes and
   another slows down. DITTO's single-trajectory reward penalizes every valid
   mode but the demonstrated one. We retrieve the K expert windows whose start
   latent is nearest to the rollout's start and reward the *best-matching*
   mode (max over K), so reproducing *any* expert behavior is rewarded.
   Two stabilizers proved necessary in driving latent spaces: a
   **contrastive baseline** (`n_negatives`) that subtracts the mean
   similarity to random expert windows — raw latent similarity to *any*
   plausible traffic state is ~0.9, leaving almost no signal — and a
   **BC trust region** (`bc_init`, `bc_kl_coef`) that keeps imagination RL
   from drifting into world-model exploits.
2. **Vectorized world model.** Agent-centric kinematic features instead of
   pixels: orders of magnitude cheaper, reproducible on CPU.
3. **Driving-native evaluation.** Closed-loop collision rate / speed /
   return, in-distribution and under traffic-density shift, against expert,
   BC, and single-mode DITTO baselines.

## Layout

```text
ditto_av/            the package (new, self-contained)
  envs.py            highway-env factory + vector featurizer
  expert.py          scripted two-style (multimodal) expert
  collect.py         demonstration collection
  data.py            trajectory store, latent bank
  rewards.py         max_cos, single/multi latent matching, lambda-returns
  models/            RSSM (categorical latents), vector encoder/decoder, actor-critic
  trainers/          world-model, DITTO actor-critic, BC trainers
  evaluate.py        closed-loop evaluation harness
  bench2drive.py     Bench2Drive (CARLA) -> DITTO-AV data adapter
scripts/run_pipeline.py   A-to-Z pipeline
configs/av.yaml           main experiment; configs/smoke.yaml for a 2-min test
tests/                    pytest suite
src/, paper/              original DITTO Atari code + paper (reference, untouched)
```

## Quickstart

```sh
pip install -r requirements_av.txt
python -m pytest tests/ -q                     # sanity
python scripts/run_pipeline.py --config configs/smoke.yaml --stage all   # ~2 min
python scripts/run_pipeline.py --config configs/av.yaml --stage all      # full run
```

Stages can be run individually: `--stage collect | wm | policies | eval`.
Results land in `runs/<name>/results/results.md`.

## Data

- **highway-env** (bundled, procedural): the development benchmark. A
  scripted expert with two styles (aggressive = overtakes, conservative =
  slows down) produces genuinely multimodal demonstrations; a noisy variant
  adds world-model coverage.
- **Bench2Drive** ([HuggingFace](https://huggingface.co/datasets/rethinklab/Bench2Drive),
  Apache-2.0): the paper benchmark. `ditto_av/bench2drive.py` converts clips
  (per-frame annotation JSONs) into the same vectorized format:

```python
from ditto_av.bench2drive import clips_to_npz
clips_to_npz([Path("clips/AccidentTwoWays_..._Weather10")], "runs/b2d/data/expert.npz")
```

See `PAPER_PLAN.md` for the paper roadmap, experiment matrix, and how this
scales to the full Bench2Drive closed-loop benchmark.

## Performance note

Set single-threaded BLAS (`torch.set_num_threads(1)`, done automatically in
`scripts/run_pipeline.py`) — the small sequential RSSM ops are 15-30x slower
under multi-threaded BLAS on Apple Silicon.

## Credit: original DITTO

This project began as a fork of and builds directly on
[**DITTO** by Branton DeMoss et al.](https://github.com/brantondemoss/DITTO)
([paper: *DITTO: Offline Imitation Learning with World Models*, arXiv:2302.03086](https://arxiv.org/abs/2302.03086)).
The RSSM world-model core in `ditto_av/models/` is adapted from that
codebase, and the original pre-release Atari implementation is preserved
unchanged in `src/` with the paper sources in `paper/`. If you use the
world-model imitation ideas here, please cite DITTO:

```bibtex
@article{demoss2023ditto,
  title={DITTO: Offline Imitation Learning with World Models},
  author={DeMoss, Branton and Duckworth, Paul and Hawes, Nick and Posner, Ingmar},
  journal={arXiv preprint arXiv:2302.03086},
  year={2023}
}
```
