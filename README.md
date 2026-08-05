# DITTO-AV: Offline Imitation Learning with World Models for Autonomous Driving

This repository adapts **DITTO** ([DeMoss et al., 2023](https://arxiv.org/abs/2302.03086),
offline imitation learning inside a learned world model) to **autonomous
driving**.

## Status — under active development

Numbers below are the **current** measurements, not final ones; they are
updated periodically as runs land. Scores are Bench2Drive closed-loop driving
score (DS) in CARLA — the only verdict we count (in-sim reward never grades
itself). Per-run ledgers with job ids: `V02_PLAN.md`, `V03_PLAN.md`,
`V031_PLAN.md`.

| line | what it changed | current headline | state |
|---|---|---|---|
| v0.1 | faithful DITTO: RSSM world model, reward = whole-latent match | DS 22.10 (220 routes) | frozen — lost to BC |
| **v0.2** | **the recorded log is the world** (replayed traffic, analytic ego); reward = ego-state match | **DS 76.10 / 75.88 (220 routes)**, 85.63 / 83.60 (dev-10) | frozen — best so far |
| v0.3 | **+ learned reactive traffic** (other agents respond to the ego) | DS 82.53 (dev-10); vehicle collisions 6 = lowest of any model here | frozen — reactivity dividend confirmed |
| v0.3.1 | + static map geometry in the training world | DS 66.01 / 74.89 (dev-10) | closed — negative result, cause measured |
| v0.3.2 | reward channels for motion smoothness | — | ongoing |

**How they differ.** Each version changes *what the training world is made of*
and *what the reward grades*. v0.1 dreamed the whole scene in a learned latent
and graded that latent — which mostly graded traffic, not driving. v0.2 stopped
dreaming: the logged clip replays as the world, the ego moves by analytic
kinematics, and the reward grades only the ego's own state against the expert's
in that same scene. v0.3 gives the replayed traffic its agency back with a
learned per-agent model, so the world reacts to the ego.

**v0.2 vs. v0.3, concretely.** v0.3 wins exactly the failure class it targets —
vehicle-to-vehicle collisions drop from 9/12 to 6 — but loses more elsewhere:
its training world contains no static geometry, so driving boldly near walls is
free in training and costs 7 layout collisions in CARLA (v0.2: 0–1). That is
the whole gap. v0.3.1 tried to close it and could not: the offending objects
(fences, props, vegetation) sit *on* drivable area and appear in no available
data source.

### Next experiments (todo)

- **E3 — factored-latent DITTO retest**: latent matching inside the
  (ego | learned-traffic) factorization; closes the question v0.1 opened.
- **v0.3.2 smoothness**: yaw-rate reward channel, then in-dream plan-churn
  penalty / policy-side temporal-consistency loss.
- **220-route run for v0.3's line**: only once a dev-10 gate is actually cleared.
- **Paper**: v0.2 headline + v0.3 reactivity dividend + the v0.3.1 negative result.

## The v0.1 approach (original framing)

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

See `V02_PLAN.md`, `V03_PLAN.md`, and `V031_PLAN.md` for the plans, experiment
matrices, pre-registered gates, and status ledgers of each version.

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
