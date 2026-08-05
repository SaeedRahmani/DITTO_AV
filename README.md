# DITTO-AV: Offline Imitation Learning with World Models for Autonomous Driving

This repository adapts **DITTO** ([DeMoss et al., 2023](https://arxiv.org/abs/2302.03086),
offline imitation learning inside a learned world model) to **autonomous
driving**, with a driving-native factorization of the world and its reward.

**Core idea.** Build a world model from offline driving logs. Then learn a
policy *fully offline* by unrolling it inside that world from expert start
states and rewarding closeness to expert trajectories — on-policy imitation
without a simulator, which corrects the covariate shift that breaks behavior
cloning.

**What's new vs. DITTO (the paper contribution).**

1. **Factored world, not a generated one.** The ego is advanced by analytic
   kinematics — never learn what is already known — while everything exogenous
   (traffic, route, lights) comes from the log, either replayed directly or
   advanced by a learned per-agent traffic model. Only the part replay cannot
   provide, how other agents react to the ego, is ever learned, which is also
   the only part the policy could exploit.
2. **Ego-state matching reward.** A time-tolerant kernel (`tau`) scores the
   simulated ego against the expert's own trajectory in position, heading and
   speed (`sigma_p`, `sigma_yaw`, `sigma_v`). Traffic never enters the reward,
   so it grades driving rather than the scene. Two components proved necessary:
   a **broad second position kernel** (`sigma_p2`, `p2_weight`) because a
   single tight kernel floors out exactly where recovery must be learned, and
   **rear-impact rejection** (`penalty_ignore_rear`) so a slower-than-log ego
   rear-ended by non-reactive replayed traffic is not charged for it.
3. **Driving-native evaluation.** Closed-loop Bench2Drive in CARLA — driving
   score, route completion, and the per-class collision decomposition — against
   same-network BC baselines. The training world never grades itself.

## Status — under active development

The numbers below are current measurements, not final results, and are updated
periodically as runs land. Scores are Bench2Drive closed-loop **driving score
(DS)** in CARLA. Full per-run ledgers with job ids live in `V02_PLAN.md`,
`V03_PLAN.md`, and `V031_PLAN.md`.

| line | training world | current DS | state |
|---|---|---|---|
| v0.1 | learned RSSM latent; reward = whole-latent match | 22.10 (220 routes) | frozen |
| v0.2 | log replayed as the world; reward = ego-state match | 76.10 / 75.88 (220 routes); 85.63 / 83.60 (dev-10) | frozen |
| v0.3 | v0.2 + learned reactive traffic model | 82.53 (dev-10) | frozen |
| v0.3.1 | v0.3 + static map geometry | 66.01 / 74.89 (dev-10) | reopened |
| v0.3.2 | v0.3 + smoothness reward channels | in progress | active |

Two arms are reported where two seeds/configs were run. **dev-10** is the
development gate: 10 Bench2Drive routes (A-half 3514, 3255, 26405, 25381,
25378; B-half 25424, 2091, 27494, 17569, 28198) × 3 repetitions = 30 runs.
**220 routes** is the full Bench2Drive closed-loop benchmark, run only on a
model that has already cleared dev-10.

### Videos

Closed-loop CARLA rollouts on dev-10 routes the policy completes cleanly — DS
100, route completion 100%, no collisions. Bird's-eye renders are the simulated
state, camera views are CARLA. Full-quality mp4s for every route are in
[docs/media/](docs/media/).

<table>
<tr>
<td width="50%"><img src="docs/media/v03_route25424_2d.gif" width="100%" alt="Bird's-eye rollout, construction obstacle on a two-way road, Town11"><br><sub>Construction obstacle, two-way road — Town11</sub></td>
<td width="50%"><img src="docs/media/v03_route25378_2d.gif" width="100%" alt="Bird's-eye rollout, yielding to an emergency vehicle, Town03"><br><sub>Yielding to an emergency vehicle — Town03</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/media/v03_route25424_3d.gif" width="100%" alt="Camera rollout, construction obstacle on a two-way road, Town11"><br><sub>Construction obstacle, two-way road — Town11</sub></td>
<td width="50%"><img src="docs/media/v03_route25381_3d.gif" width="100%" alt="Camera rollout, hazard at side lane, Town05"><br><sub>Hazard at side lane — Town05</sub></td>
</tr>
</table>

## Layout

```text
ditto_av/            the package (self-contained)
  egosim.py          the training world: log replay + analytic ego kinematics
  reactive.py        learned traffic model, autoregressive rollout (v0.3+)
  layout.py          static map geometry from OpenDRIVE; layout_torch.py = torch port
  rewards.py         ego-state matching kernels, latent matching, lambda-returns
  tracks.py          expert trajectory targets; tracker_torch.py = pure-pursuit tracker
  models/            RSSM (categorical latents), vector encoder/decoder, actor-critic
  trainers/          world-model, DITTO actor-critic, BC trainers
  evaluate.py        closed-loop evaluation harness
  bench2drive.py     Bench2Drive (CARLA) -> DITTO-AV data adapter
  carla_agent.py     deployment agent for the CARLA leaderboard
  envs.py            highway-env factory + vector featurizer (dev benchmark)
  expert.py          scripted two-style (multimodal) expert for highway-env
scripts/run_pipeline.py   A-to-Z pipeline
scripts/slurm/            cluster jobs (training, CARLA eval, video rendering)
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
