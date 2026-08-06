# Bench2Drive scenarios never spawn in our CARLA runs

**Status: OPEN, unfixed by choice.** Found 2026-08-06 while re-recording
rollout videos. The one-line fix is known and verified (below) but is
deliberately NOT applied — applying it changes what every driving number
in this project means, so it is Saeed's call when and how to land it.

## Symptom

Every CARLA run prints one of these per scenario and then drives on:

```
Skipping scenario 'ConstructionObstacleTwoWays_1' due to setup error: 'ConstructionObstacleTwoWays'
```

The route still completes and still scores; it simply contains no
scenario. The clue is that the exception text is just the scenario name
in quotes — a bare `KeyError`, swallowed by a broad `except`.

## Root cause

`leaderboard/leaderboard/scenarios/route_scenario.py:276`

```python
scenarios_list = glob.glob("{}/srunner/scenarios/*.py".format(
    os.getenv('SCENARIO_RUNNER_ROOT', "./")))
```

`get_all_scenario_classes()` discovers scenario classes by globbing the
filesystem, not by importing a package. With `SCENARIO_RUNNER_ROOT`
unset it falls back to `./`, and our jobs `cd $B2D` first, so it globs
`Bench2Drive/srunner/scenarios/*.py` — a path that does not exist. The
real tree is `Bench2Drive/scenario_runner/srunner/scenarios/` (38 files).

The registry therefore comes up empty, `all_scenario_classes[type]`
raises `KeyError` for every scenario, and `route_scenario.py:341` catches
it as a "setup error" and continues. Bench2Drive's own
`leaderboard/scripts/run_evaluation.sh:11` exports the variable; our
sbatch scripts were written from the `PYTHONPATH` lines and never
picked it up.

## Scope — every CARLA run this project has done

Counted from the job logs (`prepared` = routes started, `skipped` =
scenarios dropped):

| job | what it produced | prepared | skipped |
|---|---|---|---|
| 10587720 / 10587721 | v0.3.2 test-10 verdict, DS 82.80 | 15 / 15 | 15 / 15 |
| 10582571 | v0.3.1 W3 re-gate | 15 | 15 |
| 10577433 | v0.2 twenty-video job | 10 | 10 |
| 10591365 / 10591366 | video re-record | 14 | 14 |

Affected: every test-10 gate, both full 220-route runs (they go through
`carla_eval_chain.sbatch` via `v02_bench220_submit.sh`), and every video
recorded before this note. So all driving scores to date measure
background traffic on the route geometry, with none of the 44
safety-critical scenarios present.

NOT affected: training. Nothing in the training path runs CARLA — the
world is EgoSim over logged Bench2Drive clips, and those recordings were
collected by the dataset authors with scenarios intact. Verified: no
training sbatch references carla or the leaderboard evaluator.

## The fix, and what it costs

One line, next to the existing `PYTHONPATH` export in every sbatch that
runs `leaderboard_evaluator.py` (7 files on main, ~6 per version branch):

```bash
export SCENARIO_RUNNER_ROOT=$B2D/scenario_runner
```

Verified on two routes (job 10592289, v0.3.2 axis-3 checkpoint): 0
scenarios skipped, and the scenario actors appear in the state log for
the first time — 1 emergency vehicle on 25378, 10 construction props on
25424. Scores on the real routes:

| route | scenario | skipped (all runs to date) | scenario live |
|---|---|---|---|
| 25378 | YieldToEmergencyVehicle | DS 100.0 | DS 70.0, one yield infraction |
| 25424 | ConstructionObstacleTwoWays | DS 100.0 | DS 22.4, 52.9% completion, 2 layout collisions |

Two routes are not a benchmark, but they bound the direction: the real
routes are materially harder, and the failure modes are exactly the ones
the scenarios are designed to provoke.

## What landing it implies

- Version-to-version comparisons stay internally consistent — every
  variant took the same easy exam — but the absolute numbers are not
  comparable to published Bench2Drive baselines, so the claim of
  clearing them does not hold as stated.
- Re-running the gates and the 220-route benchmark is the only way to
  restore comparability; expect lower numbers.
- The reward/world-model findings are untouched: they were measured
  in-WM, not in CARLA.

## Reproducing the diagnosis

```bash
grep -c "Skipping scenario" /scratch/$USER/ditto_av/outputs/slurm-<jobid>.out
ls /scratch/$USER/ditto_av/Bench2Drive/srunner            # missing
ls /scratch/$USER/ditto_av/Bench2Drive/scenario_runner/srunner/scenarios | wc -l   # 38
```
