# DELFTBLUE.md — AI operating guide for this project on the DelftBlue cluster

Read this fully before doing anything on DelftBlue. It encodes the cluster's
constraints, this project's layout there, and the agreed next steps.
Project context: see `README.md` (method/code) and `PAPER_PLAN.md` (roadmap).
DelftBlue docs: <https://doc.dhpc.tudelft.nl/delftblue/>

## Access

- SSH alias `delftblue` (configured in `~/.ssh/config` on the laptop):
  `login.delftblue.tudelft.nl`, user `srahmani`, key auth.
- Login nodes (`login0X`) are shared: **no training, no data processing, no
  long jobs there** — only editing, git, tiny tests, and job submission.
  Everything heavy goes through SLURM.

## Storage rules (critical)

| Location | Path | Properties | Use for |
|---|---|---|---|
| Home | `/home/srahmani` | ~30 GB quota, small, was nearly full (cleaned to ~7.6 GB on 2026-07-26 by purging pip/uv caches) | dotfiles only. **Install nothing here.** |
| Scratch | `/scratch/srahmani` | Huge, fast (BeeGFS), **NOT backed up, cleaned at regular intervals** | everything: code clone, venv, data, outputs |

Because scratch can be wiped:

- Code safety = git. Commit and push anything worth keeping to
  `github.com/SaeedRahmani/DITTO_AV` (public). Never keep the only copy of
  code on scratch.
- Data safety = re-downloadable (Bench2Drive is on HuggingFace). Don't treat
  the downloaded data as precious; do record exactly *what* was downloaded
  (clip lists) in git.
- Checkpoint/result safety = sync important run outputs back to the laptop
  (`rsync`) or commit small result files (json/md) to git promptly.
- Touching files periodically does NOT make them safe; assume anything on
  scratch can disappear between sessions. Re-verify before relying on it.

## Project layout on the cluster (created 2026-07-26)

```text
/scratch/srahmani/ditto_av/
  DITTO_AV/     git clone (branch main = dev, commit c7af528 or later)
  envs/ditto/   python venv (module python/3.10.12 under 2024r1)
  data/         datasets go here (empty until download step)
  outputs/      SLURM run outputs / checkpoints (set run_dir here)
  cache/pip/    pip cache (PIP_CACHE_DIR) — keeps home quota safe
```

## Environment activation (every session / inside every job script)

```sh
module load 2024r1 python/3.10.12
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
export PIP_CACHE_DIR=/scratch/$USER/ditto_av/cache/pip
export OMP_NUM_THREADS=1   # see README: multithreaded BLAS is 15-30x slower for this workload
```

Installed in the venv: CUDA-build torch, numpy, gymnasium, highway-env,
pyyaml, pytest. Login nodes have no GPU — `torch.cuda.is_available()` is
False there; that's expected. GPUs exist only on `gpu-*` partitions.

Sanity check after any environment change:

```sh
cd /scratch/$USER/ditto_av/DITTO_AV && python -m pytest tests/ -q
```

## SLURM basics for this project

Partitions (2026): `compute-p1/p2` (CPU), `gpu-a100` (4x A100 80GB/node),
`gpu-v100`, `gpu-a100-small` (MIG slices — fine for our small models).
Templates live in `scripts/slurm/`. The user's SLURM accounts (verified
2026-07-26): `research-ceg-tp` (used in the templates) and `innovation`.

- `scripts/slurm/test.sbatch` — 10-min CPU smoke: runs pytest + the smoke
  pipeline. Submit this FIRST after any fresh setup.
- `scripts/slurm/download_b2d.sbatch` — Bench2Drive download into
  `/scratch/$USER/ditto_av/data/bench2drive/`. **Compute nodes have NO
  outbound internet** (verified 2026-07-26, job 10521023; no proxy
  configured) — run its body on a login node with `nohup` instead (see the
  script header). First 60-clip batch downloaded + validated 2026-07-26.
- `scripts/slurm/validate_b2d.sbatch` — extracts `anno/` from downloaded
  tarballs and parses every manifest clip via `scripts/validate_b2d.py`;
  run after every download batch.
- `scripts/slurm/train.sbatch` — full pipeline run (CPU is fine for the
  highway-env phase; the model is tiny. GPU only pays off after scaling).

Submit with `sbatch scripts/slurm/<file>`; monitor with `squeue -u $USER`;
logs land in `/scratch/$USER/ditto_av/outputs/slurm-%j.out`.

## Status (updated 2026-07-26, evening)

On 2026-07-26 the user authorized full autonomous execution ("do all …
without pause"): Phase-1 paper sweep + issue fixes + Phase 2. Superseded
the earlier per-batch download gate for this work stream.

1. **Verify state** — DONE (pytest green, smoke passes on cluster).
2. **Data**: 60-clip batch validated (see above). Manifest extended to
   **297 stratified clips (~94 GB)**; extended download running via
   login-node nohup (`outputs/b2d_download2.log`), with a detached chain
   (`outputs/b2d_chain.log`) that auto-submits `validate_b2d.sbatch` and
   then Phase-2 v2 when it finishes.
3. **Phase-1 sweep** — SUBMITTED 2026-07-26: 15 jobs (10521411-25) via
   `scripts/phase1_sweep.py generate` + `scripts/slurm/phase1.sbatch`:
   3 seeds main, K∈{1,2,4,16}, negatives∈{0,4,32}, H∈{5,10} (reusing the
   seed-0 world model via job dependency), data scale {75,150}, style 25/75.
   Aggregate with `scripts/phase1_sweep.py aggregate` → commit
   `runs/phase1/phase1_results.{md,json}`.
4. **Phase-2 (Bench2Drive)** — pipeline IMPLEMENTED (`scripts/run_b2d.py`,
   `configs/b2d.yaml`, `scripts/slurm/phase2.sbatch`): continuous Gaussian
   actor, offline training, open-loop eval on held-out clips (WM prediction
   error, action NLL/MAE, latent imitation score vs expert-replay ceiling).
   v1 job on 60 clips: 10521595. v2 on ~297 clips auto-chains after the
   download. Closed-loop CARLA eval remains a separate future task (needs
   CARLA 0.9.15 setup — check module list / apptainer).

## Conventions for the AI working on DelftBlue

- All commits under the user's identity only — **never add Claude
  attribution/co-author trailers** (explicit user requirement).
- Ask before: deleting anything outside `/scratch/$USER/ditto_av/`,
  downloading >10 GB, or submitting jobs requesting >8 GPUs·h.
- Long-running work: always `sbatch`, never foreground SSH; poll `squeue`.
- If home quota errors appear (`Disk quota exceeded`), the culprit is almost
  always a cache writing to `$HOME/.cache` — redirect it to scratch
  (`XDG_CACHE_HOME=/scratch/$USER/ditto_av/cache`), don't delete user files.
