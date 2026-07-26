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

## Agreed next steps (in order — do not skip ahead)

Status when this file was written: transfer + env DONE, tests passing on the
cluster. **No data downloaded, no training run yet** (user's explicit
instruction).

1. **Verify state**: ssh in, `git -C /scratch/$USER/ditto_av/DITTO_AV pull`,
   run pytest, submit `test.sbatch`, confirm it passes.
2. **Data download (only when the user says go)**: Bench2Drive base split
   (1000 clips, avg ~335 MB, ≈ 335 GB total — measured from the HF listing
   2026-07-26) from HF `rethinklab/Bench2Drive` into `data/bench2drive/`.
   Download via login-node `nohup` (compute nodes have no internet); keep a
   manifest of downloaded files in git. DONE 2026-07-26 for a stratified
   60-clip batch (7.9 GB, all 43 scenario types, smallest clips per
   scenario — see `manifests/b2d_clips.txt`): all 60 parse with
   `ditto_av/bench2drive.py`, combined npz has 10,766 frames. Next batches
   only on explicit user go-ahead.
3. **Phase-1 scale-up on cluster (highway-env)**: 3 seeds × {bc,
   ditto_single, ditto_multi}, K/H/negatives ablations per `PAPER_PLAN.md`;
   `run_dir` under `/scratch/$USER/ditto_av/outputs/`. Commit result
   json/md files back to git.
4. **Phase-2 (Bench2Drive)**: continuous-action Gaussian actor (see
   PAPER_PLAN), world-model training on converted clips (GPU helps here),
   closed-loop CARLA evaluation last (CARLA on DelftBlue needs its own
   setup — check module list / apptainer; treat as a separate task).

## Conventions for the AI working on DelftBlue

- All commits under the user's identity only — **never add Claude
  attribution/co-author trailers** (explicit user requirement).
- Ask before: deleting anything outside `/scratch/$USER/ditto_av/`,
  downloading >10 GB, or submitting jobs requesting >8 GPUs·h.
- Long-running work: always `sbatch`, never foreground SSH; poll `squeue`.
- If home quota errors appear (`Disk quota exceeded`), the culprit is almost
  always a cache writing to `$HOME/.cache` — redirect it to scratch
  (`XDG_CACHE_HOME=/scratch/$USER/ditto_av/cache`), don't delete user files.
