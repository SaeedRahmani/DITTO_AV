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

### THE FILE-COUNT QUOTA — the #1 operational hazard (learned 2026-07-28)

/scratch has TWO per-user quotas: **5 TiB of bytes AND 1,000,000 chunk
files (inodes)**. The file-count limit is **FIXED — DHPC will not raise
it** (confirmed). Exceeding it makes EVERY write fail with "Disk quota
exceeded" even with terabytes free; this looked like a cluster outage
for two days (2026-07-27/28) and it was not. Facts:

- Authoritative live view (refreshed each logon):
  `bash /etc/profile.d/ZZ_motd-info.sh` — read the **chunk files**
  column. `beegfs-ctl --getquota --gid ...` "1 Byte" group rows are a
  RED HERRING; per-uid accounting can also read stale/zero. Trust the
  motd table.
- Known costs: a conda env or venv = 30–100k files; an extracted
  Bench2Drive anno set = ~60k; git clones = 1–5k each; caches
  (pip/conda/apptainer layers) = tens of thousands.

Mandatory practices for every AI session:
1. Check the motd chunk-files number at session start AND before/after
   any extraction, env creation, or big download. Keep **≥100k
   headroom**.
2. Never leave both an archive's extraction and other bulk trees
   growing at once; extract → consume (build npz/pack) → delete the
   extraction (tarballs stay, they're 1 file each).
3. Prefer packed formats everywhere (npz, tar, zip — python reads zip
   directly). Never generate per-frame/per-sample small files.
4. Reuse existing envs; never create a new env when one fits.
5. Periodically: `du --inodes -d1 /scratch/$USER` to see who's eating
   the budget. Delete caches (pip/conda/apptainer) freely — they
   rebuild.
6. If writes fail with quota errors and bytes are fine: it IS the file
   count. Free ~10k files (caches, __pycache__, synced wandb dirs) and
   writes resume instantly.

### Official DHPC I/O guidance we adopt
(<https://doc.dhpc.tudelft.nl/delftblue/Optimize-Your-IO/>, section iii)

- **Many-small-file work belongs on node-local `/tmp`, not scratch.**
  In any job that unpacks/parses many files: copy the tarball to
  `/tmp/${SLURM_JOB_ID}/`, extract + process THERE (node-local SSD:
  fast random I/O, zero scratch inodes, no BeeGFS metadata storms),
  write only packed results (npz/tar) back to scratch, and `rm -rf
  /tmp/${SLURM_JOB_ID}` at script end (a trap on EXIT). /tmp is shared
  with other jobs on the node — always clean up.
  This applies to validate_b2d-style extraction and any future data
  prep; it would have prevented both 2026-07 storage incidents.
- **At larger data scale** (e.g. the full 1000-clip split), prefer
  sharded packed loaders (WebDataset/DataDings-style tar shards) over
  loose files; batches over whole-dataset passes.


| Location | Path | Properties | Use for |
|---|---|---|---|
| Home | `/home/srahmani` | ~30 GB quota, small, was nearly full (cleaned to ~7.6 GB on 2026-07-26 by purging pip/uv caches) | dotfiles + the freeze-fallback clone/outputs (see the dual-clone section). **Install nothing here.** |
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

/home/srahmani/            (freeze fallback only — see next section)
  ditto_work/DITTO_AV/     second git clone, origin = github
  ditto_out/               job outputs while scratch is write-frozen
```

## Scratch write-freezes: the dual-clone workflow

Twice so far (2026-07-27, again 2026-07-28) DHPC set 1-byte hard group
quotas on the pool backing /scratch: every WRITE fails ("Disk quota
exceeded") while reads/execution keep working. Admin-side incident —
nothing on our side causes or can prevent it. Probe at session start:

```sh
echo t > /scratch/$USER/ditto_av/outputs/t && rm $_ || echo FROZEN
```

Rules while frozen (and for keeping the two clones sane in general):

- **GitHub is the only sync channel between the clones.** Never copy
  files between them; commit + push from the clone you edited, pull in
  the other before using it. Both clones commit as the user (identity is
  set repo-locally in each).
- **Normal times: the scratch clone is the working copy** (fast, big
  data next to it). The home clone may lag — pull it when needed.
- **While frozen: work only in `~/ditto_work/DITTO_AV`** (the scratch
  clone can't even commit). Job outputs go to `~/ditto_out/` — home is
  ~30 GB, so small outputs only (checkpoints/results/logs, never
  datasets). Scratch stays usable read-only: venv, data, CARLA SIF,
  Bench2Drive clone.
- `scripts/slurm/phase2_home.sbatch` = freeze-safe Phase-2 job: cwd is
  the home clone, reads data/venv from scratch, writes run_dir, wandb
  and the SLURM log to $HOME.
- The pipeline tolerates a mid-run freeze: npz-cache writes in
  `run_b2d.py` are best-effort (skipped with a warning, truncated
  entries removed).
- **At unfreeze, in order:** (1) `git -C /scratch/$USER/ditto_av/DITTO_AV
  pull` BEFORE any job that runs from the scratch clone (closed-loop
  eval does); (2) commit small results from `~/ditto_out/` into `runs/`;
  (3) resume the normal scratch layout. Check NEXT_STEPS.md for
  anything else the freeze interrupted.

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
`gpu-v100`, `gpu-a100-small` (MIG slices — fine for our small models),
plus the probe-verified extra lanes below (2026-07-29). Walltime caps:
48 h on gpu-a100/gpu-v100, 4 h on gpu-a100-small/participation/visual,
120 h on compute.
Templates live in `scripts/slurm/`. The user's SLURM accounts (verified
2026-07-26): `research-ceg-tp` (used in the templates) and `innovation`.

**GPU lane truth table (ALL verified by real submissions 2026-07-29 —
trust nothing else; re-probe with tiny --wrap jobs when in doubt):**

| lane | GPU | verdict |
|---|---|---|
| gpu-a100 / gpu-v100 (ceg-tp) | A100 / V100 | works, but can saturate for >14 h with *hidden* long jobs (`PrivateData` hides other users in squeue — check node `AllocTRES`, not the queue) |
| `participation` (ceg-tp) | H100 NVL | **instant start**, CARLA_BOOT_OK (Town03+Town12), 4 h cap, `--gpus-per-task=1` |
| `visual` (ceg-tp) | Quadro RTX 4000 | **instant start**, CARLA_BOOT_OK, 4 h cap, GPU is unmanaged — NO `--gpus-per-task` flag |
| gpu-a100-small (BOTH accounts) | A100 MIG 1g.10gb | instant start; CUDA training only — MIG has NO graphics, CARLA cannot boot |
| innovation → any full-GPU partition | — | **REJECTED** (`AssocMaxGRESPerJob`, even 1 GPU); the old "valid fallback" note here was wrong — verified only on paper, never by submission |
| CPU + `-nullrhi` shim knob | none | CARLA server never binds the RPC port — dead end, do not retry |
| as-rst-*, participants, participation-gpu | H100 | submission rejected: "Invalid account/partition combination" |

**MANDATORY pre-run GPU audit (user instruction 2026-07-29 — do this
EVERY time a run is needed, no exceptions):**

1. Before submitting ANY job, audit ALL lanes across BOTH accounts
   (`research-ceg-tp` AND `innovation`): `sinfo -a` for partition/node
   states, `scontrol show node` AllocTRES on candidate GPU nodes
   (squeue LIES — PrivateData hides other users' jobs), and when in
   doubt fire 5-min `--wrap="hostname; nvidia-smi -L"` probes at every
   plausible lane x account combo. Probes cost nothing; assumptions
   cost nights.
2. Pick the FASTEST-STARTING lane whose hardware fits the run's need
   (CARLA eval needs a graphics-capable GPU: A100/V100/H100/Quadro all
   verified; training needs CUDA: MIG slices fine; never downgrade the
   experiment itself to fit a lane — no smaller models, fewer reps, or
   CPU fallbacks to dodge a queue).
3. **Any wait measured in hours means WE made a mistake** — the user is
   certain a GPU is reachable within a very short time on one of the
   two accounts. If a job pends >30 min: stop, re-audit everything,
   move the work. Never let jobs sit overnight in a queue.
4. Re-verify this table each session (limits/permissions change; the
   innovation full-GPU rejection deserves a periodic re-probe, and if
   it persists ask the user to raise it with DHPC).

**Disconnect-resilient orchestration (learned 2026-07-29 the hard
way):** anything watching or sequencing jobs from inside an interactive
AI session DIES when the user's SSH drops — the user then sees "no
checks happened". Multi-stage pipelines MUST be self-driving on the
cluster: chain stages with `sbatch --dependency=afterany:...` decider
jobs (scripts/pipeline_decider.py + scripts/slurm/decider.sbatch) that
aggregate results, append to `outputs/PIPELINE_STATUS.md` (the
user-readable truth — check it at every session start), commit results
into the repo, and submit the next stage. In-session monitors are a
convenience layer ONLY, never the mechanism of record.

Python environments — which one where (important):

| Env | Type | Torch | Works on | Use for |
|---|---|---|---|---|
| `/scratch/.../envs/ditto` | venv on module python | 2.13.0+cu130 | login + CPU nodes only (GPU nodes don't mount the module/spack tree) | CPU jobs, tests, tooling |
| `/scratch/.../envs/carla_eval` | conda, self-contained | 2.13.0+**cpu** | all nodes | closed-loop CARLA eval (drives the agent; no CUDA training) |
| `~/envs/ditto_gpu` | conda, self-contained (built 2026-07-28) | 2.13.0+cu130 | all nodes; CUDA only on **A100** (cu130 wheels dropped V100/sm_70) | GPU training (`phase2_gpu.sbatch`) |

New packages can be installed from login nodes (outbound internet there)
with conda/pip/uv; compute nodes have no internet. Self-contained conda
envs are the only kind that run on GPU nodes.

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
3. **Phase-1 sweep** — DONE 2026-07-26 (21 runs, all committed to
   `runs/phase1/`): 3-seed main + K/negatives/horizon/data/style ablations
   + 3-seed validation of the improved config (K=16, H=5: DITTO-multi
   near-expert, 0.10 collisions ID / 0.21 shifted) + 3-seed
   trajectory-consistent `multi_traj` ablation (== multi; rules out
   per-step reward relaxation). Headline tables in PAPER_PLAN.md.
4. **Phase-2 (Bench2Drive)** — offline pipeline DONE and run twice:
   v1 (60 clips, job 10521595) and v2 (297 clips, job 10521834; results in
   `runs/b2d_v2/`, WM H-step MSE 0.093, policies at the expert-replay
   latent ceiling). Open-loop only — NOT driving evidence (see PAPER_PLAN
   framing). Closed-loop prerequisites: route/command conditioning +
   CARLA agent adapter + CARLA 0.9.15 setup (module list / apptainer) —
   all still TODO. Observation is privileged (GT boxes): position results
   accordingly.
5. **W&B monitoring** (added 2026-07-26): jobs log offline
   (`WANDB_MODE=offline`, set in templates); keep
   `scripts/wandb_sync.sh` running via nohup on a login node to stream to
   wandb.ai project `ditto-av` (~2 min lag). Restart it after login-node
   reboots.

## Conventions for the AI working on DelftBlue

- All commits under the user's identity only — **never add Claude
  attribution/co-author trailers** (explicit user requirement).
- **Compute needs no approval** (user instruction, 2026-07-28). Submit
  large and long runs whenever the science needs them — the full
  220-route eval (~36 GPU·h), multi-day trainings, big downloads, all
  of it. Do NOT pause to ask. `research-ceg-tp` allows 48 h walltime,
  8 GPUs/job and 64 concurrent jobs; parallelise across jobs rather
  than trimming an experiment to look cheap. The only compute-side
  duty is to spend it well: run an expensive eval once, on the config
  the cheap experiments already selected.
- **Ask before REMOVING**: deleting or overwriting files, datasets,
  checkpoints or any large/important data — anywhere, including inside
  `/scratch/$USER/ditto_av/`. This is the ONLY standing approval gate.
  Exception (pre-approved, they rebuild): pip/conda/apptainer caches,
  `__pycache__`, already-synced wandb dirs, and extraction dirs whose
  packed output is confirmed written.
- Long-running work: always `sbatch`, never foreground SSH; poll `squeue`.
- If home quota errors appear (`Disk quota exceeded`), the culprit is almost
  always a cache writing to `$HOME/.cache` — redirect it to scratch
  (`XDG_CACHE_HOME=/scratch/$USER/ditto_av/cache`), don't delete user files.
