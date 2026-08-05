# DITTO_AV v0.3.2 — plan

Branch `saeed/v0.3.2`, cut from frozen `saeed/v0.3` on 2026-08-05.
Workspace: `/scratch/$USER/ditto_av/DITTO_AV_v032` (git worktree).
Runs in parallel with v0.3.1, which continues on `main` in the
DITTO_AV checkout — never edit or build on main from here.

## 1. Mission

TBD — improve v0.3 along a different axis than v0.3.1 (Saeed defines
the direction). Rules carry over from V03_PLAN §3: one axis per
iteration, pre-registered gates, every result -> ledger with job id.

## 2. Ops deltas vs V03_PLAN §7 (everything else inherits)

- This worktree replaces the removed `DITTO_AV_v03` one; ignore stale
  references to that path in older docs on this branch.
- `v03_d3` / `v03_w0` / `v03_data_cache` sbatch scripts are repointed
  to `DITTO_AV_v032`. Older v0.1/v0.2-era sbatch scripts still target
  main's checkout (`ditto_av/DITTO_AV`) — repoint before reusing any
  of them from here, or the job will run v0.3.1 code from main.
- Shared with main, coordinate instead of colliding: `../outputs`
  (CLAIM stages in `outputs/PIPELINE_STATUS.md` with tag "V03.2:",
  read the tail first), `../data`, `../envs`, CARLA sif + overlay.
  Never cancel jobs you didn't submit; main's v0.3.1 runs have right
  of way on lanes.
- Name v0.3.2 run outputs distinctly (`v032_*`) so collectors that
  glob `outputs/` don't mix versions.
- Worktree lifecycle (same as DITTO_AV_v03): when v0.3.2 merges or is
  abandoned, remove the worktree, keep the branch.
- Commits as Saeed Rahmani, no AI attribution. Scratch has a 1M-inode
  quota; extractions go to node-local /tmp.
