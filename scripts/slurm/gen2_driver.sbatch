#!/bin/bash
#SBATCH --job-name=gen2-driver
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:55:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3900M
#SBATCH --output=/home/%u/ditto_out/slurm-%j.out
#SBATCH --account=research-ceg-tp
# Gen-2 tail driver — NO Slurm dependencies (the scheduler cancelled
# dependents of RUNNING parents twice, 2026-07-30; orchestrate inside
# one allocation instead): wait for the present-manifest scan, verify
# the npz cache key matches EXACTLY, train inline (this job's GPU),
# then submit the closed-loop eval directly.
set -euo pipefail
REPO=/scratch/$USER/ditto_av/DITTO_AV
MF=$REPO/manifests/b2d_full_present.txt
CACHE=/scratch/$USER/ditto_av/data/bench2drive/npz_cache

# 1. wait for scan v2 (marker: FIND_ANNO_DONE newer than this job start)
for i in $(seq 1 60); do
  [ -f $MF.v2done ] && break
  sleep 30
done
[ -f $MF.v2done ] || { echo "scan v2 marker never appeared"; exit 1; }

# 2. recompute the cache key from the manifest, trust-semantics; abort
#    unless the cache built by the extraction job matches it exactly
KEY=$(python3 - "$MF" <<'PYEOF'
import hashlib, sys
names = sorted(l.strip().removesuffix(".tar.gz")
               for l in open(sys.argv[1]) if l.strip())
tr = [n for i, n in enumerate(names) if i % 6 != 5]
va = [n for i, n in enumerate(names) if i % 6 == 5]
print(hashlib.md5(("|".join(tr) + "##" + "|".join(va)
                   + "##routeTrue").encode()).hexdigest()[:12])
PYEOF
)
echo "computed key: $KEY"
[ -f $CACHE/${KEY}_train.npz ] && [ -f $CACHE/${KEY}_val.npz ] || {
  echo "CACHE KEY MISMATCH: $CACHE/${KEY}_*.npz absent — do not train"; exit 1; }

# 3. train inline on this allocation's GPU
export MANIFEST=manifests/b2d_full_present.txt TRUST=1
bash $REPO/scripts/slurm/phase2_gpu.sbatch configs/b2d_gen2.yaml gen2

# 4. submit the closed-loop eval directly (no dependency)
cd $REPO
VARIANTS="gen2_3x3:configs/diag_gen2.yaml:25381,25378,27494:3" \
  sbatch -p visual --cpus-per-task=17 --job-name=gen2ev \
  scripts/slurm/carla_eval_chain.sbatch
echo "GEN2_DRIVER_DONE"
