#!/bin/bash
# v0.3.2 220-route characterization launcher (run on a LOGIN node).
#   scripts/slurm/v032_bench220_submit.sh <diag_config> <run_label>
# Same chunking as v02_bench220_submit.sh (12 routes x 1 rep per job,
# spread over participation+visual, collector afterany) but runs the
# WORKTREE code via v032_carla_chain.sbatch. NON-gating: this
# characterizes the terminal arm at scale for the record/paper.
set -euo pipefail
CONF=${1:?diag config}
LABEL=${2:?run label}
REPO=/scratch/$USER/ditto_av/DITTO_AV_v032
B2D=/scratch/$USER/ditto_av/Bench2Drive
cd $REPO

IDS=$(grep -o 'route id="[0-9]*"' $B2D/leaderboard/data/bench2drive220.xml \
      | grep -o '[0-9]*')
readarray -t ID_ARR <<< "$IDS"
N=${#ID_ARR[@]}
echo "benchmark: $N routes, chunks of 12, config $CONF"

LANES=(participation visual)
JOBS=()
CH=0
for ((i = 0; i < N; i += 12)); do
  CHUNK=$(IFS=,; echo "${ID_ARR[*]:i:12}")
  TAG=$(printf "v032b220_%s_%02d" "$LABEL" "$CH")
  LANE=${LANES[$((CH % ${#LANES[@]}))]}
  GPUFLAG=""
  [ "$LANE" != "visual" ] && GPUFLAG="--gpus-per-task=1"
  JID=$(VARIANTS="$TAG:$CONF:$CHUNK:1" ROUTES_XML=bench2drive220.xml \
        sbatch --parsable --partition=$LANE $GPUFLAG \
        --job-name="v032b220-$CH" \
        scripts/slurm/v032_carla_chain.sbatch)
  echo "chunk $CH ($LANE): job $JID"
  JOBS+=("$JID")
  CH=$((CH + 1))
done

DEP=$(IFS=:; echo "${JOBS[*]}")
CJID=$(LABEL="$LABEL" sbatch --parsable --dependency=afterany:$DEP \
       scripts/slurm/v032_b220_collect.sbatch)
echo "collector: job $CJID (afterany ${#JOBS[@]} chunks)"
