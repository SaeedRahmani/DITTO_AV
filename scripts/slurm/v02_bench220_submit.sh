#!/bin/bash
# v0.2 220-route benchmark launcher (run on a LOGIN node).
#   scripts/slurm/v02_bench220_submit.sh <diag_config> <run_label>
# e.g. scripts/slurm/v02_bench220_submit.sh configs/diag_v02_999t_rl.yaml v02_999t_rl
# Splits bench2drive220.xml into 12-route chunks (1 rep), spreads them
# over the graphics lanes, and queues a collector (afterany on all
# chunks) that aggregates into runs/bench220_<label>/ and commits.
# v0.1 precedent: chunks of 12 fit the 4 h caps with margin.
set -euo pipefail
CONF=${1:?diag config}
LABEL=${2:?run label}
REPO=/scratch/$USER/ditto_av/DITTO_AV
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
  TAG=$(printf "v02b220_%s_%02d" "$LABEL" "$CH")
  LANE=${LANES[$((CH % ${#LANES[@]}))]}
  GPUFLAG=""
  [ "$LANE" != "visual" ] && GPUFLAG="--gpus-per-task=1"
  JID=$(VARIANTS="$TAG:$CONF:$CHUNK:1" ROUTES_XML=bench2drive220.xml \
        sbatch --parsable --partition=$LANE $GPUFLAG \
        --export=ALL,VARIANTS,ROUTES_XML \
        scripts/slurm/carla_eval_chain.sbatch)
  echo "chunk $CH ($LANE): job $JID"
  JOBS+=("$JID")
  CH=$((CH + 1))
done

DEP=$(IFS=:; echo "${JOBS[*]}")
sbatch --dependency=afterany:$DEP --partition=compute-p1 --time=00:20:00 \
  --ntasks=1 --cpus-per-task=1 --mem-per-cpu=2G \
  --account=research-ceg-tp \
  --output=/scratch/$USER/ditto_av/outputs/slurm-%j.out \
  --wrap="cd $REPO && /scratch/$USER/ditto_av/envs/carla_eval/bin/python \
scripts/collect_bench220.py v02b220_${LABEL} bench220_${LABEL} && \
echo '[$(date '+%F %H:%M')] v02 bench220 ${LABEL} collected' \
>> /scratch/$USER/ditto_av/outputs/PIPELINE_STATUS.md"
echo "submitted ${#JOBS[@]} chunks + collector for $LABEL"
