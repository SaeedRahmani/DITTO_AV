#!/bin/bash
# Sync offline W&B runs to wandb.ai from a login node (compute nodes have no
# internet; jobs log with WANDB_MODE=offline into outputs/wandb/).
# Usage (detached):
#   nohup setsid bash scripts/wandb_sync.sh \
#       > /scratch/$USER/ditto_av/outputs/wandb_sync.log 2>&1 &
# Stop with: pkill -f wandb_sync.sh
source /scratch/$USER/ditto_av/envs/ditto/bin/activate
DIR=/scratch/$USER/ditto_av/outputs/wandb
while true; do
    for d in "$DIR"/offline-run-*; do
        [ -d "$d" ] && wandb sync "$d" 2>&1 | grep -v "already synced"
    done
    sleep 120
done
