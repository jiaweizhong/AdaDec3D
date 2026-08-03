#!/usr/bin/env bash
# Dataset-axis cell: MSD Task01 BrainTumour, matched 3D UX-Net Full + EffiDec3D (concat).
# This script will wait for the specified PID (the currently running Task08 obs) to finish,
# and then automatically start the Task01 training.

TARGET_PID=285275

echo "=================================================="
echo "Waiting for Task 08 Observation (PID: $TARGET_PID) to finish..."
echo "=================================================="

# Loop as long as the process exists
while kill -0 $TARGET_PID 2>/dev/null; do
    echo "$(date +%H:%M:%S) - Still waiting for PID $TARGET_PID..."
    sleep 60
done

echo ""
echo "=================================================="
echo "$(date +%H:%M:%S) - PID $TARGET_PID has exited! Releasing GPU..."
echo "Starting Task 01 (BrainTumour) Training..."
echo "=================================================="

# Variables for Task 01
ROOT=${ROOT:-/root/autodl-tmp/Task01_BrainTumour}
OUT=${OUT:-/root/output}
DS=Task01_BrainTumour
CACHE=${CACHE:-0.25}
EVAL=${EVAL:-2500}
NW=${NW:-8}
GPU=${GPU:-0}

cd "$(dirname "$0")"  # cd into EffiDec3D/

log(){ echo "[brain $(date +%H:%M:%S)] $*"; }
run(){ "$@"; local rc=$?; [ $rc -eq 0 ] || { log "FAILED (rc=$rc): $*"; exit $rc; }; }

# 1. Train E0 (Full 3DUXNET)
log "training E0 (full 3D UX-Net, 4 modalities) ..."
run python main_train_MSD_Task01_10.py --root "$ROOT" --output "$OUT/E0_task01" \
  --dataset "$DS" --network 3DUXNET --n_channels 4 \
  --lr 0.001 --overlap 0.7 --crop_sample 4 --max_iter 45000 --eval_step "$EVAL" \
  --cache_rate "$CACHE" --num_workers "$NW" --gpu "$GPU"

# 2. Train E1 (EffiDec3D)
log "training E1 (EffiDec3D, concatenation, 4 modalities) ..."
run python main_train_MSD_Task01_10.py --root "$ROOT" --output "$OUT/E1_task01_concat" \
  --dataset "$DS" --network 3DUXNET_EffiDec3D --n_channels 4 --ds False --skip_aggregation concatenation \
  --lr 0.001 --overlap 0.7 --crop_sample 4 --max_iter 45000 --eval_step "$EVAL" \
  --cache_rate "$CACHE" --num_workers "$NW" --gpu "$GPU"

log "All Task 01 training completed successfully!"
