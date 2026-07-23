#!/bin/bash
set -e

ROOT=/root/autodl-tmp/btcv-synapse
OUT=/root/output
LOG=$OUT/run_E0_E1.log
mkdir -p $OUT

cd /root/AdaDec3D
git pull
cd EffiDec3D

COMMON="--root $ROOT --dataset BTCV13 --cache_rate 1.0 --num_workers 8 --gpu 0"
TRAIN_ARGS="--max_iter 20000 --eval_step 500"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

# ── E0: Full 3DUXNET (upper bound) ───────────────────────────────────────────
E0_CKPT=$(ls $OUT/E0*/3DUXNET/BTCV13/best_metric_model.pth 2>/dev/null | head -1)

if [ -n "$E0_CKPT" ]; then
    log "E0 checkpoint found: $E0_CKPT — running test/inference only"
    python main_train_BTCV_TU.py $COMMON \
        --output $OUT/E0 --network 3DUXNET \
        --mode test 2>&1 | tee -a $LOG
else
    log "E0 checkpoint not found — training from scratch"
    python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS \
        --output $OUT/E0 --network 3DUXNET 2>&1 | tee -a $LOG
fi
log "E0 done"

# ── E1: EffiDec3D baseline (Paper A) ─────────────────────────────────────────
log "E1 starting — 3DUXNET_EffiDec3D"
python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS \
    --output $OUT/E1 --network 3DUXNET_EffiDec3D \
    --ds False 2>&1 | tee -a $LOG
log "E1 done"

log "All finished. Results in $OUT  |  Metrics CSV: last_validation_metrics_btcv.csv"
