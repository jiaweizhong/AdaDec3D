#!/bin/bash
set -e

# ── Usage ────────────────────────────────────────────────────────────────────
#   bash run_E0_E1.sh          # E0 then E1 (default)
#   bash run_E0_E1.sh E0       # E0 only
#   bash run_E0_E1.sh E1       # E1 only  ← launch this after E0 finishes
#
# E1 auto-resumes from last_model.pth and saves milestone_{05000..45000}.pth
# (required for O6). Make sure `git pull` has pulled the milestone code first.
# ─────────────────────────────────────────────────────────────────────────────

STAGE="${1:-all}"

ROOT=/root/autodl-tmp/btcv-synapse
OUT=/root/output
LOG=$OUT/run_E0_E1.log
mkdir -p $OUT

cd /root/AdaDec3D
git pull
cd EffiDec3D

# Hyperparameters aligned with EffiDec3D paper (README).
# NOTE: original Orientationd/Spacingd pipeline kept — do NOT add
# --skip_spatial_resampling (removed to stay aligned with the reference codebase).
COMMON="--root $ROOT --dataset BTCV13 \
        --cache_rate 1.0 --num_workers 8 --gpu 0 \
        --lr 0.001 --overlap 0.7 --crop_sample 4"
TRAIN_ARGS="--max_iter 45000 --eval_step 250"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

# ── E0: Full 3DUXNET (upper bound) ───────────────────────────────────────────
run_E0() {
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
}

# ── E1: EffiDec3D baseline (Paper A) ─────────────────────────────────────────
run_E1() {
    log "E1 starting — 3DUXNET_EffiDec3D (auto-resumes; saves O6 milestones)"
    python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS \
        --output $OUT/E1 --network 3DUXNET_EffiDec3D \
        --ds False 2>&1 | tee -a $LOG
    log "E1 done"
}

case "$STAGE" in
    E0)  run_E0 ;;
    E1)  run_E1 ;;
    all) run_E0; run_E1 ;;
    *)   log "Unknown stage '$STAGE' (use: E0 | E1 | all)"; exit 1 ;;
esac

log "Finished stage=$STAGE. Results in $OUT  |  Metrics CSV: last_validation_metrics_btcv13.csv"
