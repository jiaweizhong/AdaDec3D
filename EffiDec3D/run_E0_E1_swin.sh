#!/bin/bash
set -eo pipefail   # pipefail so a python crash through `| tee` still aborts the script

# ── Usage ────────────────────────────────────────────────────────────────────
#   bash run_E0_E1_swin.sh          # E0 then E1, then observations (default)
#   bash run_E0_E1_swin.sh E0       # E0 only
#   bash run_E0_E1_swin.sh E1       # E1 only, then observations  ← after E0 finishes
#   bash run_E0_E1_swin.sh obs      # observations only (needs E0+E1 checkpoints)
#
# Second MATCHED backbone (SwinUNETR) for the O8 architecture-family axis, on the
# SAME BTCV13 protocol as 3D UX-Net (run_E0_E1.sh). One Full + one EffiDec3D seed.
# E1 auto-resumes from last_model.pth and saves milestone_{05000..45000}.pth (O6).
# Standard Orientationd/Spacingd pipeline — do NOT add --skip_spatial_resampling.
# Run `git pull` once before launching so this + the obs scripts are current.
# ─────────────────────────────────────────────────────────────────────────────

STAGE="${1:-all}"

ROOT=/root/autodl-tmp/btcv-synapse
OUT=/root/output
OBS=/root/obs-swin
LOG=$OUT/run_E0_E1_swin.log
mkdir -p $OUT

cd /root/AdaDec3D
git pull
cd EffiDec3D

# Hyperparameters aligned with EffiDec3D paper; feature_size 48 (SwinUNETR default).
COMMON="--root $ROOT --dataset BTCV13 \
        --cache_rate 1.0 --num_workers 8 --gpu 0 \
        --lr 0.001 --overlap 0.7 --crop_sample 4 --feature_size 48"
TRAIN_ARGS="--max_iter 45000 --eval_step 250"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

# ── E0: Full SwinUNETR (upper bound) ─────────────────────────────────────────
run_E0() {
    E0_CKPT=$(ls $OUT/E0_swin*/SwinUNETR/BTCV13/best_metric_model.pth 2>/dev/null | head -1)
    if [ -n "$E0_CKPT" ]; then
        log "E0 checkpoint found: $E0_CKPT — running test/inference only"
        python main_train_BTCV_TU.py $COMMON \
            --output $OUT/E0_swin --network SwinUNETR \
            --mode test 2>&1 | tee -a $LOG
    else
        log "E0 checkpoint not found — training SwinUNETR from scratch"
        python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS \
            --output $OUT/E0_swin --network SwinUNETR 2>&1 | tee -a $LOG
    fi
    log "E0 (SwinUNETR) done"
}

# ── E1: SwinUNETR + EffiDec3D (Paper A) ──────────────────────────────────────
run_E1() {
    log "E1 starting — SwinUNETR_EffiDec3D (auto-resumes; saves O6 milestones)"
    python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS \
        --output $OUT/E1_swin --network SwinUNETR_EffiDec3D \
        --ds False 2>&1 | tee -a $LOG
    log "E1 (SwinUNETR_EffiDec3D) done"
}

# ── Observations: O1–O11 for the matched Swin pair ───────────────────────────
run_obs() {
    E0_CKPT=$(ls $OUT/E0_swin*/SwinUNETR/BTCV13/best_metric_model.pth 2>/dev/null | head -1)
    E1_CKPT=$(ls $OUT/E1_swin*/SwinUNETR_EffiDec3D/BTCV13/best_metric_model.pth 2>/dev/null | head -1)
    if [ -z "$E0_CKPT" ] || [ -z "$E1_CKPT" ]; then
        log "Observations skipped — need both E0 and E1 (E0='$E0_CKPT' E1='$E1_CKPT')"
        return
    fi
    log "Observations: O1–O11 gate for SwinUNETR (run_observations.py --network SwinUNETR)"
    python run_observations.py \
        --root $ROOT --output $OUT --network SwinUNETR --dataset BTCV13 \
        --e0_ckpt "$E0_CKPT" --e1_ckpt "$E1_CKPT" --obs_dir $OBS 2>&1 | tee -a $LOG
    log "Observations done — figures + results.json in $OBS"
}

case "$STAGE" in
    E0)  run_E0 ;;
    E1)  run_E1; run_obs ;;
    all) run_E0; run_E1; run_obs ;;
    obs) run_obs ;;
    *)   log "Unknown stage '$STAGE' (use: E0 | E1 | all | obs)"; exit 1 ;;
esac

log "Finished stage=$STAGE. Obs: $OBS/results.json"
