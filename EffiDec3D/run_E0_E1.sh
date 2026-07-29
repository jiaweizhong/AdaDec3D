#!/bin/bash
set -e

# ── Usage ────────────────────────────────────────────────────────────────────
#   bash run_E0_E1.sh          # E0 then E1, then observations (default)
#   bash run_E0_E1.sh E0       # E0 only
#   bash run_E0_E1.sh E1       # E1 only, then observations  ← after E0 finishes
#   bash run_E0_E1.sh obs      # observations only (needs E0+E1 checkpoints)
#
# E1 auto-resumes from last_model.pth and saves milestone_{05000..45000}.pth (O6).
# After E1, observations run automatically (MAC profile + O1–O11 + patch difficulty)
# using the existing E0+E1 checkpoints — one command gives training + all results.
# Run `git pull` once before launching so this script + the obs scripts are current.
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
# --skip_spatial_resampling (that is a separate identity-affine control).
COMMON="--root $ROOT --dataset BTCV13 \
        --cache_rate 1.0 --num_workers 8 --gpu 0 \
        --lr 0.001 --overlap 0.7 --crop_sample 4"
TRAIN_ARGS="--max_iter 45000 --eval_step 250"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

# ── E0: Full 3DUXNET (upper bound) ───────────────────────────────────────────
run_E0() {
    LM=$(ls $OUT/E0*/3DUXNET/BTCV13/last_model.pth 2>/dev/null | head -1 || true)
    if [ -n "$LM" ]; then log "E0 resuming from $LM"; else log "E0 training from scratch"; fi
    # main_train auto-resumes from last_model.pth and skips the loop if already at max_iter.
    python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS \
        --output $OUT/E0 --network 3DUXNET 2>&1 | tee -a $LOG
    log "E0 done"
}

# ── E1: EffiDec3D baseline (Paper A) ─────────────────────────────────────────
run_E1() {
    log "E1 starting — 3DUXNET_EffiDec3D (auto-resumes; saves O6 milestones)"
    python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS \
        --output $OUT/E1_concat --network 3DUXNET_EffiDec3D \
        --skip_aggregation concatenation \
        --ds False 2>&1 | tee -a $LOG
    log "E1 done"
}

# ── Observations: MAC profile + O1–O11 + patch difficulty (needs E0 + E1) ────
run_obs() {
    E0_CKPT=$(ls $OUT/E0*/3DUXNET/BTCV13/best_metric_model.pth 2>/dev/null | head -1)
    E1_CKPT=$(ls $OUT/E1_concat*/3DUXNET_EffiDec3D/BTCV13/best_metric_model.pth 2>/dev/null | head -1)
    if [ -z "$E0_CKPT" ] || [ -z "$E1_CKPT" ]; then
        log "Observations skipped — need both E0 and E1 (E0='$E0_CKPT' E1='$E1_CKPT')"
        return
    fi
    log "Observations: EffiDec3D MAC profile (encoder vs decoder)"
    python profile_macs.py 2>&1 | tee -a $LOG
    log "Observations: O1–O11 gate (run_observations.py) — concat E1"
    python run_observations.py --root $ROOT --output $OUT \
        --network 3DUXNET --dataset BTCV13 \
        --e0_ckpt "$E0_CKPT" --e1_ckpt "$E1_CKPT" \
        --skip_aggregation concatenation \
        --obs_dir /root/obs 2>&1 | tee -a $LOG
    log "Observations: patch difficulty (encoder-adaptivity headroom)"
    python patch_difficulty.py --root $ROOT --with-model 2>&1 | tee -a $LOG
    log "Observations done — figures + results.json in /root/obs"
}

case "$STAGE" in
    E0)  run_E0 ;;
    E1)  run_E1; run_obs ;;
    all) run_E0; run_E1; run_obs ;;
    obs) run_obs ;;
    *)   log "Unknown stage '$STAGE' (use: E0 | E1 | all | obs)"; exit 1 ;;
esac

log "Finished stage=$STAGE. Baselines CSV: last_validation_metrics_btcv13.csv | Obs: /root/obs/results.json"
