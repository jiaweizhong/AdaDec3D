#!/bin/bash
set -eo pipefail   # pipefail so a python crash through `| tee` still aborts

# ── Third matched backbone: MedNeXt-M-K3 (BTCV) for the architecture axis ──────
#   bash run_E0_E1_mednext.sh          # E0 then E1, then observations (default)
#   bash run_E0_E1_mednext.sh E1       # E1 only, then observations
#   bash run_E0_E1_mednext.sh obs      # observations only (needs E0+E1 checkpoints)
#
# Both full (MedNeXt_M) and efficient (MedNeXt_M_EffiDec3D) use kernel size 3, so the
# matched pair differs only in the decoder (channels + resolution) — matching the
# paper's MedNeXt-M-K3 and run_observations' K3 build_model. One Full + one Effi seed.
# Standard Orientationd/Spacingd pipeline. Run `git pull` first (done below).
# ─────────────────────────────────────────────────────────────────────────────

STAGE="${1:-all}"
ROOT=/root/autodl-tmp/btcv-synapse
OUT=/root/output
OBS=/root/obs-mednext
LOG=$OUT/run_E0_E1_mednext.log
mkdir -p "$OUT" "$OBS"

cd /root/AdaDec3D && git pull && cd EffiDec3D

# feature_size 48 (MedNeXt-M default n_channels).
COMMON="--root $ROOT --dataset BTCV13 \
        --cache_rate 1.0 --num_workers 8 --gpu 0 \
        --lr 0.001 --overlap 0.7 --crop_sample 4 --feature_size 48"
TRAIN_ARGS="--max_iter 45000 --eval_step 250"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── E0: full MedNeXt-M-K3 (upper bound) ──────────────────────────────────────
run_E0() {
    CK=$(ls $OUT/E0_mednext*/MedNeXt_M/BTCV13/best_metric_model.pth 2>/dev/null | head -1 || true)
    if [ -n "$CK" ]; then
        log "E0 checkpoint found: $CK — inference only"
        python main_train_BTCV_TU.py $COMMON --output $OUT/E0_mednext \
            --network MedNeXt_M --mode test 2>&1 | tee -a "$LOG"
    else
        log "E0 not found — training full MedNeXt-M-K3 from scratch"
        python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS --output $OUT/E0_mednext \
            --network MedNeXt_M 2>&1 | tee -a "$LOG"
    fi
    log "E0 (MedNeXt_M) done"
}

# ── E1: MedNeXt-M-K3 + EffiDec3D ─────────────────────────────────────────────
run_E1() {
    log "E1 starting — MedNeXt_M_EffiDec3D (auto-resumes; saves O6 milestones)"
    python main_train_BTCV_TU.py $COMMON $TRAIN_ARGS --output $OUT/E1_mednext \
        --network MedNeXt_M_EffiDec3D --ds False 2>&1 | tee -a "$LOG"
    log "E1 (MedNeXt_M_EffiDec3D) done"
}

# ── Observations: all O1–O11 + O_pareto/O_anatomy/boundary/surface + teaser ───
run_obs() {
    E0=$(ls $OUT/E0_mednext*/MedNeXt_M/BTCV13/best_metric_model.pth 2>/dev/null | head -1 || true)
    E1=$(ls $OUT/E1_mednext*/MedNeXt_M_EffiDec3D/BTCV13/best_metric_model.pth 2>/dev/null | head -1 || true)
    if [ -z "$E0" ] || [ -z "$E1" ]; then
        log "Observations skipped — need both E0 and E1 (E0='$E0' E1='$E1')"
        return
    fi
    log "Observations: O1–O11 for MedNeXt (run_observations.py --network MedNeXt)"
    python run_observations.py --network MedNeXt --dataset BTCV13 \
        --root $ROOT --output $OUT --e0_ckpt "$E0" --e1_ckpt "$E1" --obs_dir $OBS 2>&1 | tee -a "$LOG"
    log "Figure 1 teaser"
    python make_figure1.py --network MedNeXt --dataset BTCV13 \
        --root $ROOT --output $OUT --e0_ckpt "$E0" --e1_ckpt "$E1" \
        --out $OBS/figure1_motivation.png 2>&1 | tee -a "$LOG"
    log "Observations done — results.json + figures in $OBS"
}

case "$STAGE" in
    E0)  run_E0 ;;
    E1)  run_E1; run_obs ;;
    all) run_E0; run_E1; run_obs ;;
    obs) run_obs ;;
    *)   log "Unknown stage '$STAGE' (use: E0 | E1 | all | obs)"; exit 1 ;;
esac

log "Finished stage=$STAGE. Obs: $OBS/results.json"
