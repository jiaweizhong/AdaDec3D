#!/bin/bash
set -eo pipefail   # pipefail so a python crash through `| tee` still aborts

# ── Frozen shared-encoder factor decomposition (Plan E1) ──────────────────────
# Reuse the trained Full-UX (E0) as a FROZEN encoder, then train four decoder-only
# corners on identical features to isolate the two EffiDec3D knobs:
#   Full corner       rf=1, nchan=C_FULL   (highest-res stage kept, wide channels)
#   Channel-only      rf=1, nchan=48       (wide->narrow channels, full resolution)
#   Resolution-only   rf=2, nchan=C_FULL   (drop high-res stage, wide channels)
#   Effi (combined)   rf=2, nchan=48       (EffiDec3D default)
# Then run pairwise flip analysis: Full->Channel-only (channel effect),
# Full->Resolution-only (resolution effect), Full->Effi (combined).
#
#   bash run_frozen_factorial.sh            # train all corners + analyze
#   bash run_frozen_factorial.sh analyze    # analysis only (corners already trained)
# ─────────────────────────────────────────────────────────────────────────────

STAGE="${1:-all}"
ROOT=/root/autodl-tmp/btcv-synapse
OUT=/root/output
OBS=/root/obs-fz
C_FULL=384          # wide-channel corner; lower (e.g. 192) if it OOMs. Calibrate to
                    # approximate the full decoder's capacity.
LOG=$OUT/run_frozen_factorial.log
mkdir -p "$OUT" "$OBS"

cd /root/AdaDec3D && git pull && cd EffiDec3D

COMMON="--root $ROOT --dataset BTCV13 --channels 48 96 192 384 \
        --cache_rate 1.0 --num_workers 8 --gpu 0 \
        --lr 0.001 --overlap 0.7 --crop_sample 4"
TRAIN="--max_iter 45000 --eval_step 250 --ds False"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Frozen encoder source = trained Full-UX (E0).
E0=$(ls $OUT/E0_network_3DUXNET_fc48*/3DUXNET/BTCV13/best_metric_model.pth 2>/dev/null | head -1 || true)

train_corner() {   # $1=tag  $2=rf  $3=nchan
    if [ -z "$E0" ]; then log "ERROR: no Full-UX E0 encoder checkpoint found"; exit 1; fi
    log "corner $1: rf=$2 nchan=$3 (frozen encoder $E0)"
    python main_train_BTCV_TU.py $COMMON $TRAIN \
        --output "$OUT/FZ_$1" --network 3DUXNET_EffiDec3D \
        --resolution_factor "$2" --n_decoder_channels "$3" \
        --freeze_encoder --encoder_ckpt "$E0" 2>&1 | tee -a "$LOG"
}

ckpt() {   # $1=tag -> echo checkpoint path
    ls $OUT/FZ_$1*/3DUXNET_EffiDec3D/BTCV13/best_metric_model.pth 2>/dev/null | head -1 || true
}

if [ "$STAGE" = "all" ]; then
    train_corner full "1" "$C_FULL"
    train_corner chan "1" "48"
    train_corner res  "2" "$C_FULL"
    train_corner effi "2" "48"
fi

FULL=$(ckpt full); CHAN=$(ckpt chan); RES=$(ckpt res); EFFI=$(ckpt effi)
log "checkpoints: full=$FULL chan=$CHAN res=$RES effi=$EFFI"

analyze() {   # $1=pairname  $2=e1_ckpt  $3=e1_rf  $4=e1_nchan
    log "analyze Full -> $1"
    python run_observations.py --network 3DUXNET --dataset BTCV13 \
        --root $ROOT --output $OUT \
        --e0_ckpt "$FULL" --e0_rf 1 --e0_nchan "$C_FULL" \
        --e1_ckpt "$2" --e1_rf "$3" --e1_nchan "$4" \
        --obs_dir "$OBS/$1" 2>&1 | tee -a "$LOG"
}

analyze channel_effect    "$CHAN" 1 48        # Full -> Channel-only
analyze resolution_effect "$RES"  2 "$C_FULL" # Full -> Resolution-only
analyze combined_effi     "$EFFI" 2 48        # Full -> Effi

log "Done. Per-pair results.json + figures under $OBS/{channel_effect,resolution_effect,combined_effi}"
