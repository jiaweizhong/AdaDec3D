#!/usr/bin/env bash
# Dataset-axis cell: MSD Task08 HepaticVessel, matched 3D UX-Net Full + EffiDec3D (concat).
# One-shot: pre-flight data check -> E0 (full) -> E1 (effi concat) -> six-metric obs.
# Resumable: skips a training whose best_metric_model.pth already exists.
#
# Usage (from anywhere; script cd's into its own dir):
#   bash run_E0_E1_hv.sh            # all stages
#   bash run_E0_E1_hv.sh check      # just verify the 4 data folders
#   bash run_E0_E1_hv.sh E0|E1|obs  # a single stage
# Override defaults with env vars, e.g.:  ROOT=/data/HV CACHE=0.5 bash run_E0_E1_hv.sh
set -uo pipefail

ROOT=${ROOT:-/root/autodl-tmp/Task08_HepaticVessel}
OUT=${OUT:-/root/output}
OBS=${OBS:-/root/obs-hv-concat}
DS=Task08_HepaticVessel
CACHE=${CACHE:-0.25}          # 243 train volumes -> cache_rate 1.0 risks RAM OOM
EVAL=${EVAL:-2500}            # 60 val cases @ ~9min each -> validate rarely (250 wastes ~27h)
TAG=${TAG:-hv_sp}            # output prefix; _sp = Spacingd resampling on (distinct from old no-spacing run)
NW=${NW:-8}
STAGE=${1:-all}

cd "$(dirname "$0")"          # EffiDec3D/
log(){ echo "[hv $(date +%H:%M:%S)] $*"; }
run(){ "$@"; local rc=$?; [ $rc -eq 0 ] || { log "FAILED (rc=$rc): $*"; exit $rc; }; }

check_data(){
  local bad=0
  for f in imagesTr labelsTr imagesVal labelsVal; do
    local n; n=$(ls "$ROOT/$f"/*.nii.gz 2>/dev/null | wc -l)
    log "$f: $n"
    [ "$n" -gt 0 ] || { log "ERROR: $ROOT/$f is empty"; bad=1; }
  done
  [ $bad -eq 0 ] || { log "ABORT: the loader validates on imagesVal/labelsVal (NOT imagesTs) — create/populate them first."; exit 1; }
}

train_e0(){
  if ls "$OUT"/E0_${TAG}*/3DUXNET/$DS/best_metric_model.pth >/dev/null 2>&1; then
    log "E0 checkpoint already exists — skipping full-UX training"; return; fi
  log "training E0 (full 3D UX-Net, Spacingd on) ..."
  run python main_train_BTCV_TU.py --root "$ROOT" --output "$OUT/E0_${TAG}" \
    --dataset $DS --network 3DUXNET \
    --lr 0.001 --overlap 0.7 --crop_sample 4 --max_iter 45000 --eval_step "$EVAL" \
    --cache_rate "$CACHE" --num_workers "$NW" --gpu 0
}

train_e1(){
  if ls "$OUT"/E1_${TAG}*/3DUXNET_EffiDec3D/$DS/best_metric_model.pth >/dev/null 2>&1; then
    log "E1 checkpoint already exists — skipping EffiDec3D training"; return; fi
  log "training E1 (EffiDec3D, concatenation, Spacingd on) ..."
  run python main_train_BTCV_TU.py --root "$ROOT" --output "$OUT/E1_${TAG}" \
    --dataset $DS --network 3DUXNET_EffiDec3D --ds False --skip_aggregation concatenation \
    --lr 0.001 --overlap 0.7 --crop_sample 4 --max_iter 45000 --eval_step "$EVAL" \
    --cache_rate "$CACHE" --num_workers "$NW" --gpu 0
}

run_obs(){
  local E0 E1
  E0=$(ls "$OUT"/E0_${TAG}*/3DUXNET/$DS/best_metric_model.pth 2>/dev/null | head -1 || true)
  E1=$(ls "$OUT"/E1_${TAG}*/3DUXNET_EffiDec3D/$DS/best_metric_model.pth 2>/dev/null | head -1 || true)
  [ -n "$E0" ] && [ -n "$E1" ] || { log "ERROR: missing checkpoint (E0='$E0' E1='$E1'); train first"; exit 1; }
  log "obs E0 = $E0"
  log "obs E1 = $E1"
  run python run_observations.py --network 3DUXNET --dataset $DS \
    --root "$ROOT" --output "$OUT" --skip_aggregation concatenation \
    --e0_ckpt "$E0" --e1_ckpt "$E1" --obs_dir "$OBS"
  log "obs written to $OBS"
}

case "$STAGE" in
  check) check_data ;;
  E0)    check_data; train_e0 ;;
  E1)    train_e1 ;;
  obs)   run_obs ;;
  all)   check_data; train_e0; train_e1; run_obs ;;
  *)     echo "usage: $0 [all|check|E0|E1|obs]"; exit 1 ;;
esac
log "stage '$STAGE' complete."
