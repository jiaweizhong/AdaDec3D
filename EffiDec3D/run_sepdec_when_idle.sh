#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Overnight job. Waits until the GPU is idle, then in order:
#   1. (re)trains the full 3D UX-Net E0 if its checkpoint is missing
#      (the original E0 was deleted, so this is what runs tonight);
#   2. trains the depthwise-separable decoder E1' if missing (already done -> skipped);
#   3. runs the paired observation, full E0 vs separable E1' (3DUXNET_SEP).
# Both E0 and E1' land under OBS_OUT, so the obs finds them without staging.
# Safe to launch inside tmux and leave hanging:
#     tmux new -s sepdec
#     bash run_sepdec_when_idle.sh
#     # Ctrl-b then d to detach ;  tmux attach -t sepdec to return
#
# Idle = GPU memory below MEM_FREE_MB *and* no compute processes, held for
# NEED_FREE consecutive polls (default 5 x 60s = 5 min of sustained idle).
# All config below can be overridden via env vars, e.g.  POLL=30 GPU=0 bash ...
# ---------------------------------------------------------------------------
set -u

GPU=${GPU:-0}                                   # physical GPU index to watch/use
ROOT=${ROOT:-/root/autodl-tmp/btcv-synapse}     # BTCV13 dataset root
CODE=${CODE:-/root/AdaDec3D/EffiDec3D}          # repo working dir
DATASET=${DATASET:-BTCV13}
OUTBASE=${OUTBASE:-/root/output/E1_uxnet_sepdec}   # E1' training --output; main_train mangles it
OUTBASE_E0=${OUTBASE_E0:-/root/output/E0_uxnet_full}  # E0 (full 3DUXNET) training --output
OBS_OUT=${OBS_OUT:-/root/output}                # obs --output (globs for E0 + E1' ckpts)
OBS_DIR=${OBS_DIR:-/root/obs-uxnet-sepdec}      # where results.json + figures land
LOG=${LOG:-/root/output/sepdec_when_idle.log}
MEM_FREE_MB=${MEM_FREE_MB:-2000}                # "free" when used mem < this
NEED_FREE=${NEED_FREE:-5}                        # consecutive free polls required
POLL=${POLL:-60}                                 # seconds between polls

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found"; exit 1; }

# single-instance lock so a re-run does not double-launch
LOCK=/tmp/sepdec_when_idle.lock
mkdir "$LOCK" 2>/dev/null || { echo "another instance holds $LOCK; exit"; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

mkdir -p "$(dirname "$LOG")"
log "watcher up: GPU $GPU must be idle (<${MEM_FREE_MB}MB, 0 procs) for ${NEED_FREE}x${POLL}s"

# ------------------------------- wait loop -------------------------------
free_count=0
while :; do
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  nproc=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  if ! [[ "$used" =~ ^[0-9]+$ ]]; then
    log "nvidia-smi query failed; retry in ${POLL}s"; sleep "$POLL"; continue
  fi
  if [ "$used" -lt "$MEM_FREE_MB" ] && [ "$nproc" -eq 0 ]; then
    free_count=$((free_count+1))
    log "idle ${used}MB / ${nproc} procs  [${free_count}/${NEED_FREE}]"
    [ "$free_count" -ge "$NEED_FREE" ] && break
  else
    [ "$free_count" -gt 0 ] && log "busy again (${used}MB / ${nproc} procs) -> reset"
    free_count=0
  fi
  sleep "$POLL"
done

log "GPU $GPU is free -> launching job"
cd "$CODE" || { log "cannot cd $CODE"; exit 1; }
log "git pull (fetch latest UXNET_SepDec code):"
git pull 2>&1 | tee -a "$LOG" || log "git pull failed (continuing with local code)"

# --------------------------- E0 full 3D UX-Net (retrain if missing) ---------------------------
E0CKPT=$(ls -1 "$OBS_OUT"/*/3DUXNET/"$DATASET"/best_metric_model.pth 2>/dev/null | head -1)
if [ -n "$E0CKPT" ]; then
  log "E0 (full 3DUXNET) checkpoint present ($E0CKPT) -> skipping"
else
  log "E0 full 3DUXNET checkpoint missing -> training it (45k iters) ..."
  python main_train_BTCV_TU.py \
    --root "$ROOT" --output "$OUTBASE_E0" \
    --dataset "$DATASET" --network 3DUXNET \
    --channels 48 96 192 384 --n_channels 1 --ds False --mode train --pretrain False \
    --batch_size 1 --crop_sample 4 --lr 0.001 --optim AdamW --max_iter 45000 \
    --eval_step 250 --val_batch 1 --gpu "$GPU" --cache_rate 1.0 --num_workers 4 --overlap 0.7 \
    2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { log "E0 TRAINING FAILED (rc=$rc) -> abort"; exit "$rc"; }
  log "E0 training complete"
fi

# --------------------------- E1' depthwise-separable (skip if done) ---------------------------
CKPT=$(ls -1 "$OBS_OUT"/*/3DUXNET_SepDec/"$DATASET"/best_metric_model.pth 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
  log "E1' checkpoint already present ($CKPT) -> skipping training"
else
  log "training 3DUXNET_SepDec (45k iters) ..."
  python main_train_BTCV_TU.py \
    --root "$ROOT" --output "$OUTBASE" \
    --dataset "$DATASET" --network 3DUXNET_SepDec \
    --channels 48 96 192 384 --n_channels 1 --ds False --mode train --pretrain False \
    --batch_size 1 --crop_sample 4 --lr 0.001 --optim AdamW --max_iter 45000 \
    --eval_step 250 --val_batch 1 --gpu "$GPU" --cache_rate 1.0 --num_workers 4 --overlap 0.7 \
    2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { log "E1' TRAINING FAILED (rc=$rc) -> abort"; exit "$rc"; }
  log "E1' training complete"
fi

# --------------------------- paired observation (E0 vs E1') ---------------------------
log "running observations: full 3D UX-Net (E0) vs depthwise-separable (E1') ..."
CUDA_VISIBLE_DEVICES=$GPU python run_observations.py \
  --network 3DUXNET_SEP --dataset "$DATASET" \
  --root "$ROOT" --output "$OBS_OUT" --obs_dir "$OBS_DIR" \
  --skip_aggregation concatenation 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "observations finished rc=$rc -> results in $OBS_DIR"
log "ALL DONE. Copy $OBS_DIR back and fill Observation_Study.md."
