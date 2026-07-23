#!/usr/bin/env python3
"""
AdaDec3D training script.

Two-stage training:
  Stage 1 – Load EffiDec3D weights, freeze encoder + coarse decoder, train new modules.
  Stage 2 – Unfreeze all, end-to-end fine-tune with layered learning rates.

Example – Stage 1:
    python main_train_adadec3d.py \
        --root /path/to/btcv \
        --output output/adadec3d_s1 \
        --dataset BTCV13 \
        --effidec3d_weights output/E1_effidec3d/.../best_metric_model.pth \
        --stage 1 \
        --max_iter 20000 --eval_step 500 --lr 5e-4 \
        --gpu 0

Example – Stage 2:
    python main_train_adadec3d.py \
        --root /path/to/btcv \
        --output output/adadec3d_s2 \
        --dataset BTCV13 \
        --stage1_ckpt output/adadec3d_s1/.../best_metric_model.pth \
        --stage 2 \
        --max_iter 25000 --eval_step 500 --lr 5e-4 \
        --gpu 0
"""

import os
import csv
import argparse

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import CacheDataset, DataLoader, decollate_batch
from medpy import metric

from load_datasets_transforms import data_loader, data_transforms
from monai_utils.inferers.utils import sliding_window_inference_1out
from networks.adadec3d import AdaDec3D_UXNET

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="AdaDec3D training")

# Data
parser.add_argument("--root",    type=str, required=True)
parser.add_argument("--output",  type=str, required=True)
parser.add_argument("--dataset", type=str, default="BTCV13")
parser.add_argument("--mode", type=str, default="train", choices=["train", "validation"])
parser.add_argument("--img_size", type=int, nargs="+", default=[96, 96, 96])
parser.add_argument("--n_channels", type=int, default=1)

# Model
parser.add_argument("--channels",          type=int, nargs="+", default=[48, 96, 192, 384])
parser.add_argument("--n_decoder_channels", type=int, default=48)
parser.add_argument("--skip_aggregation",   type=str, default="addition")
parser.add_argument("--n_experts",          type=int, default=3)
parser.add_argument("--expert_channels",    type=int, nargs="+", default=[32, 64, 96])
parser.add_argument("--roi_quantile",       type=float, default=0.5)
parser.add_argument("--use_moe",  type=lambda x: x.lower() != "false", default=True,
                    help="Enable MoE router (False = ablation E3)")
parser.add_argument("--use_roi",  type=lambda x: x.lower() != "false", default=True,
                    help="Enable ROI refinement (False = ablation E2)")

# Training
parser.add_argument("--stage",    type=int, default=1, choices=[1, 2],
                    help="1=freeze backbone and train new modules; 2=end-to-end fine-tune")
parser.add_argument("--effidec3d_weights", type=str, default="",
                    help="Path to trained UXNET_EffiDec3D checkpoint (required for stage 1)")
parser.add_argument("--stage1_ckpt", type=str, default="",
                    help="Path to Stage 1 best checkpoint (required for stage 2)")
parser.add_argument("--max_iter",  type=int,   default=20000)
parser.add_argument("--eval_step", type=int,   default=500)
parser.add_argument("--lr",        type=float, default=5e-4)
parser.add_argument("--backbone_lr_factor", type=float, default=0.1,
                    help="Stage 2 backbone LR = lr * backbone_lr_factor")

# Loss weights
parser.add_argument("--lambda_uncertainty", type=float, default=0.10)
parser.add_argument("--lambda_resource",    type=float, default=0.05)
parser.add_argument("--lambda_router",      type=float, default=0.10)
parser.add_argument("--lambda_coarse",      type=float, default=0.50,
                    help="Weight for auxiliary coarse decoder loss")
parser.add_argument("--target_expert_cost", type=float, default=0.50,
                    help="Target normalized expected expert cost in [0,1]")

# Inference
parser.add_argument("--overlap",      type=float, default=0.7)
parser.add_argument("--overlap_mode", type=str,   default="constant")
parser.add_argument("--val_batch",    type=int,   default=1)

# System
parser.add_argument("--batch_size",  type=int,   default=1)
parser.add_argument("--crop_sample", type=int,   default=4)
parser.add_argument("--cache_rate",  type=float, default=0.5)
parser.add_argument("--num_workers", type=int,   default=2)
parser.add_argument("--gpu",         type=str,   default="0")
parser.add_argument("--seed",        type=int,   default=0)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0")
# BF16 is native on Ampere/Hopper/Blackwell; fall back to FP16 on older cards.
_amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

if args.dataset.lower() == "feta":
    args.dataset = "feta"
set_determinism(seed=args.seed)
train_samples, valid_samples, out_classes = data_loader(args)

train_files = [
    {"image": img, "label": lbl}
    for img, lbl in zip(train_samples["images"], train_samples["labels"])
]
val_files = [
    {"image": img, "label": lbl}
    for img, lbl in zip(valid_samples["images"], valid_samples["labels"])
]

train_transforms, val_transforms = data_transforms(args)

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers,
)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms,
    cache_rate=args.cache_rate, num_workers=args.num_workers,
)
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True,
)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = AdaDec3D_UXNET(
    in_chans=args.n_channels,
    out_chans=out_classes,
    depths=[2, 2, 2, 2],
    feat_size=args.channels,
    n_decoder_channels=args.n_decoder_channels,
    skip_aggregation=args.skip_aggregation,
    n_experts=args.n_experts,
    expert_channels=args.expert_channels,
    roi_quantile=args.roi_quantile,
    use_moe=args.use_moe,
    use_roi=args.use_roi,
).to(device)

# Stage-specific weight loading
if args.stage == 1:
    assert args.effidec3d_weights, "--effidec3d_weights is required for stage 1"
    ckpt = torch.load(args.effidec3d_weights, map_location=device)
    # The checkpoint may be a bare state_dict or a dict with "model_state_dict"
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_effidec3d_weights(sd)

    # Freeze encoder and coarse decoder; only new modules get gradients
    frozen_prefixes = ("uxnet_3d", "encoder2", "encoder3", "encoder4", "encoder5",
                       "decoder3", "decoder4", "decoder5", "coarse_out")
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[Stage 1] Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )

else:  # stage 2
    assert args.stage1_ckpt, "--stage1_ckpt is required for stage 2"
    sd = torch.load(args.stage1_ckpt, map_location=device)
    sd = sd.get("model_state_dict", sd)
    model.load_state_dict(sd, strict=True)
    print(f"[Stage 2] Loaded Stage 1 checkpoint: {args.stage1_ckpt}")

    backbone_lr = args.lr * args.backbone_lr_factor
    backbone_prefixes = ("uxnet_3d", "encoder2", "encoder3", "encoder4", "encoder5",
                         "decoder3", "decoder4", "decoder5", "coarse_out")
    backbone_params, new_params = [], []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in backbone_prefixes):
            backbone_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": new_params,      "lr": args.lr},
    ], weight_decay=1e-5)
    print(f"[Stage 2] backbone lr={backbone_lr:.2e}, new modules lr={args.lr:.2e}")

scaler = GradScaler('cuda', enabled=(_amp_dtype == torch.float16))

# ---------------------------------------------------------------------------
# Output directory & checkpoint resumption
# ---------------------------------------------------------------------------

root_dir = os.path.join(args.output, f"stage{args.stage}", args.dataset)
os.makedirs(root_dir, exist_ok=True)
last_ckpt_path = os.path.join(root_dir, "last_model.pth")

global_step = 0
dice_val_best = 0.0
global_step_best = 0

if os.path.isfile(last_ckpt_path) and os.path.getsize(last_ckpt_path) > 0:
    try:
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        global_step      = ckpt.get("global_step", 0)
        dice_val_best    = ckpt.get("dice_val_best", 0.0)
        global_step_best = ckpt.get("global_step_best", 0)
        print(f"[Resume] Loaded checkpoint at step {global_step}")
    except Exception as e:
        print(f"[Resume] Failed to load checkpoint: {e}. Starting from scratch.")
        os.remove(last_ckpt_path)

writer = SummaryWriter(log_dir=os.path.join(root_dir, "tensorboard"))

# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)


def interpolate_to_label(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Upsample pred (D/2 resolution) to match full-resolution label."""
    target = (label.shape[-3], label.shape[-2], label.shape[-1])
    if pred.shape[-3:] == target:
        return pred
    return F.interpolate(pred, size=target, mode="trilinear", align_corners=False)


def uncertainty_calibration_loss(coarse_pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Calibration: voxels with high predicted entropy should be where the model is wrong.
    Entropy is normalized by its theoretical maximum log(C), preserving
    comparability between crops and subjects.
    """
    prob = coarse_pred.softmax(dim=1)
    entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1)       # [B, D', H', W']

    coarse_lbl = F.interpolate(
        label.float(), size=coarse_pred.shape[2:], mode="nearest"
    ).squeeze(1).long()                                         # [B, D', H', W']
    error = (coarse_pred.argmax(dim=1) != coarse_lbl).float()  # [B, D', H', W']

    unc_norm = entropy / np.log(coarse_pred.shape[1])
    # Balance error/non-error voxels so easy background cannot dominate.
    pos = error.sum().clamp_min(1.0)
    neg = (1.0 - error).sum().clamp_min(1.0)
    weights = error * (0.5 / pos) + (1.0 - error) * (0.5 / neg)
    return ((unc_norm - error).square() * weights).sum()


def resource_penalty(router_weights: torch.Tensor) -> torch.Tensor:
    """Encourage routing toward lighter experts (penalise expert-L usage)."""
    n = router_weights.shape[1]
    # Cost proportional to expert index (S=0, M=0.5, L=1)
    costs = torch.linspace(0.0, 1.0, n, device=router_weights.device)
    return (router_weights * costs.unsqueeze(0)).sum(dim=1).mean()


def budget_loss(router_weights: torch.Tensor) -> torch.Tensor:
    """Penalize exceeding a compute budget without forcing artificial uniform use."""
    n = router_weights.shape[1]
    costs = torch.linspace(0.0, 1.0, n, device=router_weights.device)
    expected_cost = (router_weights * costs.unsqueeze(0)).sum(dim=1).mean()
    target = torch.as_tensor(args.target_expert_cost, device=router_weights.device)
    return F.relu(expected_cost - target).square()


def compute_loss(
    final_pred: torch.Tensor,
    coarse_pred: torch.Tensor,
    router_weights: torch.Tensor,
    label: torch.Tensor,
) -> tuple:
    label_for_loss = label.long()

    # Main segmentation loss on final prediction (interpolated to full res)
    L_seg = seg_loss_fn(interpolate_to_label(final_pred, label), label_for_loss)

    # Auxiliary coarse decoder loss
    L_coarse = seg_loss_fn(interpolate_to_label(coarse_pred, label), label_for_loss)

    # Uncertainty calibration
    L_unc = uncertainty_calibration_loss(coarse_pred, label)

    # Efficiency losses
    L_res    = resource_penalty(router_weights)
    L_router = budget_loss(router_weights)

    total = (L_seg
             + args.lambda_coarse       * L_coarse
             + args.lambda_uncertainty  * L_unc
             + args.lambda_resource     * L_res
             + args.lambda_router       * L_router)

    return total, {
        "L_seg":    L_seg.item(),
        "L_coarse": L_coarse.item(),
        "L_unc":    L_unc.item(),
        "L_res":    L_res.item(),
        "L_router": L_router.item(),
    }

# ---------------------------------------------------------------------------
# Validation (fast, mean DICE only – used during training)
# ---------------------------------------------------------------------------

post_label = AsDiscrete(to_onehot=out_classes)
post_pred  = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


def validation(val_loader):
    model.eval()
    dice_vals = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate", dynamic_ncols=True, leave=False):
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            with torch.autocast("cuda", dtype=_amp_dtype):
                val_outputs = sliding_window_inference_1out(
                    val_inputs,
                    tuple(args.img_size),
                    args.val_batch,
                    model,
                    overlap=args.overlap,
                )
            # upsample to label resolution before metric
            val_outputs = interpolate_to_label(val_outputs, val_labels)
            val_outputs_list = [post_pred(x)  for x in decollate_batch(val_outputs)]
            val_labels_list  = [post_label(x) for x in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            dice_vals.append(dice_metric.aggregate().item())
            dice_metric.reset()
    model.train()
    return float(np.mean(dice_vals))

# ---------------------------------------------------------------------------
# Final validation (per-class DICE + HD95 – run once after training)
# ---------------------------------------------------------------------------

def resample_3d(img, target_size):
    zoom = tuple(t / s for t, s in zip(target_size, img.shape))
    return ndimage.zoom(img, zoom, order=0, prefilter=False)


def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.dc(pred, gt), metric.binary.hd95(pred, gt)
    if pred.sum() == 0 and gt.sum() == 0:
        return np.nan, np.nan
    # A false-positive-only or false-negative-only class has zero Dice. HD95 is
    # undefined and excluded from the aggregate rather than reported as perfect.
    return 0.0, np.nan


def validation_final(val_loader):
    model.eval()
    per_class_dice, per_class_hd = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final eval", dynamic_ncols=True):
            val_inputs  = batch["image"].to(device)
            val_labels  = batch["label"].to(device)
            _, _, h, w, d = val_labels.shape
            with torch.autocast("cuda", dtype=_amp_dtype):
                val_outputs = sliding_window_inference_1out(
                    val_inputs, tuple(args.img_size), args.val_batch,
                    model, overlap=args.overlap, mode=args.overlap_mode,
                )
            val_outputs = torch.softmax(val_outputs, dim=1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels  = val_labels.cpu().numpy()[0, 0]
            val_outputs = resample_3d(val_outputs, (h, w, d))
            dice_row, hd_row = [], []
            for c in range(1, out_classes):
                d_val, h_val = calculate_metric_percase(val_outputs == c, val_labels == c)
                dice_row.append(d_val)
                hd_row.append(h_val)
            per_class_dice.append(dice_row)
            per_class_hd.append(hd_row)
    per_class_dice = np.array(per_class_dice)   # [N_val, n_classes-1]
    per_class_hd   = np.array(per_class_hd)
    return per_class_dice, per_class_hd

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Train step {global_step}", dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        with torch.autocast("cuda", dtype=_amp_dtype):
            outputs = model(x)
            # outputs = [final_pred, coarse_pred, router_weights] during training
            final_pred    = outputs[0]
            coarse_pred   = outputs[1]
            router_weights = outputs[2]

            loss, loss_dict = compute_loss(final_pred, coarse_pred, router_weights, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        pbar.set_description(
            f"step={global_step}/{args.max_iter} "
            f"loss={loss.item():.4f} "
            f"Lseg={loss_dict['L_seg']:.3f} "
            f"Lunc={loss_dict['L_unc']:.3f} "
            f"Lres={loss_dict['L_res']:.3f} "
            f"Lrtr={loss_dict['L_router']:.3f}"
        )

        # Tensorboard
        writer.add_scalar("Loss/total",    loss.item(),               global_step)
        writer.add_scalar("Loss/seg",      loss_dict["L_seg"],        global_step)
        writer.add_scalar("Loss/coarse",   loss_dict["L_coarse"],     global_step)
        writer.add_scalar("Loss/unc",      loss_dict["L_unc"],        global_step)
        writer.add_scalar("Loss/resource", loss_dict["L_res"],        global_step)
        writer.add_scalar("Loss/router",   loss_dict["L_router"],     global_step)

        # Periodic validation
        if global_step % args.eval_step == 0 or global_step == args.max_iter:
            dice_val = validation(val_loader)
            writer.add_scalar("Val/mean_dice", dice_val, global_step)

            # Always overwrite last_model.pth for checkpoint resumption
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "global_step":          global_step,
                "dice_val_best":        dice_val_best,
                "global_step_best":     global_step_best,
            }, last_ckpt_path)

            if dice_val > dice_val_best:
                dice_val_best    = dice_val
                global_step_best = global_step
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    os.path.join(root_dir, "best_metric_model.pth"),
                )
                print(f"[Saved] step={global_step}  best_dice={dice_val_best:.4f}")
            else:
                print(f"[Val]   step={global_step}  dice={dice_val:.4f}  best={dice_val_best:.4f}")

        global_step += 1
        if global_step > args.max_iter:
            break

    return global_step, dice_val_best, global_step_best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(f"AdaDec3D stage {args.stage} | dataset={args.dataset} | max_iter={args.max_iter}")
print(f"Output dir: {root_dir}")

while global_step < args.max_iter:
    global_step, dice_val_best, global_step_best = train_one_epoch(
        global_step, train_loader, dice_val_best, global_step_best
    )

# Final evaluation
print("\nRunning final per-class evaluation on best checkpoint...")
best_ckpt = os.path.join(root_dir, "best_metric_model.pth")
ckpt = torch.load(best_ckpt, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

per_class_dice, per_class_hd = validation_final(val_loader)

mean_dice_per_class = np.nanmean(per_class_dice, axis=0)  # [n_classes-1]
mean_hd_per_class   = np.nanmean(per_class_hd, axis=0)

BTCV13_NAMES = [
    "Aorta", "Gallbladder", "Spleen", "L.Kidney", "R.Kidney",
    "Liver", "Stomach", "IVC", "Port.Vein",
    "Pancreas", "R.Adrenal", "L.Adrenal", "Duodenum",
]
CLASS_NAMES = BTCV13_NAMES if args.dataset == "BTCV13" else \
              [f"class_{i}" for i in range(1, out_classes)]

print("\n=== Final Results ===")
print(f"{'Class':>15}  {'DICE':>6}  {'HD95':>7}")
print("-" * 35)
for name, d, h in zip(CLASS_NAMES, mean_dice_per_class, mean_hd_per_class):
    print(f"{name:>15}  {d:.4f}  {h:7.2f}")
print("-" * 35)
print(f"{'Mean':>15}  {np.nanmean(mean_dice_per_class):.4f}  {np.nanmean(mean_hd_per_class):7.2f}")

# Save CSV
csv_path = os.path.join(root_dir, "per_class_results.csv")
with open(csv_path, "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["class", "mean_dice", "mean_hd95"])
    for name, d, h in zip(CLASS_NAMES, mean_dice_per_class, mean_hd_per_class):
        writer_csv.writerow([name, f"{d:.4f}", f"{h:.2f}"])
    writer_csv.writerow(["Mean", f"{np.nanmean(mean_dice_per_class):.4f}",
                          f"{np.nanmean(mean_hd_per_class):.2f}"])
print(f"Results saved to {csv_path}")
writer.close()
