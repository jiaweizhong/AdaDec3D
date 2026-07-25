#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-validate an existing E1 (EffiDec3D) checkpoint under FP32 vs BF16 precision.

The training script measured E1's best dice under BF16 autocast (both training
and validation). The original EffiDec3D code trains/validates in FP32
(autocast disabled). This script loads the SAME BF16-trained weights and runs
validation twice — once FP32, once BF16 — to isolate how much of the
E0-vs-E1 gap is caused by validation precision alone (no retraining).

Usage (from /root/AdaDec3D/EffiDec3D):
  python revalidate_fp32.py \
      --root /root/autodl-tmp/btcv-synapse \
      --checkpoint /root/output/E1_network_3DUXNET_EffiDec3D_*/3DUXNET_EffiDec3D/BTCV13/best_metric_model.pth
"""
import argparse
import glob
import numpy as np
import torch

from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.utils import set_determinism

from load_datasets_transforms import data_loader, data_transforms
from monai_utils.inferers.utils import sliding_window_inference_1out
from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D

# Standard BTCV/Synapse 13-organ order (matches CSV columns)
BTCV13_NAMES = ["Spleen", "R.Kidney", "L.Kidney", "Gallbladder", "Esophagus",
                "Liver", "Stomach", "Aorta", "IVC", "Veins",
                "Pancreas", "R.Adrenal", "L.Adrenal"]


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/autodl-tmp/btcv-synapse")
    p.add_argument("--dataset", default="BTCV13")
    p.add_argument("--checkpoint", required=True,
                   help="Path (or glob) to E1 best_metric_model.pth")
    p.add_argument("--img_size", type=int, nargs="+", default=[96, 96, 96])
    p.add_argument("--overlap", type=float, default=0.7)
    p.add_argument("--val_batch", type=int, default=1)
    p.add_argument("--crop_sample", type=int, default=4)
    p.add_argument("--mode", default="train")   # data_loader returns valid split either way
    return p.parse_args()


def run_validation(model, val_loader, out_classes, img_size, overlap, val_batch,
                   amp_dtype):
    """Replicate the training-time validation, with a precision toggle.

    amp_dtype=None -> FP32 (autocast disabled, matches original code).
    amp_dtype=torch.bfloat16 -> BF16 (matches our modified training/validation).

    Returns (scalar_mean_like_training, per_class_mean[np.array]).
    """
    post_label = AsDiscrete(to_onehot=out_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
    # Per-class metric (clean: accumulate all, aggregate once)
    per_class_metric = DiceMetric(include_background=False, reduction="mean_batch",
                                  get_not_nans=False)
    # Training-style scalar metric (running-aggregate averaged over samples)
    running_metric = DiceMetric(include_background=False, reduction="mean",
                                get_not_nans=False)

    use_amp = amp_dtype is not None
    model.eval()
    dice_vals = []
    with torch.no_grad():
        for batch in val_loader:
            val_inputs = batch["image"].cuda()
            val_labels = batch["label"].cuda()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                val_outputs = sliding_window_inference_1out(
                    val_inputs, tuple(img_size), val_batch, model, overlap=overlap)
                labels_convert = [post_label(l) for l in decollate_batch(val_labels)]
                outputs_convert = [post_pred(o) for o in decollate_batch(val_outputs)]
                per_class_metric(y_pred=outputs_convert, y=labels_convert)
                running_metric(y_pred=outputs_convert, y=labels_convert)
                dice_vals.append(running_metric.aggregate().item())
    per_class = per_class_metric.aggregate().cpu().numpy()   # [13]
    per_class_metric.reset(); running_metric.reset()
    return float(np.mean(dice_vals)), per_class


def main():
    args = build_args()
    set_determinism(seed=0)
    device = torch.device("cuda:0")

    ckpts = sorted(glob.glob(args.checkpoint))
    assert ckpts, f"No checkpoint matched: {args.checkpoint}"
    ckpt_path = ckpts[-1]
    print(f"Checkpoint: {ckpt_path}")

    # Data (validation split)
    _, valid_samples, out_classes = data_loader(args)
    val_files = [{"image": im, "label": lb}
                 for im, lb in zip(valid_samples["images"], valid_samples["labels"])]
    _, val_transforms = data_transforms(args)
    val_ds = CacheDataset(data=val_files, transform=val_transforms,
                          cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    print(f"Val cases: {len(val_files)}  out_classes: {out_classes}")

    # Model (identical to main_train_BTCV_TU.py 3DUXNET_EffiDec3D instantiation)
    model = UXNET_EffiDec3D(
        in_chans=1, out_chans=out_classes, depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384], n_decoder_channels=48,
        drop_path_rate=0, layer_scale_init_value=1e-6, spatial_dims=3,
        skip_aggregation="addition", resolution_factor=2,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state)

    # FP32 first (matches original), then BF16 (matches our training)
    fp32_mean, fp32_pc = run_validation(model, val_loader, out_classes,
                                        args.img_size, args.overlap, args.val_batch,
                                        amp_dtype=None)
    bf16_mean, bf16_pc = run_validation(model, val_loader, out_classes,
                                        args.img_size, args.overlap, args.val_batch,
                                        amp_dtype=torch.bfloat16)

    print("\n=== Same E1 weights, FP32 vs BF16 validation ===")
    print(f"{'Organ':12s} {'FP32':>8} {'BF16':>8} {'Δ(FP32-BF16)':>14}")
    names = BTCV13_NAMES if len(fp32_pc) == 13 else [f"c{i+1}" for i in range(len(fp32_pc))]
    for n, a, b in zip(names, fp32_pc, bf16_pc):
        print(f"{n:12s} {a:8.4f} {b:8.4f} {a-b:14.4f}")
    print("-" * 46)
    print(f"{'MEAN(pc)':12s} {np.mean(fp32_pc):8.4f} {np.mean(bf16_pc):8.4f} "
          f"{np.mean(fp32_pc)-np.mean(bf16_pc):14.4f}")
    print(f"\nTraining-style scalar mean:  FP32={fp32_mean:.4f}  BF16={bf16_mean:.4f}")
    print(f"(BF16 scalar should be ~0.7549, confirming this matches the training run)")


if __name__ == "__main__":
    main()
