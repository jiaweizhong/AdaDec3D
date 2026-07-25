#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Break down EffiDec3D (UXNET_EffiDec3D) MACs into encoder vs decoder per top-level
module, to see how much decoder compute AdaDec3D can actually cut.

MACs are counted per Conv3d / ConvTranspose3d / Linear via forward hooks
(output-spatial convention), tagged by top-level child module, then grouped into
ENCODER (uxnet_3d + encoder2..5 skip blocks) and DECODER (decoder3..5 + out head).
Total is cross-checked against ptflops.

Usage (from /root/AdaDec3D/EffiDec3D):
  python profile_macs.py
"""
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D

ENCODER_PREFIXES = ("uxnet_3d", "encoder")
DECODER_PREFIXES = ("decoder", "out")


def module_macs(m, out):
    o = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
        k = 1
        for kk in m.kernel_size:
            k *= kk
        spatial = 1
        for s in o.shape[2:]:
            spatial *= s
        return o.shape[0] * m.out_channels * (m.in_channels // m.groups) * k * spatial
    if isinstance(m, nn.Linear):
        lead = 1
        for s in o.shape[:-1]:
            lead *= s
        return lead * m.in_features * m.out_features
    return 0


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UXNET_EffiDec3D(
        in_chans=1, out_chans=14, depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384], n_decoder_channels=48,
        drop_path_rate=0, layer_scale_init_value=1e-6, spatial_dims=3,
        skip_aggregation="addition", resolution_factor=2).to(device).eval()

    top_macs = {name: 0 for name, _ in model.named_children()}
    handles = []
    for tname, tmod in model.named_children():
        def make_hook(nm):
            def hook(m, inp, out):
                top_macs[nm] += module_macs(m, out)
            return hook
        for sub in tmod.modules():
            if isinstance(sub, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                handles.append(sub.register_forward_hook(make_hook(tname)))

    x = torch.zeros(1, 1, 96, 96, 96, device=device)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    hook_total = sum(top_macs.values())
    macs_pt, params_pt = get_model_complexity_info(
        model, (1, 96, 96, 96), as_strings=False, print_per_layer_stat=False, verbose=False)

    def group(name):
        if name.startswith(DECODER_PREFIXES):
            return "DECODER"
        if name.startswith(ENCODER_PREFIXES):
            return "ENCODER"
        return "OTHER"

    print(f"\nptflops total: {macs_pt/1e9:.2f} GMac, {params_pt/1e6:.3f} M params")
    print(f"hook  total: {hook_total/1e9:.2f} GMac (conv+linear only; norms/act ~0)\n")

    print(f"{'Module':14s} {'Group':8s} {'GMac':>9} {'% total':>9}")
    print("-" * 44)
    for name in sorted(top_macs, key=lambda n: -top_macs[n]):
        g = top_macs[name] / 1e9
        print(f"{name:14s} {group(name):8s} {g:9.3f} {100*top_macs[name]/hook_total:8.1f}%")

    enc = sum(v for n, v in top_macs.items() if group(n) == "ENCODER")
    dec = sum(v for n, v in top_macs.items() if group(n) == "DECODER")
    oth = sum(v for n, v in top_macs.items() if group(n) == "OTHER")
    print("-" * 44)
    print(f"{'ENCODER':14s} {'':8s} {enc/1e9:9.3f} {100*enc/hook_total:8.1f}%")
    print(f"{'DECODER':14s} {'':8s} {dec/1e9:9.3f} {100*dec/hook_total:8.1f}%")
    if oth:
        print(f"{'OTHER':14s} {'':8s} {oth/1e9:9.3f} {100*oth/hook_total:8.1f}%")
    print(f"\n=> Decoder is {100*dec/hook_total:.1f}% of compute; "
          f"that is the ceiling AdaDec3D's MoE/ROI can act on.")


if __name__ == "__main__":
    main()
