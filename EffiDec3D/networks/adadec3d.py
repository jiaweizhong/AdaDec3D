"""
AdaDec3D: Uncertainty-Guided Adaptive Decoder for Efficient 3D Medical Image Segmentation

Builds on top of UXNET_EffiDec3D (resolution_factor=2, n_decoder_channels=48).
Three new modules on top of the frozen EffiDec3D backbone:
  1. Uncertainty estimation (entropy on coarse softmax output)
  2. Adaptive Router + MoE Experts (3 experts with different channel widths)
  3. ROI-aware Refinement (residual refinement on high-uncertainty voxels only)
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetBasicBlock, get_conv_layer
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from networks.UXNet_3D.uxnet_encoder import uxnet_conv
from networks.UXNet_3D.network_backbone import ModifiedUnetrUpBlock


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class ExpertHead(nn.Module):
    """
    A single MoE expert: two-layer bottleneck conv block + output head.
    Input/output channels stay at `in_ch` so experts are interchangeable;
    `hidden_ch` controls the capacity (S=32, M=64, L=96).
    """

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=in_ch, out_channels=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.relu(x + self.block(x)))


class ROIRefineBlock(nn.Module):
    """
    Applies a residual conv block only inside the high-uncertainty ROI mask.
    Voxels outside the mask are kept unchanged (zero residual update).
    """

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        # roi_mask: [B, 1, D/2, H/2, W/2] float in {0, 1}
        residual = self.conv(feat) * roi_mask
        return self.relu(feat + residual)


class AdaptiveRouter(nn.Module):
    """
    Routes each sample to the appropriate expert based on:
      - global_feat: average-pooled bottleneck feature [B, feat_dim]
      - unc_stat:    mean uncertainty of the coarse prediction [B, 1]
    """

    def __init__(self, feat_dim: int, n_experts: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim + 1, n_experts)

    def forward(self, global_feat: torch.Tensor, unc_stat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([global_feat, unc_stat], dim=-1)
        return torch.softmax(self.fc(x), dim=-1)  # [B, n_experts]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AdaDec3D_UXNET(nn.Module):
    """
    AdaDec3D built on a 3D UX-Net backbone with EffiDec3D's compressed decoder.

    Shared with UXNET_EffiDec3D (resolution_factor=2):
      uxnet_3d, encoder2-5, decoder3-5

    New modules:
      roi_refiner   – uncertainty-masked residual refinement
      router        – sample-level expert selection
      experts       – ModuleList of 3 ExpertHead (S/M/L)

    Training forward returns a list [final_pred, coarse_pred, router_weights]
    so the existing training loop (`if type(p) is not list`) works unchanged.

    Eval forward returns only final_pred (same shape as EffiDec3D output).
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 14,
        depths: list = [2, 2, 2, 2],
        feat_size: list = [48, 96, 192, 384],
        n_decoder_channels: int = 48,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        res_block: bool = True,
        skip_aggregation: str = "addition",
        # AdaDec3D-specific
        n_experts: int = 3,
        expert_channels: list = [32, 64, 96],
        roi_quantile: float = 0.5,
        # Ablation flags (disable individual modules for E2/E3 experiments)
        use_moe: bool = True,   # False → always use middle expert (Expert-M)
        use_roi: bool = True,   # False → skip ROI refinement
    ) -> None:
        super().__init__()

        assert len(expert_channels) == n_experts, \
            "len(expert_channels) must equal n_experts"

        self.n_decoder_channels = n_decoder_channels
        self.n_experts = n_experts
        self.roi_quantile = roi_quantile
        self.use_moe = use_moe
        self.use_roi = use_roi

        # --- Encoder (identical to UXNET_EffiDec3D, resolution_factor=2) ---
        out_indice = list(range(len(feat_size)))
        self.uxnet_3d = uxnet_conv(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            out_indices=out_indice,
        )

        n_ch = n_decoder_channels
        n_ch_lo = min(n_ch, feat_size[0])  # for the finest skip (encoder2/decoder3)

        self.encoder2 = UnetrBasicBlock(3, feat_size[0], n_ch_lo, 3, 1, norm_name, res_block)
        self.encoder3 = UnetrBasicBlock(3, feat_size[1], n_ch,    3, 1, norm_name, res_block)
        self.encoder4 = UnetrBasicBlock(3, feat_size[2], n_ch,    3, 1, norm_name, res_block)
        self.encoder5 = UnetrBasicBlock(3, feat_size[3], n_ch,    3, 1, norm_name, res_block)

        # --- Coarse Decoder (identical to UXNET_EffiDec3D, resolution_factor=2) ---
        self.decoder5 = ModifiedUnetrUpBlock(3, n_ch,    n_ch,    3, 2, norm_name, res_block, skip_aggregation)
        self.decoder4 = ModifiedUnetrUpBlock(3, n_ch,    n_ch,    3, 2, norm_name, res_block, skip_aggregation)
        self.decoder3 = ModifiedUnetrUpBlock(3, n_ch,    n_ch_lo, 3, 2, norm_name, res_block, skip_aggregation)
        self.coarse_out = UnetOutBlock(spatial_dims=3, in_channels=n_ch_lo, out_channels=out_chans)

        # --- ROI Refinement ---
        self.roi_refiner = ROIRefineBlock(n_ch_lo)

        # --- Router ---
        self.router = AdaptiveRouter(feat_dim=n_ch, n_experts=n_experts)

        # --- Expert Decoders ---
        self.experts = nn.ModuleList([
            ExpertHead(in_ch=n_ch_lo, hidden_ch=ch, out_ch=out_chans)
            for ch in expert_channels
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, x_in: torch.Tensor):
        outs = self.uxnet_3d(x_in)
        enc2 = self.encoder2(outs[0])
        enc3 = self.encoder3(outs[1])
        enc4 = self.encoder4(outs[2])
        enc_hidden = self.encoder5(outs[3])
        return enc2, enc3, enc4, enc_hidden

    def _coarse_decode(self, enc2, enc3, enc4, enc_hidden):
        d = self.decoder5(enc_hidden, enc4)
        d = self.decoder4(d, enc3)
        coarse_feat = self.decoder3(d, enc2)   # [B, n_ch_lo, D/2, H/2, W/2]
        coarse_pred = self.coarse_out(coarse_feat)
        return coarse_feat, coarse_pred

    def _uncertainty(self, coarse_pred: torch.Tensor) -> torch.Tensor:
        prob = coarse_pred.softmax(dim=1)
        return -(prob * torch.log(prob + 1e-8)).sum(dim=1)  # [B, D/2, H/2, W/2]

    def _roi_mask(self, uncertainty: torch.Tensor) -> torch.Tensor:
        B = uncertainty.shape[0]
        flat = uncertainty.view(B, -1)
        threshold = flat.quantile(self.roi_quantile, dim=-1).view(B, 1, 1, 1)
        return (uncertainty > threshold).unsqueeze(1).float()  # [B, 1, D/2, H/2, W/2]

    def _route_and_combine(
        self,
        refined_feat: torch.Tensor,
        enc_hidden: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = refined_feat.shape[0]
        global_feat = enc_hidden.flatten(2).mean(-1)                     # [B, n_ch]
        unc_stat = uncertainty.mean(dim=[1, 2, 3], keepdim=False).unsqueeze(-1)  # [B, 1]
        router_weights = self.router(global_feat, unc_stat)              # [B, n_experts]

        expert_outs = torch.stack(
            [expert(refined_feat) for expert in self.experts], dim=1
        )  # [B, n_experts, C, D/2, H/2, W/2]

        w = router_weights.view(B, self.n_experts, 1, 1, 1, 1)
        final_pred = (w * expert_outs).sum(dim=1)                        # [B, C, D/2, H/2, W/2]
        return final_pred, router_weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_in: torch.Tensor,
        return_uncertainty: bool = False,
        return_router: bool = False,
        return_roi: bool = False,
    ):
        enc2, enc3, enc4, enc_hidden = self._encode(x_in)
        coarse_feat, coarse_pred = self._coarse_decode(enc2, enc3, enc4, enc_hidden)

        uncertainty = self._uncertainty(coarse_pred)
        roi_mask    = self._roi_mask(uncertainty)

        # ROI refinement (ablation: skip when use_roi=False)
        refined_feat = self.roi_refiner(coarse_feat, roi_mask) if self.use_roi else coarse_feat

        # MoE routing (ablation: fixed middle expert when use_moe=False)
        if self.use_moe:
            final_pred, router_weights = self._route_and_combine(
                refined_feat, enc_hidden, uncertainty
            )
        else:
            mid = len(self.experts) // 2
            final_pred = self.experts[mid](refined_feat)
            B = x_in.shape[0]
            router_weights = torch.zeros(B, self.n_experts, device=x_in.device)
            router_weights[:, mid] = 1.0

        if self.training:
            # [final_pred, coarse_pred, router_weights] — training loop unpacks by index
            return [final_pred, coarse_pred, router_weights]

        # Eval: optionally return auxiliary outputs for analysis
        if return_uncertainty or return_router or return_roi:
            extras = {}
            if return_uncertainty:
                extras["uncertainty"] = uncertainty
            if return_router:
                extras["router_weights"] = router_weights
            if return_roi:
                extras["roi_mask"] = roi_mask
            return final_pred, extras

        return final_pred

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_effidec3d_weights(self, state_dict: dict, strict: bool = False) -> None:
        """
        Load encoder + coarse decoder weights from a trained UXNET_EffiDec3D checkpoint.
        Keys that don't exist in AdaDec3D (e.g. `out.*`) are silently skipped.
        The `coarse_out` in AdaDec3D maps from `out` in the checkpoint.
        """
        remap = {}
        for k, v in state_dict.items():
            if k.startswith("out."):
                # EffiDec3D's output head → AdaDec3D's coarse_out
                remap["coarse_" + k] = v
            else:
                remap[k] = v

        missing, unexpected = self.load_state_dict(remap, strict=strict)
        shared = [k for k in remap if k not in missing]
        print(f"[AdaDec3D] Loaded {len(shared)} shared keys from EffiDec3D checkpoint")
        if missing:
            print(f"[AdaDec3D] Missing (new modules, expected): {missing[:8]}{'...' if len(missing) > 8 else ''}")
