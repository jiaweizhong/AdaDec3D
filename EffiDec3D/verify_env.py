import sys
import monai
import torch

print(f"Python   : {sys.version.split()[0]}")
print(f"PyTorch  : {torch.__version__}")
print(f"MONAI    : {monai.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPU      : {torch.cuda.get_device_name(0)}")
print(f"BF16     : {torch.cuda.is_bf16_supported()}")

from networks.UXNet_3D.network_backbone import UXNET_EffiDec3D
from networks.swin_unetr_effidec3d import SwinUNETR_EffiDec3D
from monai_utils.inferers.utils import sliding_window_inference_1out
print("Imports  : OK")
