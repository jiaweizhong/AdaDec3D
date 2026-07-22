import os, numpy as np, h5py, nibabel as nib
from collections import defaultdict

src = "/root/autodl-tmp/Synapse"
dst = "/root/autodl-tmp/btcv-synapse"
for d in ["imagesTr", "labelsTr", "imagesVal", "labelsVal"]:
    os.makedirs(f"{dst}/{d}", exist_ok=True)

# H5 test volumes → validation NIfTI (all 12)
val_cases = []
for fname in sorted(os.listdir(f"{src}/test_vol_h5")):
    if not fname.endswith(".h5"):
        continue
    cid = fname.replace("case", "").replace(".npy.h5", "").replace(".h5", "")
    with h5py.File(f"{src}/test_vol_h5/{fname}") as f:
        img = np.array(f["image"])
        lbl = np.array(f["label"])
    nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)),
             f"{dst}/imagesVal/img{cid}.nii.gz")
    nib.save(nib.Nifti1Image(lbl.astype(np.uint8), np.eye(4)),
             f"{dst}/labelsVal/label{cid}.nii.gz")
    val_cases.append(cid)
    print(f"Val  {cid}: {img.shape}")

# NPZ 2D slices → reconstruct 3D training NIfTI (all 18)
slices = defaultdict(dict)
for fname in sorted(os.listdir(f"{src}/train_npz")):
    if not fname.endswith(".npz"):
        continue
    parts = fname.replace(".npz", "").split("_")
    cid  = parts[0].replace("case", "")
    sidx = int(parts[1].replace("slice", ""))
    slices[cid][sidx] = fname

train_cases = []
for cid in sorted(slices):
    indices = slices[cid]
    n  = max(indices) + 1
    s0 = np.load(f"{src}/train_npz/{indices[min(indices)]}")
    H, W = s0["image"].shape
    vol = np.zeros((H, W, n), dtype=np.float32)
    seg = np.zeros((H, W, n), dtype=np.uint8)
    for idx, fn in indices.items():
        d = np.load(f"{src}/train_npz/{fn}")
        vol[:, :, idx] = d["image"]
        seg[:, :, idx] = d["label"]
    nib.save(nib.Nifti1Image(vol, np.eye(4)), f"{dst}/imagesTr/img{cid}.nii.gz")
    nib.save(nib.Nifti1Image(seg, np.eye(4)), f"{dst}/labelsTr/label{cid}.nii.gz")
    train_cases.append(cid)
    print(f"Train {cid}: {vol.shape}")

print(f"\nTRAIN = {sorted(train_cases)}")
print(f"VAL   = {sorted(val_cases)}")
print(f"\nDone → {dst}")
