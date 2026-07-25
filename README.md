# AdaDec3D

Adaptive-resolution 3D medical image segmentation built on top of EffiDec3D (a lightweight 3DUX-Net variant).

## Key documents

| Document | Purpose |
|---|---|
| [Research_Proposal.md](Research_Proposal.md) | Project overview, motivation, method design, target venues |
| [Observation_Study.md](Observation_Study.md) | O1–O11 exploratory observations; Go/No-Go gates for Paper B |
| [Experiment-Design-AdaDec3D.md](Experiment-Design-AdaDec3D.md) | Training recipes, ablation plan, result tables for AdaDec3D (Paper B) |

## Codebase layout

```
EffiDec3D/
  main_train_BTCV_TU.py     # E0 (3DUX-Net) and E1 (EffiDec3D) training
  main_train_adadec3d.py    # E2–E4 AdaDec3D training
  networks/
    adadec3d.py             # AdaDec3D model (MoE router + ROI refiner)
    network_backbone.py     # 3DUX-Net encoder backbone
  load_datasets_transforms.py
  run_E0_E1.sh              # AutoDL launch script for E0/E1
results/                    # CSVs and checkpoints (not committed)
```

## Baselines

| Model | Params | MACs | BTCV13 Dice |
|---|---|---|---|
| E0: 3DUX-Net (full) | 53.007 M | 578.74 GMac | ~79.74% (paper) |
| E1: EffiDec3D | 2.955 M | 41.06 GMac | TBD (calibration run) |

## Quick start (AutoDL RTX 5090)

```bash
# Install deps
pip install -r EffiDec3D/requirements.txt

# Download BTCV13 data
kaggle datasets download -d shinjinidey/synapse-dataset
unzip synapse-dataset.zip -d /root/autodl-tmp/btcv-raw

# Run E0 + E1 baseline calibration
cd EffiDec3D && bash run_E0_E1.sh

# Run observations (after E0/E1 checkpoints available)
cd /root && python Observation_Study.py
```

## Go/No-Go gate

Paper B (AdaDec3D) development starts only after Paper A core observations pass:
- **O5**: entropy vs net-gain Pearson r bootstrap CI lower bound > 0
- **O9**: entropy routing outperforms random at 10–30% ROI budget
