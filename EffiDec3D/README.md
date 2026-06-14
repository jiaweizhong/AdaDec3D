# EffiDec3D

Official Pytorch implementation of the paper [EffiDec3D: An Optimized Decoder for High-Performance and Efficient 3D Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Rahman_EffiDec3D_An_Optimized_Decoder_for_High-Performance_and_Efficient_3D_Medical_CVPR_2025_paper.html) published in CVPR 2025. [code](https://github.com/SLDGroup/EffiDec3D) [video](https://youtu.be/y4TGTeXau_4?si=XxM7bPShf-NdMn7o)
<br>
[Md Mostafijur Rahman](https://mostafij-rahman.github.io/), [Radu Marculescu](https://radum.ece.utexas.edu/)
<p>The University of Texas at Austin</p>

## Update

### **🚀 July 20, 2025: Code released for EffiDec3D optimized networks!!!**

## Usage:
### Recommended environment:
**Please run the following commands.**
```
conda create -n effidec3denv python=3.8
conda activate effidec3denv
```

### Training:
```
cd into EffiDec3D
python main_finetune_BTCV_TU.py --root </data/btcv_trns/> --output output_folder/run1 --dataset BTCV13 --img_size 96 96 96 --n_channels 1 --network 3DUXNET_EffiDec3D --channels 48 96 192 384 --n_decoder_channels 48 --ds False --mode train --pretrain False --batch_size 1 --crop_sample 4 --lr 0.001 --optim AdamW --max_iter 45000  --eval_step 250 --val_batch 1 --gpu 0 --cache_rate 1.0 --num_workers 4 --overlap 0.7 #> output_folder/BTCV13_3DUXNET_EffiDec3D_loss_dsFalse_1out_96x96x96_lr1e3_itr45000_overlap070_run1.txt

```

### Testing:
```
cd into EffiDec3D 
```

## Citations

``` 
@inproceedings{rahman2025effidec3d,
  title={EffiDec3D: An Optimized Decoder for High-Performance and Efficient 3D Medical Image Segmentation},
  author={Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={10435--10444},
  year={2025}
}
```
