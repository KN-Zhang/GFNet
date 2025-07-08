# GFNet
Adapting Dense Matching for Homography Estimation with Grid-based Acceleration (CVPR'25)

# Setup
1. Torch version: 2.3.1
```
conda create --name GFNet python==3.10.13 && \
conda activate GFNet && \
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
2. Other requirements
```
pip install -r requirements.txt
```

# Dataset & Pre-trained weights
**Download link**: [https://pan.baidu.com/s/1CwyHIYBwr3PdFatqbPn-4g](https://pan.baidu.com/s/1CwyHIYBwr3PdFatqbPn-4g)  
**Extraction code**: `qwer`

Please create a folder named **ckpts** and place the pre-trained weights inside.
For the dataset, make sure to update the data path in the code to match your local setup.

# Test

--dataset: ['**mscoco**', '**vis_ir_drone**', '**googlemap_448x448**', '**googlemap_224x224**', '**googlemap_672x672**']

--conf_path: ['**./configs/basic.json**', '**./configs/vis_ir.json**', '**./configs/map.json**']

--ckpt_path: ['**./ckpts/basic/latest.pth**', '**./ckpts/vis_ir_drone/latest.pth**', '**./ckpts/googlemap/latest.pth**']

For example, to test on MSCOCO, run:
```
CUDA_VISIBLE_DEVICES=0 python -m demo_dataset \
        --dataset mscoco \
        --conf_path ./configs/basic.json \
        --ckpt_path ./ckpts/basic/latest.pth
```

# Train

Coming soon.

