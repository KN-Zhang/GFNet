# GFNet
👉  [Adapting Dense Matching for Homography Estimation with Grid-based Acceleration (CVPR'25)](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Adapting_Dense_Matching_for_Homography_Estimation_with_Grid-based_Acceleration_CVPR_2025_paper.pdf)


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

Please create a folder named ```ckpts``` and place the pre-trained weights inside with the following file structure:
```
project_root/
├── ckpts/
│   ├── basic/
│   │   └── latest.pth          # for mscoco
│   ├── vis_ir/
│   │   └── latest.pth          # for vis-ir-drone
│   └── googlemap/
│       └── latest.pth          # for googlemap

```

For the dataset, please create a folder named ```data``` and place the downloaded files inside.
Make sure to update the dataset root path in ```configs/__init__.py``` to match your local directory structure. 
You can train our model on any dataset by providing aligned image pairs.

The data file structure should look like:
```
project_root/
├── data/
│   ├── train/
│   │   ├── glunet_448x448_occlusion/
│   │   ├── VIS-IR-drone/
│   │   ├── GoogleMap/
│   │   └── your_own_data/      # directory for custom aligned image pairs across modalities
│   │       ├── modality_1/
│   │       └── modality_2/
│   └── test/
│       ├── mscoco_1k_448x448/
│       │   ├── source/         # source image
│       │   ├── target/         # target image
│       │   └── H_s2t/          # ground truth homography
│       └── ...                 # other test sets
```


# Test

To run inference, execute:
```
bash scripts/test_script.sh
```

You can configure the script with the following arguments:

- `--dataset`: one of  
  `['mscoco', 'vis_ir_drone', 'googlemap_448x448', 'googlemap_224x224', 'googlemap_672x672', 'your_own_dataset']`

- `--conf_path`: configuration file path, e.g.,  
  `['configs/basic.json', 'configs/vis_ir.json', 'configs/map.json']`

- `--ckpt_path`: path to the pre-trained weights, e.g.,  
  `['ckpts/basic/latest.pth', 'ckpts/vis_ir_drone/latest.pth', 'ckpts/googlemap/latest.pth']`


# Train

To run training, execute:
```
bash scripts/train_script.sh
```
Please revise the arguments accordingly.
If you would like to fine-tune from a pre-trained checkpoint, simply uncomment the ```--ft``` option and specify the checkpoint path using ```--ft_ckpt```.

# 📚 Citation
If you find this work helpful, please cite our paper:
```
@inproceedings{zhang2025adapting,
  title={Adapting dense matching for homography estimation with grid-based acceleration},
  author={Zhang, Kaining and Deng, Yuxin and Ma, Jiayi and Favaro, Paolo},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={6294--6303},
  year={2025}
}
```

# 🙏 Acknowledgement

This project is built upon the [RoMa](https://github.com/Parskatt/RoMa) codebase.
We sincerely thank the original authors for their excellent work and open-source contributions.