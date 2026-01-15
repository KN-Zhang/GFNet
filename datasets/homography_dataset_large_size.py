import os
import glob
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import kornia.geometry.transform as KGT

import gfnet_configs
from datasets.generate_random_H_large_size import randomH


class RandomGaussianBlur:
    """Apply Gaussian Blur randomly with a given probability."""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img
    
class HomographyDataset(Dataset):
    def __init__(self,
                 dataset,
                 mode,
                 input_resolution=448,
                 initial_transforms=None,
                 bi=False,
                 normalize=True,
                 deformation_ratio=[0.3],
                 input_format='rgb',
                 **kwargs):
        super().__init__()

        self.mode = mode
        self.dataset = dataset
        assert input_resolution is not None, 'you should provide an input resolution.'
        self.input_resolution = input_resolution
        self.initial_transforms = initial_transforms
        self.bi = bi
        self.input_resize = transforms.Resize(size=self.input_resolution, interpolation=3, antialias=None)

        self.normalize = normalize
        self.input_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.deformation_ratio = deformation_ratio
        self.input_format = input_format
        imgs0 = []
        imgs1 = []
        
        if mode == 'train':
            if dataset == 'vis_ir_drone':
                path = f'{gfnet_configs.cfg.DATA_PATH}/train/VIS-IR-drone'
                test_list = open(f'{path}/test_list_original.txt').read().split('\n')
                all_list = os.listdir(f'{path}/train/trainimg/')
                train_list = [x for x in all_list if x not in test_list][:5000]
                for image_name in train_list:
                    if torch.rand(1)>0.5:
                        imgs0.append(f'{path}/train/trainimg/' + image_name)
                        imgs1.append(f'{path}/train/trainimgr/' + image_name) ## r
                    else:
                        imgs0.append(f'{path}/train/trainimgr/' + image_name) ## r
                        imgs1.append(f'{path}/train/trainimg/' + image_name)
            elif dataset == 'googlemap':
                path = f'{gfnet_configs.cfg.DATA_PATH}/train/GoogleMap'
                train_list = os.listdir(f'{path}/map/')[:5000]
                for image_name in train_list:
                    if torch.rand(1)>0.5:
                        imgs0.append(f'{path}/satellite/' + image_name)
                        imgs1.append(f'{path}/map/' + image_name)
                    else:
                        imgs0.append(f'{path}/map/' + image_name)
                        imgs1.append(f'{path}/satellite/' + image_name)
            elif dataset == 'glunet_448x448_occlusion':
                path = f'{gfnet_configs.cfg.DATA_PATH}/train/glunet_448x448_occlusion/target'
                train_list = glob.glob(os.path.join(path, '*'))
                self.H_stg = []
                self.mask = []
                for image_path in train_list:
                    image_name = image_path.split('/')[-1]
                    imgs0.append(image_path)
                    imgs1.append(os.path.join(path.replace('target', 'source'), image_name))
                    self.mask.append(os.path.join(path.replace('target', 'mask'), image_name))
                    self.H_stg.append(os.path.join(path.replace('target', 'H_s2t'), image_name.replace('jpg', 'json')))
        elif mode == 'val':
            self.input_resize = transforms.Compose([
                transforms.Resize(size=input_resolution, interpolation=3),
                transforms.ToTensor()
                ])                      
            if dataset == 'vis_ir_drone':
                path = f'{gfnet_configs.cfg.DATA_PATH}/test/visir_1k_448x448/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('png', 'json')) for i in test_list]
            elif dataset == 'googlemap':
                path = f'{gfnet_configs.cfg.DATA_PATH}/test/googlemap_1k_448x448_new/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('jpg', 'json')) for i in test_list]
            elif dataset == 'googlemap_224x224':
                path = f'{gfnet_configs.cfg.DATA_PATH}/test/googlemap_1k_224x224/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('jpg', 'json')) for i in test_list]
            elif dataset == 'googlemap_672x672':
                path = f'{gfnet_configs.cfg.DATA_PATH}/test/googlemap_1k_672x672/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('jpg', 'json')) for i in test_list]            
            elif dataset == 'mscoco':
                path = f'{gfnet_configs.cfg.DATA_PATH}/test/mscoco_1k_448x448/target'
                test_list = os.listdir(path)
                self.H_stg = [os.path.join(path.replace('target', 'H_s2t'), i.replace('png', 'json')) for i in test_list]
                test_list = os.listdir(path)
            imgs0 = [os.path.join(path, i) for i in test_list] ## target
            imgs1 = [os.path.join(path.replace('target', 'source'), i) for i in test_list] ## source
            
        self.imgs0 = imgs0
        self.imgs1 = imgs1

    def __len__(self):
        return len(self.imgs0)

    def __getitem__(self, index, visualization=False):

        # --- load image0 ---
        img0_rgb = Image.open(self.imgs0[index])
        if img0_rgb.mode != "RGB":
            img0_rgb = img0_rgb.convert("RGB")
        img0_gray = img0_rgb.convert("L")   # 灰度（1 channel）

        # --- load image1 ---
        img1_rgb = Image.open(self.imgs1[index])
        if img1_rgb.mode != "RGB":
            img1_rgb = img1_rgb.convert("RGB")
        img1_gray = img1_rgb.convert("L")
        
        if self.input_format == 'rgb':
            img0 = img0_rgb
            img1 = img1_rgb
        elif self.input_format == 'gray':
            img0 = img0_gray
            img1 = img1_gray
            
        if self.mode == 'train':
            if self.dataset == 'vis_ir_drone':
                img0 = np.array(img0) ## H W C
                img1 = np.array(img1)
                h0, w0 = img0.shape[:2]
                h1, w1 = img1.shape[:2]
                assert h0 == h1
                assert w0 == w1
                img0 = Image.fromarray(img0[100:-100, 100:-100])
                img1 = Image.fromarray(img1[100:-100, 100:-100])
            if self.dataset == 'googlemap':
                img0 = np.array(img0) ## H W C
                img1 = np.array(img1)
                h0, w0 = img0.shape[:2]
                h1, w1 = img1.shape[:2]
                assert h0 == h1
                assert w0 == w1
                img0 = Image.fromarray(img0[:-100, :])
                img1 = Image.fromarray(img1[:-100, :])
            
            img0, img1 = self.initial_transforms(img0), self.initial_transforms(img1)

            bi = self.bi
            
            if 'glunet_' not in self.dataset:
                ## online generation
                deformation_ratio = float(random.sample(self.deformation_ratio, 1)[0])
                crop_size = int(self.input_resolution[0]/(1-deformation_ratio))
                img0, img1, H_s2t, warped_img1 = randomH(img0, img1, crop_size=crop_size, input_size=self.input_resize, deformation_ratio=deformation_ratio, bi=bi)
            else:
                ## offline generation
                with open(self.H_stg[index], 'r') as json_file:
                    data = json.load(json_file)
                H_s2t = torch.tensor(data['H']).float() ##  3 3
                warped_img1 = img1
                    
            if self.normalize:
                img0, img1 = self.input_norm(img0), self.input_norm(img1)
            
            if self.dataset == 'glunet_448x448_occlusion':
                mask = torch.from_numpy(np.array(Image.open(self.mask[index]))).float()/255.
            else:
                mask = None
                
        elif self.mode == 'val':
            
            w0_original, h0_original = img0.size
            w1_original, h1_original = img1.size
            img0, img1 = self.input_resize(img0), self.input_resize(img1)

            with open(self.H_stg[index], 'r') as json_file:
                data = json.load(json_file)
            H_s2t = torch.tensor(data['H']).float() ##  3 3

            
            H_s2t = torch.diag(torch.tensor([self.input_resolution/w1_original, self.input_resolution/h1_original, 1.])).float() @ \
            H_s2t @ \
            torch.diag(torch.tensor([self.input_resolution/w0_original, self.input_resolution/h0_original, 1.])).float().inverse()
        
            warped_img1 = KGT.warp_perspective(img0.unsqueeze(0), H_s2t.inverse().unsqueeze(0), (self.input_resolution, self.input_resolution), align_corners=True).squeeze(0) #warp img2

            mask = None

        
        sample = {
            "im_A": img1,
            "im_B": img0,
            "im_A_path": self.imgs1[index],
            "im_B_path": self.imgs0[index],
            "H_s2t": H_s2t,
            "warped_img1": warped_img1,
            "dataset_name": self.dataset,
            "original_h": self.input_resolution,
            # "image0_rgb": img1_rgb,
            # "image1_rgb": img0_rgb,
        }

        if mask is not None:
            sample["mask"] = mask

        return sample
          
  