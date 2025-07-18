import numpy as np
import torch
from tqdm import tqdm

from datasets.homography_dataset_large_size import HomographyDataset
from estimation import demo_estimation

def auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

class MultimodalHomogBenchmark:
    def __init__(self, dataset, input_resolution=448) -> None:
        if "glunet" in dataset:
            self.dataset_name = 'mscoco'
        else:
            self.dataset_name = dataset
        self.dataset = [HomographyDataset(dataset=self.dataset_name,
                                        mode='val',
                                        input_resolution=input_resolution                   
        )]

    def convert_coordinates(self, im_A_coords, im_A_to_im_B, wq, hq, wsup, hsup):
        im_A_coords = (
            np.stack(
                (
                    (wq-1) * (im_A_coords[..., 0] + 1) / 2,
                    (hq-1) * (im_A_coords[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
        )
        im_A_to_im_B = (
            np.stack(
                (
                    (wsup-1) * (im_A_to_im_B[..., 0] + 1) / 2,
                    (hsup-1) * (im_A_to_im_B[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
        )
        return im_A_coords, im_A_to_im_B

    def benchmark(self, model):
        results = {}
        model = model.cuda()
        for idx_dataset, dataset in enumerate(self.dataset):
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, num_workers=1, shuffle=False
            )
            homog_dists = []
            for idx, data in enumerate(tqdm(dataloader)):
                
                im_A_path, im_B_path, H_s2t_path = (
                    data["im_A_path"][0],
                    data["im_B_path"][0],
                    data["H_s2t_path"][0] if 'H_s2t_path' in data else data.get("H_s2t", [None])[0]
                )

                homog_dists.append(demo_estimation(model, im_A_path, im_B_path, H_s2t_path, if_print=False)[0])
                                 
            thresholds = [3, 5, 10, 20]
            aucs = auc(homog_dists, thresholds)
            results.update({f'auc@{t}_{self.dataset_name}': v for t, v in zip(thresholds, aucs)})
            
            homog_dists = np.array(homog_dists)
            results.update({f'mace_{self.dataset_name}': np.mean(homog_dists)})

        return results