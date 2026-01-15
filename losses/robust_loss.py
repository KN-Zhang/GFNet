import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import wandb
import gfnet_configs


def get_gt_warp_homography(H_s2t, img_src, img_tgt, H, W, im_A_coords=None, normalized=True, return_x1_n=False):
    B = img_src.shape[0]
    if im_A_coords is None:
        with torch.no_grad():
            x1_n = torch.meshgrid(
                *[
                    torch.linspace(
                        -1 + 1 / n, 1 - 1 / n, n, device=img_src.device
                    )
                    for n in (B, H, W)
                ]
            )
            x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    else:
        x1_n = im_A_coords.reshape(B, 2, -1).permute(0, 2, 1)
        
    x1 = (x1_n + 1) * (img_src.shape[2]-1) * 0.5
    x2 = kornia.geometry.linalg.transform_points(H_s2t, x1)

    _, _, h, w = img_tgt.shape
    x2_n = (x2 / (h-1)) * 2 - 1
    x2_n = x2_n.reshape(B, H, W, 2)
    mask = torch.logical_and(x2_n<1, x2_n>-1).sum(-1) == 2
    # mask = torch.ones_like(x1_n)[:, :, 0]
    prob = mask.float().reshape(B, H, W)
                
    if normalized:
        if return_x1_n:
            return x1_n.reshape(B, H, W, 2), x2_n, prob
        else:
            return x2_n, prob
    
    else:
        return x2.reshape(B, H, W, 2), prob

class RobustLosses(nn.Module):
    def __init__(
        self,
        ce_weight=0.01,
        local_dist=None,
        local_largest_scale=8,
        depth_interpolation_mode = "bilinear",
        alpha = 1.,
        c = 1e-3,
        iteration_base=0.85,
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.depth_interpolation_mode = depth_interpolation_mode
        self.alpha = alpha
        self.c = c
        self.iteration_base = iteration_base

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        ce_loss = 0.
        reg_loss = 0.
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale        
        for num_itr in flow.keys():
            epe = (flow[num_itr].permute(0,2,3,1).contiguous() - x2).norm(dim=-1)
            if num_itr == len(flow):
                num_pixles = 448. / scale
                pck_05 = (epe[prob > 0.99] < 0.5 * (2/num_pixles)).float().mean()
                wandb.log({f"train_pck_05_scale_{scale}": pck_05}, step = gfnet_configs.cfg.GLOBAL_STEP)

            gt_cert = prob
            ce_loss = ce_loss + self.iteration_base**(len(flow)-num_itr) * F.binary_cross_entropy_with_logits(certainty[num_itr][:, 0], gt_cert)
            

            x = epe[prob > 0.99]
            reg_loss = reg_loss + self.iteration_base**(len(flow)-num_itr) * cs**a * ((x/(cs))**2 + 1**2)**(a/2)
            if not torch.any(reg_loss):
                reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step = gfnet_configs.cfg.GLOBAL_STEP)
        return losses
    
    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1}
        # scale_weights due to differences in scale for regression gradients and classification gradients
        for scale in scales:
            scale_corresps = corresps[scale]
            num_itrs = len(scale_corresps.keys())
            flow = {k: scale_corresps[k]['flow'] for k in scale_corresps.keys()}
            certainty = {k: scale_corresps[k]['certainty'] for k in scale_corresps.keys()}
            
            b, _, h, w = flow[1].shape
            gt_warp, gt_prob = get_gt_warp_homography(batch['H_s2t'], batch['im_A'], batch['im_B'], H=h, W=w)
            

            x2 = gt_warp.float()
            prob = gt_prob
            
            if scale == 'gm':
                loss = self.regression_loss(x2, prob, flow, certainty, 16, mode='gm')
                reg_loss = self.ce_weight * loss[f"gm_certainty_loss_16"] + loss[f"gm_regression_loss_16"]
                tot_loss = tot_loss + scale_weights[16] * reg_loss
                
                prev_epe = (flow[num_itrs].permute(0,2,3,1) - x2).norm(dim=-1).detach()
            else:
                if self.local_largest_scale >= int(scale):
                    prob = prob * (
                            F.interpolate(prev_epe[:, None], size=(h, w), mode="nearest-exact")[:, 0]
                            < (2 / 448) * (self.local_dist[int(scale)] * int(scale)))            

                loss = self.regression_loss(x2, prob, flow, certainty, int(scale), mode='delta')
                
                reg_loss = self.ce_weight * loss[f"delta_certainty_loss_{scale}"] + loss[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[int(scale)] * reg_loss
                
                prev_epe = (flow[num_itrs].permute(0,2,3,1) - x2).norm(dim=-1).detach()
        return tot_loss
