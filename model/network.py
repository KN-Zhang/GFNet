import math
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from utils.kde import kde
from utils.local_correlation import local_correlation
from utils.utils import get_autocast_params, get_tuple_transform_ops

from model.crossview_decoder_light import CrossVITDecoder_noself
from model.FPN import FPNEncoder, FPNDecoder_concat, Swish

class GFNet(nn.Module):
    def __init__(self,
                 conf,
                 sample_mode = "threshold_balanced", 
                 exact_softmax = False,
                 amp=True,
                 amp_dtype=torch.float16,
                 initial_res=(448, 448),
                 upsample_res=(560, 560),
                 symmetric = False,
                 upsample_preds = False,
                 attenuate_cert=False,
                 ):
        super().__init__()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.sample_mode = sample_mode
        self.exact_softmax = exact_softmax
        self.h_resized, self.w_resized = initial_res
        self.initial_res = initial_res
        self.upsample_preds = upsample_preds
        self.upsample_res = upsample_res
        self.symmetric = symmetric
        self.attenuate_cert = attenuate_cert
        self.sample_thresh = 0.05
        
        args = conf
        self.num_grid = args['matcher']['num_grid']
        ## dino part
        dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )
        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        for param in dinov2_vitl14.parameters():
            param.requires_grad = False        
        self.dino = [dinov2_vitl14]
        self.dino_decoder = CrossVITDecoder_noself(conf=args, upsample=False)
        
        feature_dim = args['encoder_cfg']['feat_chs'] ## coarse to fine
        self.encoder = FPNEncoder(feat_chs=feature_dim[::-1])
        self.decoder = FPNDecoder_concat(feat_chs=feature_dim[::-1])
        
        final_dim = feature_dim[0]
        self.merge_layer = nn.Sequential(nn.Conv2d(2*final_dim, final_dim, kernel_size=3, padding=1), nn.BatchNorm2d(final_dim), Swish())

        radius = args["matcher"]["radius"] ## coarse to fine
        self.radius = radius
        self.num_itr = args["matcher"]["num_itr"] ## coarse to fine
        displacement_dim = args["matcher"]["displacement_dim"] ## coarse to fine
        dw = True
        hidden_blocks = 8
        kernel_size = 5
        displacement_emb = "linear"
        disable_local_corr_grad = True        
        self.conv_refiner = nn.ModuleDict(
            {
                "16": ConvRefiner(
                    2*feature_dim[0] + displacement_dim[0] + (2*radius[0]+1)**2,
                    2*feature_dim[0] + displacement_dim[0] + (2*radius[0]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[0],
                    local_corr_num = radius[0],
                    corr_in_other = True,
                    amp = True,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                ),
                "8": ConvRefiner(
                    2*feature_dim[0] + displacement_dim[1] + (2*radius[1]+1)**2,
                    2*feature_dim[0] + displacement_dim[1] + (2*radius[1]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[1],
                    local_corr_num = radius[1],
                    corr_in_other = True,
                    amp = True,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                ),
                "4": ConvRefiner(
                    2*feature_dim[1] + displacement_dim[2] + (2*radius[2]+1)**2,
                    2*feature_dim[1] + displacement_dim[2] + (2*radius[2]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[2],
                    local_corr_num = radius[2],
                    corr_in_other = True,
                    amp = True,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                ),
                "2": ConvRefiner(
                    2*feature_dim[2] + displacement_dim[3] + (2*radius[3]+1)**2,
                    2*feature_dim[2] + displacement_dim[3] + (2*radius[3]+1)**2,
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks=hidden_blocks,
                    displacement_emb=displacement_emb,
                    displacement_emb_dim=displacement_dim[3],
                    local_corr_num = radius[3],
                    corr_in_other = True,
                    amp = True,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                ),
                "1": ConvRefiner(
                    2*feature_dim[3] + displacement_dim[4],
                    2*feature_dim[3] + displacement_dim[4],
                    2 + 1,
                    kernel_size=kernel_size,
                    dw=dw,
                    hidden_blocks = hidden_blocks,
                    displacement_emb = displacement_emb,
                    displacement_emb_dim = displacement_dim[4],
                    local_corr_num = radius[4],
                    corr_in_other = False,
                    amp = True,
                    disable_local_corr_grad = disable_local_corr_grad,
                    bn_momentum = 0.01,
                ),
            }
        )         
    def extract_features(self, x, upsample=False):
        twoB, C, H, W = x.shape
        vit_h, vit_w = int(H // 14 * 14), int(W // 14 * 14)
        
        if H != vit_h:
            vit_imgs = F.interpolate(x, (vit_h, vit_w), mode='bilinear',
                                     align_corners=False)            
        else:
            vit_imgs = x
        with torch.no_grad():
            if self.dino[0].device != vit_imgs.device:
                self.dino[0] = self.dino[0].to(vit_imgs.device).to(self.amp_dtype)
            dinov2_features_14 = self.dino[0].forward_features(vit_imgs.to(self.amp_dtype))
            features_14 = dinov2_features_14['x_norm_patchtokens']  #.permute(0,2,1).reshape(B,1024,H//14, W//14)
            del dinov2_features_14
        with torch.autocast(device_type="cuda", enabled=self.amp, dtype=self.amp_dtype):
            vit0, vit1 = self.dino_decoder(features_14.chunk(2)[0], features_14.chunk(2)[1], vit_shape=(twoB//2, C, vit_h//14, vit_w//14))        
        vit_feat = torch.cat((vit0.float(), vit1.float()), dim=0)
            
        conv31_h, conv31_w = H // 8, W // 8
        if vit_feat.shape[2] != conv31_h or vit_feat.shape[3] != conv31_w:
            vit_feat_up = F.interpolate(vit_feat, size=(conv31_h, conv31_w), mode='bilinear',
                                        align_corners=False)
        else:
            vit_feat_up = vit_feat
        conv01, conv11, conv21, conv31 = self.encoder(x)
        conv31 = conv31 + self.merge_layer(torch.cat((conv31, vit_feat_up), dim=1))
        feat1, feat2, feat3, feat4 = self.decoder(conv01, conv11, conv21, conv31)
        
        f_q_pyramid = {
            "16": vit_feat.chunk(2)[0],
            "8": feat1.chunk(2)[0],
            "4": feat2.chunk(2)[0],
            "2": feat3.chunk(2)[0],
            "1": feat4.chunk(2)[0],
        }
        f_s_pyramid = {
            "16": vit_feat.chunk(2)[1],
            "8": feat1.chunk(2)[1],
            "4": feat2.chunk(2)[1],
            "2": feat3.chunk(2)[1],
            "1": feat4.chunk(2)[1],
        }
        if upsample:
            del f_q_pyramid["16"], f_s_pyramid["16"]
        return f_q_pyramid, f_s_pyramid
    
    def forward(self, batch, symmetric=False, upsample=False, scale_factor=1, pre_corresps=None, visualization=False):
        im0 = batch["im_A"]
        im1 = batch["im_B"]
        corresps = {}
        B, C, H0, W0 = im0.shape
        B, C, H1, W1 = im1.shape
        
        x = torch.cat([im0, im1], dim=0)
        features0, features1 = self.extract_features(x, upsample)
        all_scales = list(features0.keys())
        if symmetric:
            f_q_pyramid = {
                scale: torch.cat((features0[scale], features1[scale]), dim = 0)
                for scale in features0.keys()
            }
            f_s_pyramid = {
                scale: torch.cat((features1[scale], features0[scale]), dim = 0)
                for scale in features0.keys()
            }            
            features0, features1 = f_q_pyramid, f_s_pyramid
        if upsample:
            num_grid = self.num_grid_up
            num_itr = self.num_itr_up
        else:
            num_grid = self.num_grid
            num_itr = self.num_itr
     
        for idx, scale in enumerate(features0.keys()):
            f0 = features0[scale]
            f1 = features1[scale]
            
            if scale == all_scales[0]:
                if upsample:
                    assert pre_corresps is not None, "you should provide a pre_corresps for upsampling refine."
                    flow, certainty = pre_corresps["flow"], pre_corresps["certainty"]
                    flow = F.interpolate(
                            flow,
                            size=num_grid[0],
                            align_corners=False,
                            mode="bilinear",
                        )
                    certainty = F.interpolate(
                            certainty,
                            size=num_grid[0],
                            align_corners=False,
                            mode="bilinear",
                        )                
                else:
                    corr_volume = self.corr_volume(f0, f1)
                    flow = self.pos_embed(corr_volume) ## B 2 H W
                    certainty = torch.zeros_like(flow)[:, 0][:, None] ## B 1 H W
                
            corresps[scale] = {}
            displacement_pre = torch.zeros_like(flow) + 1e-7
            for itr in range(num_itr[idx]):
                
                delta_flow, delta_certainty, local_corr = self.conv_refiner[scale](
                    num_grid[idx], f0, f1, flow, scale_factor=scale_factor, logits=None,
                )
                displacement = int(scale) * torch.stack((delta_flow[:, 0].float() / (4 * W0),
                                                delta_flow[:, 1].float() / (4 * H0),),dim=1,)
                if not self.training:
                    displacement[((displacement-displacement_pre).abs()/(displacement_pre).abs())<1e-6] = 0
                flow = flow + displacement                
                certainty = certainty + delta_certainty
                corresps[scale][itr+1] = {'flow': flow, 'certainty': certainty}
                displacement_pre = displacement
            
            if scale != '1':
                flow = F.interpolate(
                    flow,
                    size=num_grid[idx+1],
                    mode='bilinear',
                ).detach()
                certainty = F.interpolate(
                    certainty,
                    size=num_grid[idx+1],
                    mode='bilinear',
                ).detach()

        return corresps
    @torch.inference_mode()
    def match(self, im0, im1, *args, batched = True):
        if isinstance(im0, (str, Path)):
            im0, im1 = Image.open(im0).convert("RGB"), Image.open(im1).convert("RGB")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            im0 = ToTensor()(im0)[None].to(device)
            im1 = ToTensor()(im1)[None].to(device)
            imA, imB = im0, im1
            test_transform = get_tuple_transform_ops(
                resize=(self.h_resized, self.w_resized), mode=2, normalize=True, clahe = False
            )
            im0, im1 = test_transform((im0.squeeze(0), im1.squeeze(0)))
            im0 = im0.to(device).unsqueeze(0)
            im1 = im1.to(device).unsqueeze(0)
        elif isinstance(im0, Image.Image):
            batched = False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            im0 = ToTensor()(im0)[None].to(device)
            im1 = ToTensor()(im1)[None].to(device)
            imA, imB = im0, im1
            test_transform = get_tuple_transform_ops(
                resize=(self.h_resized, self.w_resized), normalize=True, clahe = False
            )
            im0, im1 = test_transform((im0.squeeze(0), im1.squeeze(0)))
            im0 = im0.to(device).unsqueeze(0)
            im1 = im1.to(device).unsqueeze(0)                         
        elif isinstance(im0, torch.Tensor):
            imA, imB = im0, im1
            test_transform = get_tuple_transform_ops(
                resize=(self.h_resized, self.w_resized), normalize=True, clahe = False
            )
            im0, im1 = test_transform((im0.squeeze(0), im1.squeeze(0)))            
            batched = False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            im0 = im0.to(device).unsqueeze(0)
            im1 = im1.to(device).unsqueeze(0)
        B,C,H0,W0 = im0.shape
        B,C,H1,W1 = im1.shape
        self.train(False)
        corresps = self.forward({"im_A":im0, "im_B":im1}, symmetric=self.symmetric)

        if self.upsample_preds:
            hs, ws = self.upsample_res
            ## from coarse to fine
            self.num_grid_up = [int(hs/14), 2*int(hs/14), 4*int(hs/14), 8*int(hs/14)]
            # self.num_grid_up = self.num_grid[1:]
            self.radius_up = self.radius[-len(self.num_grid_up):]
            self.num_itr_up = self.num_itr[-len(self.num_grid_up):]
        if self.attenuate_cert:
            low_res_certainty = F.interpolate(
            corresps["16"][self.num_itr[0]]["certainty"], size=(self.num_grid_up[-1], self.num_grid_up[-1]), align_corners=False, mode="bilinear"
            )
            cert_clamp = 0
            factor = 0.5
            low_res_certainty = factor*low_res_certainty*(low_res_certainty < cert_clamp) 
        if self.upsample_preds:
            finest_corresps = corresps["1"][self.num_itr[-1]]
            torch.cuda.empty_cache()
            test_transform = get_tuple_transform_ops(
                resize=(hs, ws), mode=2, normalize=True
            )
            # im_A, im_B = test_transform((Image.open(im_A_path).convert('RGB'), Image.open(im_B_path).convert('RGB')))
            im0, im1 = test_transform((imA.squeeze(0), imB.squeeze(0)))
            im0, im1 = im0[None].to(device), im1[None].to(device)
            scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
            batch = {"im_A": im0, "im_B": im1}            
            corresps = self.forward(batch, pre_corresps=finest_corresps, scale_factor=scale_factor, upsample=True, symmetric=self.symmetric)            

        #return 1,1
        if self.upsample_preds:
            num_grid = self.num_grid_up
            num_itr = self.num_itr_up
        else:
            num_grid = self.num_grid
            num_itr = self.num_itr
        G = num_grid[-1]
        flow = corresps["1"][num_itr[-1]]["flow"].permute(0,2,3,1).reshape(-1,G,G,2)
        certainty = corresps["1"][num_itr[-1]]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
        certainty = certainty.sigmoid()
        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1+1/G,1-1/G, G), 
                torch.linspace(-1+1/G,1-1/G, G), 
                indexing = "xy"), 
            dim = -1).float().to(flow.device).expand(B, G, G, 2)
        if (flow.abs() > 1).any() and True:
            wrong = (flow.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:,None]] = 0
        flow = torch.clamp(flow, -1, 1)        
        
        if self.symmetric:
            A_to_B, B_to_A = flow.chunk(2)
            q_warp = torch.cat((grid, A_to_B), dim=-1)
            s_warp = torch.cat((B_to_A, grid), dim=-1)
            warp = torch.cat((q_warp, s_warp),dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)
        else:
            warp = torch.cat((grid, flow), dim = -1)
        if batched:
            return warp, certainty[:, 0]
        else:
            return warp[0], certainty[0, 0]
    def sample(
        self,
        matches,
        certainty,
        num=5_000,
    ):
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                        num_samples = min(expansion_factor*num, len(certainty)), 
                        replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty 
        use_half = True if matches.device.type == "cuda" else False
        down = 1 if matches.device.type == "cuda" else 8
        density = kde(good_matches, std=0.1, half = use_half, down = down)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                        num_samples = min(num,len(good_certainty)), 
                        replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]                
    def corr_volume(self, feat0, feat1):
        """
            input:
                feat0 -> torch.Tensor(B, C, H, W)
                feat1 -> torch.Tensor(B, C, H, W)
            return:
                corr_volume -> torch.Tensor(B, H, W, H, W)
        """
        B, C, H0, W0 = feat0.shape
        B, C, H1, W1 = feat1.shape
        feat0 = feat0.view(B, C, H0*W0)
        feat1 = feat1.view(B, C, H1*W1)
        corr_volume = torch.einsum('bci,bcj->bji', feat0, feat1).reshape(B, H1, W1, H0 , W0)/math.sqrt(C) #16*16*16
        return corr_volume

    def pos_embed(self, corr_volume: torch.Tensor):
        B, H1, W1, H0, W0 = corr_volume.shape 
        grid = torch.stack(
                torch.meshgrid(
                    torch.linspace(-1+1/W1,1-1/W1, W1), 
                    torch.linspace(-1+1/H1,1-1/H1, H1), 
                    indexing = "xy"), 
                dim = -1).float().to(corr_volume).reshape(H1*W1, 2)
        P = corr_volume.reshape(B,H1*W1,H0,W0).softmax(dim=1) # B, HW, H, W
        pos_embeddings = torch.einsum('bchw,cd->bdhw', P, grid)
        return pos_embeddings
    


class ConvRefiner(nn.Module):
    def __init__(
        self,
        in_dim=6,
        hidden_dim=16,
        out_dim=2,
        dw=False,
        kernel_size=5,
        hidden_blocks=3,
        displacement_emb = None,
        displacement_emb_dim = None,
        local_corr_num = None,
        corr_in_other = None,
        no_im_B_fm = False,
        amp = False,
        concat_logits = False,
        use_bias_block_1 = True,
        use_cosine_corr = False,
        disable_local_corr_grad = False,
        is_classifier = False,
        sample_mode = "bilinear",
        norm_type = nn.BatchNorm2d,
        bn_momentum = 0.1,
        amp_dtype = torch.float16,
    ):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.block1 = self.create_block(
            in_dim, hidden_dim, dw=dw, kernel_size=kernel_size, bias = use_bias_block_1,
        )
        self.hidden_blocks = nn.Sequential(
            *[
                self.create_block(
                    hidden_dim,
                    hidden_dim,
                    dw=dw,
                    kernel_size=kernel_size,
                    norm_type=norm_type,
                )
                for hb in range(hidden_blocks)
            ]
        )
        self.hidden_blocks = self.hidden_blocks
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)
        if displacement_emb:
            self.has_displacement_emb = True
            self.disp_emb = nn.Conv2d(2,displacement_emb_dim,1,1,0)
        else:
            self.has_displacement_emb = False
        self.local_corr_radius = local_corr_num
        self.local_corr_num = local_corr_num
        self.corr_in_other = corr_in_other
        self.no_im_B_fm = no_im_B_fm
        self.amp = amp
        self.concat_logits = concat_logits
        self.use_cosine_corr = use_cosine_corr
        self.disable_local_corr_grad = disable_local_corr_grad
        self.is_classifier = is_classifier
        self.sample_mode = sample_mode
        self.amp_dtype = amp_dtype
        
    def create_block(
        self,
        in_dim,
        out_dim,
        dw=False,
        kernel_size=5,
        bias = True,
        norm_type = nn.BatchNorm2d,
    ):
        num_groups = 1 if not dw else in_dim
        if dw:
            assert (
                out_dim % in_dim == 0
            ), "outdim must be divisible by indim for depthwise"
        conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_groups,
            bias=bias,
        )
        norm = norm_type(out_dim, momentum = self.bn_momentum) if norm_type is nn.BatchNorm2d else norm_type(num_channels = out_dim)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, 1, 1, 0)
        return nn.Sequential(conv1, norm, relu, conv2)
        
    def forward(self, num_grid, x, y, flow, scale_factor = 1, logits = None):
        b,c,hs,ws = x.shape
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, enabled=self.amp, dtype=self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):            
            x_hat = F.grid_sample(y, flow.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)
            if self.has_displacement_emb:
                im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / num_grid, 1 - 1 / num_grid, num_grid, device=x.device),
                    torch.linspace(-1 + 1 / num_grid, 1 - 1 / num_grid, num_grid, device=x.device),
                ), indexing='ij'
                )
                im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
                im_A_coords = im_A_coords[None].expand(b, 2, num_grid, num_grid)
                grid_feature = F.grid_sample(x, im_A_coords.permute(0, 2, 3, 1), align_corners=False, mode = self.sample_mode)                
                in_displacement = flow-im_A_coords
                emb_in_displacement = self.disp_emb(40/32 * scale_factor * in_displacement)

                # Corr in other means take a kxk grid around the predicted coordinate in other image
                if self.corr_in_other:
                    local_corr = local_correlation((b,c,hs,ws), grid_feature, y, local_radius=self.local_corr_radius, num_grid=num_grid, flow = flow, im_A_coords=None,
                                                    sample_mode = self.sample_mode, grid_based_correlation=False)
                    d = torch.cat((grid_feature, x_hat, emb_in_displacement, local_corr), dim=1)
                else:
                    local_corr = None
                    d = torch.cat((grid_feature, x_hat, emb_in_displacement), dim=1)

            d = self.block1(d)
            d = self.hidden_blocks(d)
        d = self.out_conv(d.float())
        displacement, certainty = d[:, :2], d[:, 2:3]
        return displacement, certainty, local_corr
        