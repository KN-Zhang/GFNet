import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from model.transformer.layers.attention import get_attention_type
from model.transformer.layers.block import CrossBlock
from model.transformer.layers.mlp import Mlp
from model.transformer.layers.swiglu_ffn import SwiGLU


class CrossVITDecoder_noself(nn.Module):
    def __init__(self,
                 conf=None,
                 convtrans=True,
                 upsample=True) -> None:
        super(CrossVITDecoder_noself, self).__init__()

        args = conf
        self.encoder_cfg = args["encoder_cfg"]
            
        self.dino_cfg = args["dino_cfg"]
        self.decoder_cfg = args["dino_cfg"]["decoder_cfg"]
        attention_class = get_attention_type(self.decoder_cfg['attention_type'])
        
        ffn_type = self.decoder_cfg.get("ffn_type", "ffn")
        if ffn_type == "ffn":
            ffn_class = Mlp
        elif ffn_type == "glu":
            ffn_class = SwiGLU
        else:
            raise NotImplementedError(f"Unknown FFN...{ffn_type}")
        
        vit_ch = self.dino_cfg['d_model']
        out_dim = self.encoder_cfg["feat_chs"][0]
        
        self.pe = PositionEncodingSineNorm(d_model=out_dim, max_shape=(128, 128))
        self.cross_attn_blocks = nn.ModuleList()
        
        for _ in range(self.decoder_cfg['num_cross_attn']):
            self.cross_attn_blocks.append(CrossBlock(dim=out_dim, num_heads=self.decoder_cfg['nhead'],
                                                    attn_class=attention_class, ffn_layer=ffn_class, **self.decoder_cfg))

        self.proj = nn.Linear(vit_ch, out_dim, bias=False)
            
    def forward(self, x, y, vit_shape=None):
        B, _, H, W = vit_shape
        x, y = self.proj(x), self.proj(y)
        x = einops.rearrange(self.pe(einops.rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)), 'n c h w -> n (h w) c').contiguous()
        y = einops.rearrange(self.pe(einops.rearrange(y, 'n (h w) c -> n c h w', h=H, w=W)), 'n c h w -> n (h w) c').contiguous()        
        for i in range(len(self.cross_attn_blocks)):
         
            x_new = self.cross_attn_blocks[i](x=x, key=y, value=y)
            y_new = self.cross_attn_blocks[i](x=y, key=x, value=x)
            
            x, y = x_new, y_new


        x_new = x_new.reshape(B, H, W, -1).contiguous().permute(0, 3, 1, 2)
        y_new = y_new.reshape(B, H, W, -1).contiguous().permute(0, 3, 1, 2)

        return x_new, y_new


class PositionEncodingSineNorm(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(128, 128)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        self.d_model = d_model
        self.max_shape = max_shape
        self.pe_dict = dict()

    def reset_pe(self, new_shape, device):
        (H, W) = new_shape
        pe = torch.zeros((self.d_model, H, W))
        y_position = torch.ones((H, W)).cumsum(0).float().unsqueeze(0) * self.max_shape[0] / H
        x_position = torch.ones((H, W)).cumsum(1).float().unsqueeze(0) * self.max_shape[1] / W

        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        return pe.unsqueeze(0).to(device)

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        # if in testing, and test_shape!=train_shape, reset PE
        _, _, H, W = x.shape
        if f"{H}-{W}" in self.pe_dict:  # if the cache has this PE weights, use it
            pe = self.pe_dict[f"{H}-{W}"]
        else:  # or re-generate new PE weights for H-W
            pe = self.reset_pe((H, W), x.device)
            self.pe_dict[f"{H}-{W}"] = pe  # save new PE

        return x + pe  # the shape must be the same