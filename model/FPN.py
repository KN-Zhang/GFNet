import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNEncoder(nn.Module):
    def __init__(self, feat_chs, norm_type='BN'):
        super(FPNEncoder, self).__init__()
        self.conv00 = Conv2d(3, feat_chs[0], 7, 1, padding=3, norm_type=norm_type)
        self.conv01 = Conv2d(feat_chs[0], feat_chs[0], 5, 1, padding=2, norm_type=norm_type)

        self.downsample1 = Conv2d(feat_chs[0], feat_chs[1], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv10 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)
        self.conv11 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)

        self.downsample2 = Conv2d(feat_chs[1], feat_chs[2], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv20 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)
        self.conv21 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)

        self.downsample3 = Conv2d(feat_chs[2], feat_chs[3], 3, stride=2, padding=1, norm_type=norm_type)
        self.conv30 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)
        self.conv31 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)

    def forward(self, x):
        conv00 = self.conv00(x)
        conv01 = self.conv01(conv00)
        down_conv0 = self.downsample1(conv01)
        conv10 = self.conv10(down_conv0)
        conv11 = self.conv11(conv10)
        down_conv1 = self.downsample2(conv11)
        conv20 = self.conv20(down_conv1)
        conv21 = self.conv21(conv20)
        down_conv2 = self.downsample3(conv21)
        conv30 = self.conv30(down_conv2)
        conv31 = self.conv31(conv30)

        return [conv01, conv11, conv21, conv31]
    
    
class FPNDecoder_concat(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoder_concat, self).__init__()
        final_ch = feat_chs[-1]
        self.out0 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[-1], kernel_size=1), nn.BatchNorm2d(feat_chs[-1]), Swish())

        self.inner1 = nn.Sequential(nn.Conv2d(feat_chs[-1]+feat_chs[-2], feat_chs[-2], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[-2]), Swish())
        self.out1 = nn.Sequential(nn.Conv2d(feat_chs[-2], feat_chs[-2], 1), nn.BatchNorm2d(feat_chs[-2]), Swish())

        self.inner2 = nn.Sequential(nn.Conv2d(feat_chs[-2]+feat_chs[-3], feat_chs[-3], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[-3]), Swish())
        self.out2 = nn.Sequential(nn.Conv2d(feat_chs[-3], feat_chs[-3], 1), nn.BatchNorm2d(feat_chs[-3]), Swish())

        self.inner3 = nn.Sequential(nn.Conv2d(feat_chs[-3]+feat_chs[-4], feat_chs[-4], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[-4]), Swish())
        self.out3 = nn.Sequential(nn.Conv2d(feat_chs[-4], feat_chs[-4], 1), nn.BatchNorm2d(feat_chs[-4]), Swish())

    def forward(self, conv01, conv11, conv21, conv31):
        intra_feat = conv31
        out0 = self.out0(intra_feat)

        # conv21.shape[2:]
        intra_feat = conv21 + self.inner1(torch.cat((F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=False), conv21), dim=1))
        out1 = self.out1(intra_feat)


        intra_feat = conv11 + self.inner2(torch.cat((F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=False), conv11), dim=1))
        out2 = self.out2(intra_feat)

        intra_feat = conv01 + self.inner3(torch.cat((F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=False), conv01), dim=1))
        out3 = self.out3(intra_feat)

        return [out0, out1, out2, out3]


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, norm_type='IN', **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == 'BN':
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)