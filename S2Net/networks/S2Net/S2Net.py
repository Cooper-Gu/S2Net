import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvNeXtV2_Block import ConvNeXtV2_Block,LayerNorm
from SWFM import Self_Adaptive_Weighted_Fusion_Module
from SAM import Self_Adaptive_Aligned_Module

def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def conv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """2x2 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


class Bottom(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=1, dropout=0):
        super(Bottom, self).__init__(
            conv2x2(in_channels, out_channels, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
            )

class UBlock(nn.Sequential):
    """Unet mainstream downblock."""
    def __init__(self, inplanes, midplanes, outplanes, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(
            Bottom(inplanes, midplanes, dilation=dilation[0], dropout=dropout),
            Bottom(midplanes, outplanes, dilation=dilation[1], dropout=dropout)
        )

class Encoder(nn.Sequential):
    def __init__(self, channels, layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(ConvNeXtV2_Block(channels))
        super(Encoder, self).__init__(
            *layers
        )

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

class Encoder4(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=1, layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(ConvNeXtV2_Block(out_channels))
        super(Encoder4, self).__init__(
            conv2x2(in_channels, out_channels, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            *layers
        )

class S2Net(nn.Module):
    def __init__(self, num_classes, dims=[32, 64, 128, 256]):
        super(S2Net, self).__init__()

        stem_1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=dims[0], kernel_size=3, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        stem_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=dims[0], kernel_size=3, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.stem_1 = stem_1
        self.stem_2 = stem_2

        self.enconder1_1 = Encoder(channels=dims[0], layer_num=1)
        self.enconder1_2 = Encoder(channels=dims[1], layer_num=1)
        self.enconder1_3 = Encoder(channels=dims[2], layer_num=3)
        self.enconder1_4 = Encoder(channels=dims[3], layer_num=1)

        self.enconder2_1 = Encoder(channels=dims[0], layer_num=1)
        self.enconder2_2 = Encoder(channels=dims[1], layer_num=1)
        self.enconder2_3 = Encoder(channels=dims[2], layer_num=3)
        self.enconder2_4 = Encoder(channels=dims[3], layer_num=1)

        self.down1_1 = Down(in_channels=dims[0], out_channels=dims[1])
        self.down1_2 = Down(in_channels=dims[1], out_channels=dims[2])
        self.down1_3 = Down(in_channels=dims[2], out_channels=dims[3])

        self.down2_1 = Down(in_channels=dims[0], out_channels=dims[1])
        self.down2_2 = Down(in_channels=dims[1], out_channels=dims[2])
        self.down2_3 = Down(in_channels=dims[2], out_channels=dims[3])

        self.fusion1 = Self_Adaptive_Weighted_Fusion_Module(dims[0], is_first=True)
        self.fusion2 = Self_Adaptive_Weighted_Fusion_Module(dims[1], is_first=False)
        self.fusion3 = Self_Adaptive_Weighted_Fusion_Module(dims[2], is_first=False)
        self.fusion4 = Self_Adaptive_Weighted_Fusion_Module(dims[3], is_first=False)

        self.bottom = Bottom(dims[3], dims[2])

        self.SAM_1 = Self_Adaptive_Aligned_Module(features=dims[2])
        self.SAM_2 = Self_Adaptive_Aligned_Module(features=dims[1])
        self.SAM_3 = Self_Adaptive_Aligned_Module(features=dims[0])

        self.decoder3 = UBlock(dims[3], dims[2], dims[1])
        self.decoder2 = UBlock(dims[2], dims[1], dims[0])
        self.decoder1 = UBlock(dims[1], dims[0], dims[0])

        self.outconv = conv1x1(dims[0], num_classes)


    def forward(self, x):
        # backbone
        x1 = torch.cat((x[:, 0, None, :, :], x[:, 1, None, :, :]), dim=1)
        x2 = x[:, 2, None, :, :]

        x1 = self.stem_1(x1)
        x2 = self.stem_2(x2)

        f1_1 = self.enconder1_1(x1)
        d1_1 = self.down1_1(f1_1)

        f1_2 = self.enconder1_2(d1_1)
        d1_2 = self.down1_2(f1_2)

        f1_3 = self.enconder1_3(d1_2)
        d1_3 = self.down1_3(f1_3)

        f1_4 = self.enconder1_4(d1_3)

        f2_1 = self.enconder2_1(x2)
        d2_1 = self.down2_1(f2_1)

        f2_2 = self.enconder2_2(d2_1)
        d2_2 = self.down2_2(f2_2)

        f2_3 = self.enconder2_3(d2_2)
        d2_3 = self.down2_3(f2_3)

        f2_4 = self.enconder2_4(d2_3)

        fusion1 = self.fusion1(0, f1_1, f2_1)
        down_fusion1 = self.down1_1(fusion1)
        fusion2 = self.fusion2(down_fusion1, f1_2, f2_2)
        down_fusion2 = self.down1_2(fusion2)
        fusion3 = self.fusion3(down_fusion2, f1_3, f2_3)
        down_fusion3 = self.down1_3(fusion3)
        fusion4 = self.fusion4(down_fusion3, f1_4, f2_4)

        bottom = self.bottom(fusion4)

        up3_align_concat_tensor = self.SAM_1(high_resolution=fusion3, low_resolution=bottom)
        up3_align_concat_tensor = self.decoder3(up3_align_concat_tensor)
        up3 = up3_align_concat_tensor

        up2_align_concat_tensor = self.SAM_2(high_resolution=fusion2, low_resolution=up3_align_concat_tensor)
        up2_align_concat_tensor = self.decoder2(up2_align_concat_tensor)
        up2 = up2_align_concat_tensor

        up1_align_concat_tensor = self.SAM_3(high_resolution=fusion1, low_resolution=up2_align_concat_tensor)
        up1_align_concat_tensor = self.decoder1(up1_align_concat_tensor)
        out = self.outconv(up1_align_concat_tensor)

        return out


if __name__ == '__main__':
    image  = torch.randn(16, 3, 224, 224)
    target = torch.randn(16, 1, 224, 224)
    model = S2Net(num_classes=2)
    x1 = model(image)

    print(x1.size())


    from thop import profile
    flops, params = profile(model, (image,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
