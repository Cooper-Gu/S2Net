import torch
import torch.nn as nn

class Self_Adaptive_Weighted_Fusion_Module(nn.Module):
    def __init__(self, in_chan, is_first=False):
        super(Self_Adaptive_Weighted_Fusion_Module, self).__init__()
        self.inchan = in_chan
        self.is_first = is_first
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x0, x1, x2):
        w = torch.sigmoid(self.conv1(torch.cat((x1, x2), dim=1)))
        feat_x1 = torch.mul(x1, w)
        feat_x2 = torch.mul(x2, w)
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2), x1, x2), dim=1))
        if not self.is_first:
            x = self.conv(torch.cat((x0, x), dim=1))
        return x

if __name__ == '__main__':
    input1 = torch.randn((16, 16, 224, 224))
    input2 = torch.randn((16, 16, 224, 224))
    input3 = torch.randn((16, 16, 224, 224))
    SWFM = Self_Adaptive_Weighted_Fusion_Module(16, is_first=True)
    output = SWFM(input1, input2, input3)
    print(output.shape)










