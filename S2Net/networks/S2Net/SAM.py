from torch import nn
import torch
from torch.nn import functional as F

class Self_Adaptive_Aligned_Module(nn.Module):
    def __init__(self, features):
        super(Self_Adaptive_Aligned_Module, self).__init__()
        assert features % 16 == 0, 'base 16 filters'

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.delta_flow_gen = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=features),
            # volumetric out feat: 3 grid
            nn.Conv2d(features, 2, kernel_size=3, padding=1)
        )

        self.aligned_and_upsample_merge_conv = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=1),
        )

    def bilinear_interpolate_torch_gridsample(self, input, size, delta=0):
        out_h, out_w = size

        n, c, h, w = input.shape
        s = 1.0

        # normal
        norm = torch.tensor([[[[h / s, w / s]]]]).type_as(input).to(input.device)

        h_list = torch.linspace(-1.0, 1.0, out_h)
        w_list = torch.linspace(-1.0, 1.0, out_w)

        h_list, w_list = torch.meshgrid(h_list, w_list)
        grid = torch.cat([w_list.unsqueeze(2), h_list.unsqueeze(2)], dim=2)

        # n, d, h, w, c
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)

        # n, d, h, w, c
        delta_permute = delta.permute(0, 2, 3, 1)

        grid = grid + delta_permute / norm

        output = F.grid_sample(input, grid, align_corners=False)
        return output

    def forward(self, high_resolution, low_resolution):
        high_resolution_h, high_resolution_w = high_resolution.size(2), high_resolution.size(3)
        low_resolution_h, low_resolution_w = low_resolution.size(2), low_resolution.size(3)

        assert low_resolution_h == high_resolution_h // 2 and low_resolution_w == high_resolution_w // 2

        low_stage = high_resolution
        high_stage = low_resolution

        h, w = low_stage.size(2), low_stage.size(3)

        # upscale
        high_stage = self.upsample(high_stage)

        concat = torch.cat((low_stage, high_stage), 1)

        # error back propagation  to delta_gen
        delta_flow = self.delta_flow_gen(concat)

        # split
        high_stage_aligned = self.bilinear_interpolate_torch_gridsample(high_stage, (h, w), delta_flow)
        high_stage_upsample = high_stage

        low_stage_final = low_stage
        high_stage_final = self.aligned_and_upsample_merge_conv(torch.cat((high_stage_aligned, high_stage_upsample), 1))

        # concat
        output_alignment_tensor = torch.cat([low_stage_final, high_stage_final], dim=1)

        return output_alignment_tensor


if __name__ == '__main__':
    input1 = torch.randn((16, 16, 224, 224))
    input2 = torch.randn((16, 32, 112, 112))
    SAM = Self_Adaptive_Aligned_Module(32)

    output = SAM(high_resolution=input1, low_resolution=input2)
    print(output.size())









