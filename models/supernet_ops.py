"""
refer : https://github.com/skmhrk1209/Single-Path-NAS-PyTorch
"""
import torch
import torch.nn as nn

from models.common import Conv

class SuperConv2d(nn.Module):
    """
    e 랑 k size differentiable search
    EAutoDet 은 threshold 안하고 그냥 weighted sum 하는 거 같은데 나중에 어떻게 refine 하지? => 똑같이 mask, threshold
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=None, stride=1, padding=None, out_channels_list=[], kernel_size_list=[], 
                 dilation=1, groups=1, bias=True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        max_out_channels = max(out_channels_list) if out_channels_list else out_channels
        max_kernel_size = max(kernel_size_list[:-1]) if kernel_size_list else kernel_size  # TODO : -1 하드코딩 케어
        self.padding = padding if padding is not None else max_kernel_size // 2

        channel_masks = []
        # prev_out_channels = None
        for out_channels in out_channels_list:
            # channel_mask = torch.ones(max_out_channels)
            channel_mask = nn.functional.pad(torch.ones(out_channels), [0, max_out_channels - out_channels], value=0)
            # if prev_out_channels:
            #     channel_mask *= nn.functional.pad(torch.zeros(prev_out_channels), [0, max_out_channels - prev_out_channels], value=1)
            channel_mask = channel_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # prev_out_channels = out_channels
            channel_masks.append(channel_mask)

        self.register_buffer('channel_masks', torch.stack(channel_masks, dim=0) if out_channels_list else None)
        self.register_parameter('channel_scores', nn.Parameter(torch.zeros(len(out_channels_list))) if out_channels_list else None)
    

        kernel_masks = []

        for kernel_size in kernel_size_list:
            if kernel_size == "dilated":
                kernel_mask = torch.zeros(max_kernel_size, max_kernel_size)
                kernel_mask[0::2, 0::2] = 1
                kernel_masks.append(kernel_mask.unsqueeze(0).unsqueeze(0))
            else:
                kernel_mask = torch.ones(max_kernel_size, max_kernel_size)
                kernel_mask *= nn.functional.pad(torch.ones(kernel_size, kernel_size), [(max_kernel_size - kernel_size) // 2] * 4, value=0)
                kernel_mask = kernel_mask.unsqueeze(0).unsqueeze(0)
                kernel_masks.append(kernel_mask)

        self.register_buffer('kernel_masks', torch.stack(kernel_masks, dim=0) if kernel_size_list else None)
        # self.register_parameter('kernel_scores', nn.Parameter(torch.zeros(len(kernel_size_list), max_kernel_size, max_kernel_size)) if kernel_size_list else None)
        self.register_parameter('kernel_scores', nn.Parameter(torch.zeros(len(kernel_size_list))) if kernel_size_list else None)

        self.register_parameter('weight', nn.Parameter(torch.Tensor(max_out_channels, in_channels // groups, max_kernel_size, max_kernel_size)))
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        self.register_parameter('bias', nn.Parameter(torch.Tensor(max_out_channels)) if bias else None)
        nn.init.zeros_(self.bias)

        self.max_out_channels = max_out_channels
        self.max_kernel_size = max_kernel_size

    def forward(self, input, prev_ch=None):
        weight = self.weight
        # gumbel softmax
        if self.channel_masks is not None and self.channel_scores is not None:
            scores = nn.functional.gumbel_softmax(self.channel_scores, tau=0.1, hard=True)
            mask = self.channel_masks * scores.view(-1, 1, 1, 1, 1)  # (2, 1,1,1) * (2, 1, 1, 1)
            weight = weight * mask.sum(dim=0)

        if self.kernel_masks is not None and self.kernel_scores is not None:
            mask = self.kernel_masks * torch.sigmoid(self.kernel_scores).view(-1, 1, 1, 1, 1)  # (4,1,1,5,5) * (4,1,1) => (4,1,1,5,5)
            weight = weight * mask.sum(dim=0)

        return nn.functional.conv2d(input, weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

    # def parametrized_mask(self, masks, scores):
    #     mask = masks * scores
    #     return mask.sum(dim=0)

    def freeze_weight(self):
        weight = self.weight
        # if self.channel_masks is not None and self.channel_thresholds is not None:
        #     prev_out_channels = None
        #     for channel_mask, channel_threshold, out_channels in zip(self.channel_masks, self.channel_thresholds, self.out_channels_list):
        #         if prev_out_channels:
        #             channel_norm = torch.norm(self.weight * channel_mask)
        #             if channel_norm < channel_threshold:
        #                 weight = weight[..., :prev_out_channels]
        #                 break
        #         prev_out_channels = out_channels
        # if self.kernel_masks is not None and self.kernel_thresholds is not None:
        #     prev_kernel_size = None
        #     for kernel_mask, kernel_threshold, kernel_size in zip(self.kernel_masks, self.kernel_thresholds, self.kernel_size_list):
        #         if prev_kernel_size:
        #             kernel_norm = torch.norm(self.weight * kernel_mask)
        #             if kernel_norm < kernel_threshold:
        #                 cut = (self.max_kernel_size - prev_kernel_size) // 2
        #                 weight = weight[..., cut:-cut, cut:-cut]
        #                 break
        #         prev_kernel_size = kernel_size

        norms = torch.norm(self.kernel_scores, dim=(1,2))
        idx = torch.argmax(norms)
        # idx=0 -> 1x1, idx=1 -> 3x3, idx=2 -> 5x5, idx=3 -> dilated
        # self.weight = nn.Conv



class SuperBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e_list=[0.5, 0.75, 1.0]):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()

        self.cv1 = SuperConv2d(c1, out_channels_list=[int(e*c1) for e in e_list], kernel_size=1)
        self.cv2 = SuperConv2d(c1, None, kernel_size_list=[1,3,5,"dilated"], out_channels_list=[int(e*c2) for e in e_list])  # search k,d,e
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SuperC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e_list=[0.75, 1.0], bottleneck_e_list=[0.5, 0.75, 1.0]):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * 0.5)  # hidden channels
        self.cv1 = SuperConv2d(c1, out_channels_list=[int(e*c_) for e in e_list], kernel_size=1)  # search e
        self.cv2 = SuperConv2d(c1, out_channels_list=[int(e*c_) for e in e_list], kernel_size=1)  # search e
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)  # TODO : 논문이랑 다르게 구현, yoloV5 기본 따름
        self.m = nn.Sequential(*(SuperBottleneck(c_, c_, shortcut, g, e_list=bottleneck_e_list) for _ in range(n)))

    def forward(self, x):
        out1 = self.m(self.cv1(x))
        out2 = self.cv2(x)
        out = torch.cat((out1, out2), dim=1)
        return self.cv3(out)