import math
from concurrent.futures._base import LOGGER
from typing import Any
import torch

from models.common import *
from torch import Tensor


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=5):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) + self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECASPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(ECASPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        # self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.m = nn.ModuleList([ECALayer(channel=c_, k_size=x) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


##### CBAM #####
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, silu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.silu = nn.SiLU() if silu else None
        # self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.silu is not None:
            x = self.silu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, silu=True)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
##### CBAM #####


##### ECBAM #####
class EfficientChannelGate(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(EfficientChannelGate, self).__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv_2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        branch1 = self.avg_pool(x)
        branch1 = self.conv_1(branch1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        branch2 = self.max_pool(x)
        branch2 = self.conv_2(branch2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(branch1 + branch2)

        return x * y.expand_as(x)


class ECBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ECBAM, self).__init__()
        self.EfficientChannelGate = EfficientChannelGate(in_channels, kernel_size)
        self.SpatialGate = SpatialGate(kernel_size)
    def forward(self, x):
        x_out = self.EfficientChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class ECBAM2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ECBAM2, self).__init__()
        self.EfficientChannelGate = EfficientChannelGate(in_channels)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.EfficientChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out
##### ECBAM #####


##### ECBAM-SPPCSPC #####
class ECBAMSPPCSPC(nn.Module):
    def __init__(self, c1, c2, e=0.5, k=(5, 9, 13)):
        super(ECBAMSPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([ECBAM(in_channels=c_, kernel_size=x) for x in k])
        # Shuffle
        self.shuffle1 = ChannelShuffle(4 * c_, 4)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        # Shuffle
        self.shuffle2 = ChannelShuffle(2 * c_, 4)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = torch.cat([x1] + [m(x1) for m in self.m], 1)
        # Shuffle
        y1 = self.shuffle1(y1)
        y1 = self.cv6(self.cv5(y1))
        y2 = self.cv2(x)
        y2 = torch.cat((y1, y2), dim=1)
        # Shuffle
        y2 = self.shuffle2(y2)
        return self.cv7(y2)
##### ECBAM-SPPCSPC #####

##### PixelShuffle #####
class PixelShuffle(nn.Module):
    r"""Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`.
    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.
    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.
    Args:
        upscale_factor (int): factor to increase spatial resolution by
    Shape:
        - Input: :math:`(N, L, H_{in}, W_{in})` where :math:`L=C \times \text{upscale\_factor}^2`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} \times \text{upscale\_factor}`
          and :math:`W_{out} = W_{in} \times \text{upscale\_factor}`
    Examples::
        # >>> pixel_shuffle = nn.PixelShuffle(3)
        # >>> input = torch.randn(1, 9, 4, 4)
        # >>> output = pixel_shuffle(input)
        # >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """
    __constants__ = ['upscale_factor']
    upscale_factor: int

    def __init__(self, upscale_factor: int) -> None:
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self) -> str:
        return 'upscale_factor={}'.format(self.upscale_factor)
##### PixelShuffle #####

##### StepUp #####
class StepUp(nn.Module):
    def __init__(self, c1, c2):
        super(StepUp, self).__init__()
        self.in_channels = c1
        self.mip = c1 // 2
        self.conv1_1 = Conv(self.mip, self.mip, 1, 1)
        self.conv3_1 = Conv(self.mip, self.mip, 3, 1)
        self.conv3_2 = Conv(self.mip, self.mip, 3, 1)
        self.conv3_3 = Conv(self.mip, self.mip, 3, 1)
        self.conv3_4 = Conv(self.mip, self.mip, 3, 1)
        # Shuffle
        self.shuffle1 = ChannelShuffle(2 * c1, 4)

    def forward(self, x):
        splited = torch.split(x, self.mip, dim=1)
        idenity = x
        branch1 = splited[0]
        branch2 = self.conv1_1(splited[1])
        branch1 = branch1 + branch2
        branch2 = self.conv3_1(branch2)
        branch1 = branch1 + branch2
        branch2 = self.conv3_2(branch2)
        branch1 = branch1 + branch2
        branch2 = self.conv3_3(branch2)
        branch1 = branch1 + branch2
        branch2 = self.conv3_4(branch2)
        branch1 = branch1 + branch2
        y = torch.cat([branch1, branch2, idenity], dim=1)
        # Shuffle
        y = self.shuffle1(y)
        return y
##### StepUp #####

##### ECBAM-RepIdentity #####
class ECBAMRep(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ECBAMRep, self).__init__()
        self.EfficientChannelGate = EfficientChannelGate(in_channels)
        self.repconv1 = RepConv(in_channels, in_channels, 3, 1)
        self.repconv2 = RepConv(in_channels, in_channels, 3, 1)
        self.SpatialGate = SpatialGate()
        self.repconv3 = RepConv(in_channels, in_channels, 3, 1)
        self.repconv4 = RepConv(in_channels, in_channels, 3, 1)

    def forward(self, x):
        identity = x
        att_out = self.EfficientChannelGate(x)
        rep_out = self.repconv2(self.repconv1(x))
        temp = identity * att_out * rep_out
        identity = temp
        att_out = self.SpatialGate(temp)
        rep_out = self.repconv4(self.repconv3(temp))
        y = identity * att_out * rep_out
        return y

class ECBAMRep2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ECBAMRep2, self).__init__()
        self.EfficientChannelGate = EfficientChannelGate(in_channels)
        self.repconv1 = Conv(in_channels, in_channels, 3, 1)
        self.repconv2 = Conv(in_channels, in_channels, 3, 1)
        self.SpatialGate = SpatialGate()
        self.repconv3 = Conv(in_channels, in_channels, 3, 1)
        self.repconv4 = Conv(in_channels, in_channels, 3, 1)

    def forward(self, x):
        identity = x
        att_out = self.EfficientChannelGate(x)
        rep_out = self.repconv2(self.repconv1(x))
        temp = identity * att_out * rep_out
        identity = temp
        att_out = self.SpatialGate(temp)
        rep_out = self.repconv4(self.repconv3(temp))
        y = identity * att_out * rep_out
        return y
##### ECBAM-RepIdentity #####

##### LKA ######
class LKA(nn.Module):
    def __init__(self, dim, kernel=5, p=6, d=3):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel, padding=autopad(kernel, None), groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel, stride=1, padding=p, groups=dim, dilation=d)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
##### LKA ######

##### 自己的SCCS2 #####
class SCCS2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SCCS2, self).__init__()
        self.dim = in_channel // 2
        self.conv11 = Conv(self.dim, self.dim, 1, 1)
        self.branch1 = nn.Sequential(
            Conv(self.dim, self.dim, 1, 1),
            Conv(self.dim, self.dim, 3, 1, p=autopad(3, None), g=self.dim),
            Conv(self.dim, self.dim, 3, 1, p=autopad(3, None), g=self.dim),
        )
        self.branch2 = nn.Sequential(
            Conv(self.dim, self.dim, 1, 1),
            Conv(self.dim, self.dim, 3, 1, p=autopad(3, None), g=self.dim),
            Conv(self.dim, self.dim, 3, 1, p=autopad(3, None), g=self.dim),
        )
        self.conv12 = Conv(self.dim, self.dim, 1, 1)

        self.conv31 = nn.Sequential(
            ChannelShuffle(in_channel, 4),
            Conv(in_channel, in_channel, 3, 1, p=autopad(3, None), g=in_channel),
            Conv(in_channel, in_channel, 3, 1, p=autopad(3, None)),
        )
        self.conv32 = nn.Sequential(
            ChannelShuffle(in_channel, 4),
            Conv(in_channel, in_channel, 3, 1, p=autopad(3, None), g=in_channel),
            Conv(in_channel, in_channel, 3, 1, p=autopad(3, None)),
        )
        self.shuffle = ChannelShuffle(in_channel * 2, 4)
        self.conv13 = Conv(in_channel * 2, in_channel * 2, 1, 1)

    def forward(self, x):
        identity = x
        splited = torch.split(x, self.dim, dim=1)
        x1 = splited[0]
        x2 = splited[1]
        y1 = torch.cat([self.conv11(x1), self.branch2(x2)], dim=1)
        y2 = torch.cat([self.conv12(x1), self.branch1(x2)], dim=1)

        y1 = self.conv31(y1) + identity
        y2 = self.conv32(y2) + identity
        y = torch.cat([y1, y2], dim=1)
        y = self.conv13(self.shuffle(y))

        return y
##### 自己的SCCS2 #####


##### Self-Attention #####
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention
##### Self-Attention #####


##### Fusion_softmax #####
class Fusion_softmax(nn.Module):
    def __init__(self, out_channel, upscale=2):
        super(Fusion_softmax, self).__init__()
        self.upscale = upscale
        if upscale > 1:
            self.pixelshuffle = PixelShuffle(upscale_factor=upscale)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        high, low = x
        if self.upscale > 1:
            high = self.pixelshuffle(high)
        return self.softmax(high) * low + low
##### Fusion_softmax #####


##### BiFusion #####
class BiFusion(nn.Module):
    def __init__(self, in_len=2, epsilon=1e-4):
        super(BiFusion, self).__init__()
        self.epsilon = epsilon
        self.in_len = in_len
        # self.swish = MemoryEfficientSwish()
        self.swish = Swish()
        self.w1 = nn.Parameter(torch.ones(in_len, dtype=torch.float32), requires_grad=True)
        # self.silu = nn.SiLU()

    def forward(self, inputs):
        out = self._forward_fast_attention(inputs)
        return out

    def _forward_fast_attention(self, inputs):
        assert self.in_len == 2 or self.in_len == 3

        in1 = inputs[0]
        in2 = inputs[1]
        if self.in_len == 2:
            w1 = F.relu(self.w1)
            weight = w1 / (torch.sum(w1, dim=0) + self.epsilon)
            out = self.swish(weight[0] * in1 + weight[1] * in2)
        elif self.in_len == 3:
            in3 = inputs[2]
            w1 = F.relu(self.w1)
            weight = w1 / (torch.sum(w1, dim=0) + self.epsilon)
            out = self.swish(weight[0] * in1 + weight[1] * in2 + weight[2] * in3)
        return out

class MBiFusion(nn.Module):
    def __init__(self, in_len=2, epsilon=1e-4):
        super(MBiFusion, self).__init__()
        self.epsilon = epsilon
        self.in_len = in_len
        self.swish = Swish()
        self.w = nn.Parameter(torch.ones(in_len, dtype=torch.float32), requires_grad=True)
        if in_len == 3:
            self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            # self.act2 = nn.ReLU()
            self.swish2 = Swish()
            self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            # self.act3 = nn.ReLU()
            self.swish3 = Swish()
            self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            # self.act4 = nn.ReLU()
            self.swish4 = Swish()

        # self.act = nn.ReLU()

    def forward(self, inputs):
        out = self._forward_fast_attention(inputs)
        return out

    def _forward_fast_attention(self, inputs):
        assert self.in_len == 2 or self.in_len == 3

        in1 = inputs[0]
        in2 = inputs[1]
        if self.in_len == 2:
            w = F.silu(self.w)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            out = self.swish(weight[0] * in1 + weight[1] * in2)
        elif self.in_len == 3:
            in3 = inputs[2]
            w = F.silu(self.w)
            weight = w / (torch.sum(w, dim=0) + self.epsilon)

            w2 = F.silu(self.w2)
            weight2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

            w3 = F.silu(self.w3)
            weight3 = w3 / (torch.sum(w3, dim=0) + self.epsilon)

            w4 = F.silu(self.w4)
            weight4 = w4 / (torch.sum(w4, dim=0) + self.epsilon)

            y2 = self.swish2(weight2[0] * in1 + weight2[1] * in2)
            y3 = self.swish3(weight3[0] * in1 + weight3[1] * in3)
            y4 = self.swish4(weight4[0] * in2 + weight4[1] * in3)

            out = self.swish(weight[0] * y2 + weight[1] * y3 + weight[2] * y4)
        return out


# Adjacent layer fusion
class ALFusion(nn.Module):
    def __init__(self, in_len=2, epsilon=1e-4):
        super(ALFusion, self).__init__()
        self.epsilon = epsilon
        self.in_len = in_len
        self.swish = Swish()
        self.w = nn.Parameter(torch.ones(in_len, dtype=torch.float32), requires_grad=True)
        if in_len == 3:
            self.w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.swish2 = Swish()

            self.w3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.swish3 = Swish()

    def forward(self, inputs):
        out = self._forward_fast_attention(inputs)
        return out

    def _forward_fast_attention(self, inputs):
        assert self.in_len == 2 or self.in_len == 3

        in1 = inputs[0]
        in2 = inputs[1]
        w = F.relu(self.w, False)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        out = self.swish(weight[0] * in1 + weight[1] * in2)
        if self.in_len == 3:
            in3 = inputs[2]

            w2 = F.relu(self.w2, False)
            weight2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

            w3 = F.relu(self.w3, False)
            weight3 = w3 / (torch.sum(w3, dim=0) + self.epsilon)

            out2 = self.swish2(weight2[0] * in2 + weight2[1] * in3)

            out = self.swish3(weight3[0] * out + weight3[1] * out2)
        return out


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
##### BiFusion #####


##### heatmap fusion #####
class HMFusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HMFusion, self).__init__()
        self.conv1 = nn.Sequential(
            Conv(in_channel, in_channel // 2, 3, 1),
            Conv(in_channel // 2, in_channel // 2, 3, 1),
            Conv(in_channel // 2, in_channel // 2, 3, 1),
            Conv(in_channel // 2, out_channel, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        high, low = x
        return self.sigmoid(self.conv1(high)) * low

##### heatmap fusion #####


##### involution #####
class Involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size=3,
                 stride=1,
                 act=True):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.conv1 = nn.Conv2d(channels, channels // reduction_ratio, 1,)
        self.conv2 = nn.Conv2d(channels // reduction_ratio, kernel_size**2 * self.groups, 1, 1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return self.act(self.bn(out))
##### involution #####


##### RepGhost #####
class RepGhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True, reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True
##### RepGhost #####


##### BN Shortcut #####
class ShortcutBN(nn.Module):
    def __init__(self, channel, dimension=0):
        super(ShortcutBN, self).__init__()
        self.d = dimension
        self.bn = nn.BatchNorm2d(channel)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(x[0] + self.bn(x[1]))
##### BN Shortcut ######


##### CBAM Fusion ######
class CBAM_Fusion(nn.Module):
    def __init__(self, channel, reduction_ratio=16, kernel_size=7, pool_types=['avg', 'max']):
        super(CBAM_Fusion, self).__init__()
        # channel att
        self.channel = channel
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channel // reduction_ratio, channel)
        )
        self.pool_types = pool_types

        # Spatial att
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, silu=True)

    def forward(self, x):
        low, high = x[0], x[1]
        # Channel att
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(low, (low.size(2), low.size(3)), stride=(low.size(2), low.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(low, (low.size(2), low.size(3)), stride=(low.size(2), low.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(low, 2, (low.size(2), low.size(3)), stride=(low.size(2), low.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(low)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        c_scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(low)

        # Spatial att
        x_compress = self.compress(low)
        x_out = self.spatial(x_compress)
        s_scale = F.sigmoid(x_out)  # broadcasting

        return high * c_scale * s_scale


class CBAM_Fusion2(nn.Module):
    def __init__(self, channel, reduction_ratio=16, kernel_size=7, pool_types=['avg', 'max']):
        super(CBAM_Fusion2, self).__init__()
        self.cbam1 = CBAM(channel)
        self.cbam2 = CBAM(channel)
        self.BiF = BiFusion()

    def forward(self, x):
        y1 = self.cbam1(x[0])
        y2 = self.cbam2(x[1])
        # return  self.BiF([y1, y2])
        return y1 * y2
##### CBAM Fusion ######


##### Spatial Fusion #####
class SpatialFusion(nn.Module):
    def __init__(self, in_channel, in_len=2, kernel_size=7):
        super(SpatialFusion, self).__init__()
        self.in_len = in_len
        self.Spatial1 = nn.Sequential(
            ChannelPool(),
            BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, silu=True)
        )
        if in_len == 3:
            self.Spatial2 = nn.Sequential(
                ChannelPool(),
                BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, silu=True)
            )
        self.channel = EfficientChannelGate(in_channel)
        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.silu(self.Spatial1(x[0]))
        s_scale1 = x1 * F.sigmoid(x1)  # broadcasting
        y = x[1] * s_scale1
        if self.in_len == 3:
            x2 = self.silu(self.Spatial2(x[2]))
            s_scale2 = x2 * F.sigmoid(x2)
            y = y * s_scale2
        return self.channel(y)
##### Spatial Fusion #####


##### WeightFusion #####
class WeightFusion(nn.Module):
    def __init__(self, in_len=4, epsilon=1e-4):
        super(WeightFusion, self).__init__()
        self.epsilon = epsilon
        self.in_len = in_len
        # self.swish = MemoryEfficientSwish()
        self.swish = Swish()
        self.w1 = nn.Parameter(torch.ones(in_len, dtype=torch.float32), requires_grad=True)
        self.silu = nn.SiLU()

    def forward(self, inputs):
        out = self._forward_fast_attention(inputs)
        return out

    def _forward_fast_attention(self, inputs):
        assert self.in_len == 4 or self.in_len == 6

        in1 = inputs[0]
        in2 = inputs[1]
        in3 = inputs[2]
        in4 = inputs[3]
        w1 = self.silu(self.w1)
        weight = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        if self.in_len == 4:
            out = self.swish(weight[0] * in1 + weight[1] * in2 + weight[2] * in3 + weight[3] * in4)
        elif self.in_len == 6:
            in5 = inputs[4]
            in6 = inputs[5]
            out = self.swish(weight[0] * in1 + weight[1] * in2 + weight[2] * in3 + weight[3] * in4 + weight[4] * in5 + weight[5] * in6)
        return out
##### WeightFusion #####


##### ShortCuts #####
class ShortCuts(nn.Module):
    def __init__(self, in_len=4):
        super(ShortCuts, self).__init__()
        self.in_len = in_len

    def forward(self, inputs):
        in1 = inputs[0]
        in2 = inputs[1]
        in3 = inputs[2]
        in4 = inputs[3]
        out = in1 + in2 + in3 + in4
        if self.in_len == 6:
            in5 = inputs[4]
            in6 = inputs[5]
            out = out + in5 + in6
        return out
##### ShortCuts #####


##### CA #####
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out
##### CA #####


##### SCA #####
class SimpleCoordAttention(nn.Module):
    def __init__(self, inp, oup):
        super(SimpleCoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv_h = nn.Conv2d(inp, 1, kernel_size=1, stride=1)
        self.conv_w = nn.Conv2d(inp, 1, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        a_h = self.conv_h(self.pool_h(x)).sigmoid()
        a_w = self.conv_w(self.pool_w(x)).sigmoid()

        out = x * a_w * a_h + identity

        return out
##### SCA #####


##### SPPCSPC-SelfAttention #####
class SPPCSPCSA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), num_heads=4):
        super(SPPCSPCSA, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)
        # self.cv7 = Conv(4 * c_, c2, 1, 1)
        self.attBranch = nn.Sequential(
            MultiHeadAttLayer(c1, num_heads),
            Conv(c1, c_, 1, 1),
        )
        # self.convBranch = nn.Sequential(
        #     RepConv(c1, c_, 3, 1),
        # )
        # self.identity = Conv(c1, c_, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        # y2 = self.cv2(x)
        # y2 = self.convBranch(x)
        y3 = self.attBranch(x)
        # y4 = self.identity(x)
        # output = self.cv7(torch.cat((y1, y2, y3, y4), dim=1))
        output = self.cv7(torch.cat((y1, y3), dim=1))

        return output
##### SPPCSPC-SelfAttention #####


##### MHAT #####
class MultiHeadAttLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super(MultiHeadAttLayer, self).__init__()
        self.c = c
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        b, _, w, h = x.shape
        x = x.flatten(2)
        x = x.unsqueeze(0)
        x = x.transpose(0, 3)
        x = x.squeeze(3)

        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x

        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c, w, h)
        return x
##### MHAT #####


##### GlobalPooling #####
class GlobalPooling(nn.Module):
    # pp-yolo  eseblock
    def __init__(self, c1):
        super(GlobalPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv2d(c1, c1, 1, 1)
        self.act = nn.Sigmoid()
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, x):
        weight = self.act(self.fc(self.pool(x)))
        return x * weight
        # return self.pool(x).expand_as(x)
##### GlobalPooling #####


##### MBS #####
class MBS(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, shortcut=True, p=None):
        super(MBS, self).__init__()
        # self.pool = nn.AdaptiveMaxPool2d(1)
        self.shortcut = shortcut
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=autopad(k, p))
        self.bn1 = nn.BatchNorm2d(c1)
        if shortcut:
            self.bn2 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.bn1(self.pool(x))
        if self.shortcut:
            out = out + self.bn2(x)
        out = self.act(out)
        return out
##### MBS #####

##### AMBS #####
class AMBS(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, shortcut=True, p=None):
        super(AMBS, self).__init__()
        # self.pool = nn.AdaptiveMaxPool2d(1)
        self.shortcut = shortcut
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=s, padding=autopad(k, p))
        self.avgpool = nn.AvgPool2d(kernel_size=k, stride=s, padding=autopad(k, p))
        self.conv = Conv(c1, c2, k=k, s=s, p=p)

    def forward(self, x):
        out = self.conv(self.maxpool(x) + self.avgpool(x))
        if self.shortcut:
            out = out + x
        return out
##### AMBS #####

##### CMBS #####
class CMBS(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, shortcut=True, p=None):
        super(CMBS, self).__init__()
        self.shortcut = shortcut
        # self.pool = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=k, stride=s, padding=autopad(k, p)),
        #     nn.BatchNorm2d(c1)
        # )
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=autopad(k, p))
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=k, stride=s, padding=autopad(k, p)),
            nn.BatchNorm2d(c1)
        )
        # if shortcut:
        #     self.id = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.conv(x) + self.pool(x)
        if self.shortcut:
            out = out + x
        out = self.act(out)
        return out
##### CMBS #####


##### MSConv #####
class MSConv(nn.Module):
    def __init__(self, c1, c2):
        super(MSConv, self).__init__()
        c_ = c1 // 4
        self.conv1 = Conv(c1, c_, 1, 1)
        # 多尺度特征分支
        self.msconv1 = Conv(c_, c_, 3, 2, 1)
        self.up1 = nn.Upsample(None, scale_factor=2, mode='bilinear')
        self.msconv2 = Conv(c_, c_, 5, 2, 2)
        self.up2 = nn.Upsample(None, scale_factor=2, mode='bilinear')
        self.msconv3 = Conv(c_, c_, 7, 2, 3)
        self.up3 = nn.Upsample(None, scale_factor=2, mode='bilinear')

        self.dc1 = DeformConv2d(c_, c_, 3, 1, 3 // 2)
        self.dc1_offset = nn.Conv2d(c_, 2 * 3 * 3, kernel_size=3, stride=1, padding=3 // 2)
        self.dc1_mask = nn.Conv2d(c_, 3 * 3, kernel_size=3, stride=1, padding=3 // 2)
        self.bn1 = nn.BatchNorm2d(c_)
        self.silu1 = nn.SiLU()

        self.dc2 = DeformConv2d(c_, c_, 5, 1, 5 // 2)
        self.dc2_offset = nn.Conv2d(c_, 2 * 5 * 5, kernel_size=5, stride=1, padding=5 // 2)
        self.dc2_mask = nn.Conv2d(c_, 5 * 5, kernel_size=5, stride=1, padding=5 // 2)
        self.bn2 = nn.BatchNorm2d(c_)
        self.silu2 = nn.SiLU()

        self.dc3 = DeformConv2d(c_, c_, 7, 1, 7 // 2)
        self.dc3_offset = nn.Conv2d(c_, 2 * 7 * 7, kernel_size=7, stride=1, padding=7 // 2)
        self.dc3_mask = nn.Conv2d(c_, 7 * 7, kernel_size=7, stride=1, padding=7 // 2)
        self.bn3 = nn.BatchNorm2d(c_)
        self.silu3 = nn.SiLU()

        self.conv2 = Conv(4 * c_, c_, 1, 1)
        self.conv3 = Conv(c_, c_, 3, 1)

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.conv4 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        out1_1 = self.conv1(x)
        ms1 = self.msconv1(out1_1)
        ms2 = self.msconv2(out1_1)
        ms3 = self.msconv3(out1_1)
        deconv1 = self.silu1(self.bn1(self.dc1(ms1, self.dc1_offset(ms1), self.dc1_mask(ms1))))
        deconv2 = self.silu2(self.bn2(self.dc2(ms2, self.dc2_offset(ms2), self.dc2_mask(ms2))))
        deconv3 = self.silu3(self.bn3(self.dc3(ms3, self.dc3_offset(ms3), self.dc3_mask(ms3))))
        out1_2 = self.up1(deconv1)
        out1_3 = self.up2(deconv2)
        out1_4 = self.up3(deconv3)
        out1 = self.conv3(self.conv2(torch.cat((out1_2, out1_3, out1_4, out1_1), dim=1)))

        out2 = self.conv4(self.pool(x).expand_as(x))

        out = self.outConv(torch.cat((out1, out2), dim=1))
        return out
##### MBS #####


##### DeConv ASPP（DeConv Py Module） #####
class DCPM(nn.Module):
    def __init__(self, c1):
        super(DCPM, self).__init__()
        c_ = c1 // 4
        self.conv1 = Conv(c1, c_, 1, 1)
        # 多尺度特征分支
        self.msconv1 = Conv(c_, c_, 3, 1)
        self.msconv2 = Conv(c_, c_, 5, 1)
        self.msconv3 = Conv(c_, c_, 7, 1)

        self.dc1 = DeformConv2d(c_, c_, 3, 1, 3 // 2)
        self.dc1_offset = nn.Conv2d(c_, 2 * 3 * 3, kernel_size=3, stride=1, padding=3 // 2)
        self.dc1_mask = nn.Conv2d(c_, 3 * 3, kernel_size=3, stride=1, padding=3 // 2)
        self.bn1 = nn.BatchNorm2d(c_)
        self.silu1 = nn.SiLU()

        self.dc2 = DeformConv2d(c_, c_, 5, 1, 5 // 2)
        self.dc2_offset = nn.Conv2d(c_, 2 * 5 * 5, kernel_size=5, stride=1, padding=5 // 2)
        self.dc2_mask = nn.Conv2d(c_, 5 * 5, kernel_size=5, stride=1, padding=5 // 2)
        self.bn2 = nn.BatchNorm2d(c_)
        self.silu2 = nn.SiLU()

        self.dc3 = DeformConv2d(c_, c_, 7, 1, 7 // 2)
        self.dc3_offset = nn.Conv2d(c_, 2 * 7 * 7, kernel_size=7, stride=1, padding=7 // 2)
        self.dc3_mask = nn.Conv2d(c_, 7 * 7, kernel_size=7, stride=1, padding=7 // 2)
        self.bn3 = nn.BatchNorm2d(c_)
        self.silu3 = nn.SiLU()

        self.conv2 = Conv(4 * c_, c_, 1, 1)         # 多尺度Concat融合后降低通道数
        self.conv3 = Conv(c_, c_, 3, 1)

        # 全局特征分支
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.maxpool = nn.AdaptiveMaxPool2d(1)

        # self.sigmoid = nn.Sigmoid()
        self.poolconv = Conv(c1, c_, 1, 1)      # 全局Concat后降低通道数

        self.outConv = Conv(2 * c_, c1, 1, 1)       # 全部Concat后输出

    def forward(self, x):
        out = self.conv1(x)
        # 多尺度
        ms1 = self.msconv1(out)
        ms2 = self.msconv2(out)
        ms3 = self.msconv3(out)
        deconv3 = self.silu3(self.bn3(self.dc3(ms3, self.dc3_offset(ms3), self.dc3_mask(ms3))))
        deconv2 = self.silu2(self.bn2(self.dc2(ms2, self.dc2_offset(ms2), self.dc2_mask(ms2))))
        deconv1 = self.silu1(self.bn1(self.dc1(ms1, self.dc1_offset(ms1), self.dc1_mask(ms1))))
        # 全局
        globalout = self.avgpool(x) + self.maxpool(x)
        poolout = self.poolconv(globalout.expand_as(x))
        # 局部和多尺度
        msout = self.conv3(self.conv2(torch.cat((out, deconv1, deconv2, deconv3), dim=1)))
        # 再乘全局
        # out = out * pool.expand_as(out)

        out = self.outConv(torch.cat((msout, poolout), dim=1))
        return out
##### DeConv ASPP #####


##### CCABlock #####
class CCABlock(nn.Module):
    def __init__(self, c1, c2):
        super(CCABlock, self).__init__()
        self.cca = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 1, 1),
            nn.BatchNorm2d(c2),
            CoordAttention(c2, c2),
            nn.Conv2d(c2, c2, 1, 1),
            # nn.BatchNorm2d(c2),
            nn.Sigmoid(),
        )

        self.act = nn.SiLU()

    def forward(self, x):
        weight = self.cca(x)
        out = x + weight
        return self.act(out)
##### CCABlock #####


##### AsyInception #####
class AsyInception(nn.Module):
    def __init__(self, c1, c2):
        super(AsyInception, self).__init__()
        c_ = c1 // 2
        self.conv1_1 = Conv(c1, c_, 1, 1)
        self.conv5_5 = nn.Sequential(
            Conv(c_, c_, (1, 5), 1),
            Conv(c_, c_, (5, 1), 1)
        )

        self.conv7_7 = nn.Sequential(
            Conv(c_, c_, (1, 7), 1),
            Conv(c_, c_, (7, 1), 1)
        )

        self.conv9_9 = nn.Sequential(
            Conv(c_, c_, (1, 9), 1),
            Conv(c_, c_, (9, 1), 1)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1_2 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(5 * c_, c2, 1, 1)

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv5_5 = self.conv5_5(conv1_1)
        conv7_7 = self.conv7_7(conv1_1 + conv5_5)
        conv9_9 = self.conv9_9(conv1_1 + conv7_7)
        pool = self.conv1_2(self.pool(x).expand_as(x))
        out = self.outConv(torch.cat([conv1_1, conv5_5, conv7_7, conv9_9, pool], dim=1))
        return out
##### AsyInception #####


class SPPFGCSPC2(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 5, 5)):
        super(SPPFGCSPC2, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_ // 2, 1, 1)
        self.mp1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.mp2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.mp3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.mp4 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

        self.glbf = GlobalPooling(c1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv3(x1)
        x3 = self.cv4(x2)
        x4 = self.mp1(x3)
        x5 = self.mp2(x4)
        x6 = self.mp3(x5)
        # x7 = self.mp4(x6)
        y1 = self.cv6(self.cv5(torch.cat([x1, x2, x3, x4, x5, x6], dim=1)))
        y2 = self.cv2(self.glbf(x))
        output = self.cv7(torch.cat((y1, y2), dim=1))

        return output


class SPPFGCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPFGCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.cv4 = Conv(c_, c_ // 2, 3, 1)
        self.cv5 = Conv(c_ // 2, c_ // 2, 3, 1)
        self.cv6 = Conv(c_ // 2, c_ // 2, 3, 1)
        self.cv7 = Conv(c_ // 2, c_ // 2, 3, 1)
        self.cv8 = Conv(4 * c_, c2, 1, 1)
        self.glbf = GlobalPooling(c1)

    def forward(self, x):
        x1 = self.cv1(x)
        y1_0 = self.cv3(torch.cat([x1] + [m(x1) for m in self.m], 1))
        y1_1 = self.cv4(y1_0)
        y1_2 = self.cv5(y1_1)
        y1_3 = self.cv6(y1_2)
        y1_4 = self.cv7(y1_3)
        y2 = self.cv2(self.glbf(x))
        output = self.cv8(torch.cat((y1_0, y1_1, y1_2, y1_3, y1_4, y2), dim=1))

        return output


##### ECCA #####  ECALayer and CoordAttention
class ECCA(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(ECCA, self).__init__()
        # CoordAttention
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(16, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # ECALayer
        self.eca_h_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_h_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.eca_h_sig = nn.Sigmoid()

        self.eca_w_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_w_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.eca_w_sig = nn.Sigmoid()

        # Spatial Attention

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)

        a_h = x_h * self.eca_h_sig(self.eca_h_conv(self.eca_h_pool(x_h).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)).expand_as(x_h)
        a_w = x_w * self.eca_w_sig(self.eca_w_conv(self.eca_w_pool(x_w).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)).expand_as(x_w)

        a_w = a_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(a_h).sigmoid()
        a_w = self.conv_w(a_w).sigmoid()

        out = identity * a_w * a_h

        return out
##### ECCA #####


##### ECCA2 #####  ECALayer and CoordAttention
class ECCA2(nn.Module):
    def __init__(self, inp, oup, reduction=16):
        super(ECCA2, self).__init__()
        # CoordAttention
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(16, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # ECALayer
        self.eca_h_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_h_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.eca_h_sig = nn.Sigmoid()

        self.eca_w_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_w_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.eca_w_sig = nn.Sigmoid()

        self.conv = Conv(inp, oup, 1, 1, 0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        x_h = x_h * self.eca_h_sig(
            self.eca_h_conv(
                self.eca_h_pool(x_h).squeeze(-1).transpose(-1, -2)
            ).transpose(-1, -2).unsqueeze(-1)
        ).expand_as(x_h)
        x_w = x_w * self.eca_w_sig(
            self.eca_w_conv(
                self.eca_w_pool(x_w).squeeze(-1).transpose(-1, -2)
            ).transpose(-1, -2).unsqueeze(-1)
        ).expand_as(x_w)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)

        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = self.conv(x) + identity * a_w * a_h

        return out
##### ECCA2 #####

##### LCA #####  CoordAttention
class LCA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(LCA, self).__init__()
        # CoordAttention
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.eca_h_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_h_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.eca_h_sig = nn.Sigmoid()

        self.eca_w_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_w_conv = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.eca_w_sig = nn.Sigmoid()

        self.h_conv = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.h_sig = nn.Sigmoid()
        self.w_conv = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, bias=False)
        self.w_sig = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        x_h = x_h * self.eca_h_sig(self.eca_h_conv(self.eca_h_pool(x_h).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
        x_w = x_w * self.eca_w_sig(self.eca_w_conv(self.eca_w_pool(x_w).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))

        y_h = self.h_sig(self.h_conv(x_h))
        y_w = self.w_sig(self.w_conv(x_w.permute(0, 1, 3, 2)))

        out = identity * y_w * y_h
        return out
##### ECCA #####


##### SplitConv #####
class SplitConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, group=1, reduction=2):
        super(SplitConv, self).__init__()
        self.c = c1 // reduction
        self.conv1 = Conv(self.c, self.c, k=k, s=s, p=p, g=group)
        self.conv2 = Conv(self.c, self.c, k=k, s=s, p=p, g=group)
        self.shuffle = ChannelShuffle(c1, 2)

        self.conv3 = Conv(self.c, self.c, k=k, s=s, p=p, g=group)

    def forward(self, x):
        splited = torch.split(x, self.c, dim=1)
        identity, y = splited[0], splited[1]
        y = self.conv2(self.conv1(y))
        identity = self.conv3(identity)
        y = torch.cat([identity, y], dim=1)
        out = self.shuffle(y)
        return out
##### SplitConv #####


##### MS #####
class MS(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super(MS, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=autopad(k, p))
        self.conv = Conv(c1, 1, k, s, p)
        self.act = nn.Sigmoid()
    def forward(self, x):
        out = self.act(self.conv(self.pool(x)))
        out = x * out.expand_as(x)
        return out
##### MS #####


##### Better Coordinate Attention #####
class BCoodAttention(nn.Module):
    def __init__(self, c1, c2):
        super(BCoodAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c1, 1, 1, 1)
        self.conv2 = nn.Conv2d(c1, 1, 1, 1)
        self.softmax_h = nn.Softmax(dim=2)
        self.softmax_w = nn.Softmax(dim=3)
        self.conv3 = Conv(1, c2, 1, 1)

    def forward(self, x):
        x_h = self.conv1(self.pool_h(x))
        x_w = self.conv2(self.pool_w(x))
        h_weight = self.softmax_h(x_h)
        w_weight = self.softmax_w(x_w)
        att_map = self.conv3(h_weight * w_weight)
        out = x * att_map
        return out
##### Better Coordinate Attention #####


##### Better Coordinate Attention #####
class BCoodAttention2(nn.Module):
    def __init__(self, c1, c2):
        super(BCoodAttention2, self).__init__()
        self.conv1 = nn.Conv2d(c1, 1, 1, 1)
        self.softmax_h = nn.Softmax(dim=2)
        self.softmax_w = nn.Softmax(dim=3)
        self.conv3 = Conv(1, c2, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        h_weight = self.softmax_h(y)
        w_weight = self.softmax_w(y)
        att_map = self.conv3(y * h_weight * w_weight)
        out = x + att_map
        return out
##### Better Coordinate Attention #####


##### Attention #####
class CSAttention(nn.Module):
    def __init__(self, c1, c2, reduction=2):
        super(CSAttention, self).__init__()
        c_ = c1 // reduction
        self.conv1_1 = Conv(c1, c_, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv1_2 = Conv(c_ * 3, c_, 1, 1)

        self.gpool = nn.AdaptiveAvgPool2d(1)
        # self.conv1d = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Conv2d(c_, c_, 1, 1)
        self.sigmoid = nn.Sigmoid()

        self.conv2 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        b1 = self.conv1_1(x)
        b2 = self.conv2(x)
        b1 = self.conv1_2(torch.cat([self.maxpool(b1), self.avgpool(b1), b1], dim=1))
        # ch_w = self.sigmoid(self.conv1d(self.gpool(b1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
        ch_w = self.sigmoid(self.fc(self.gpool(b1)))
        b1 = b1 * ch_w
        out = self.outConv(torch.cat([b1, b2], dim=1))
        return out
##### Attention #####


##### Attention #####
class CSAttention2(nn.Module):
    def __init__(self, c1, c2, reduction=2):
        super(CSAttention2, self).__init__()
        c_ = c1 // reduction
        self.conv1_1 = Conv(c1, c_, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv1_2 = Conv(c_ * 3, c_, 1, 1)

        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2)
        # self.fc = nn.Conv2d(c_, c_, 1, 1)
        self.sigmoid = nn.Sigmoid()

        self.conv2 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        b1 = self.conv1_1(x)
        b2 = self.conv2(x)
        b1 = self.conv1_2(torch.cat([self.maxpool(b1), self.avgpool(b1), b1], dim=1))
        ch_w = self.sigmoid(self.conv1d(self.gpool(b1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
        # ch_w = self.sigmoid(self.fc(self.gpool(b1)))
        b1 = b1 * ch_w
        out = self.outConv(torch.cat([b1, b2], dim=1))
        return out
##### Attention #####


##### Attention #####
class CSAttentionC(nn.Module):
    def __init__(self, c1, c2, reduction=2):
        super(CSAttentionC, self).__init__()
        c_ = c1 // reduction
        self.conv1_1 = Conv(c1, c_, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv1_2 = nn.Conv2d(c_ * 2, c_, 1, 1)
        # self.conv1_2 = nn.Conv2d(c_ * 2, c_, kernel_size=3, stride=1, padding=1, groups=c_)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = Conv(c1, c_, 1, 1)

        self.conv1_3 = Conv(c_ * 2, c_, 1, 1)
        # self.conv1_3 = Conv(c_ * 2, c_, k=3, s=1, p=1, g=c_)

        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c_, c_, 1, 1)
        self.sigmoid2 = nn.Sigmoid()

        self.conv3 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        b1 = self.conv1_1(x)
        b2 = self.conv2(x)
        b3 = self.conv3(x)
        b1 = b1 * self.sigmoid1(self.conv1_2(torch.cat([self.maxpool(b1), self.avgpool(b1)], dim=1)))
        b1 = self.conv1_3(torch.cat([b1, b2], dim=1))
        ch_w = self.sigmoid2(self.fc(self.gpool(b1)))
        b1 = b1 * ch_w
        out = self.outConv(torch.cat([b1, b3], dim=1))
        return out
##### Attention #####


##### Attention #####
class CSAttentionL(nn.Module):
    def __init__(self, c1, c2, reduction=2):
        super(CSAttentionL, self).__init__()
        c_ = c1 // reduction
        self.conv1_1 = Conv(c1, c_, 1, 1)
        self.conv1_12 = Conv(c_, c_, k=3, s=1, g=c_)
        self.conv1_13 = Conv(c_, c_, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # self.conv1_2 = nn.Conv2d(c_ * 2, c_, 1, 1)
        self.conv1_2 = nn.Conv2d(c_ * 2, c_, kernel_size=3, stride=1, padding=1, groups=c_)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = Conv(c1, c_, 1, 1)

        # self.conv1_3 = Conv(c_ * 2, c_, 1, 1)
        self.conv1_3 = Conv(c_ * 2, c_, k=3, s=1, p=1, g=c_)

        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c_, c_, 1, 1)
        self.sigmoid2 = nn.Sigmoid()

        self.conv3 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        b1 = self.conv1_13(self.conv1_12(self.conv1_1(x)))
        b2 = self.conv2(x)
        b3 = self.conv3(x)
        b1 = b1 * self.sigmoid1(self.conv1_2(torch.cat([self.maxpool(b1), self.avgpool(b1)], dim=1)))
        b1 = self.conv1_3(torch.cat([b1, b2], dim=1))
        ch_w = self.sigmoid2(self.fc(self.gpool(b1)))
        b1 = b1 * ch_w
        out = self.outConv(torch.cat([b1, b3], dim=1))
        return out
##### Attention #####


##### Attention #####
class CSAttentionL(nn.Module):
    def __init__(self, c1, c2, reduction=2):
        super(CSAttentionL, self).__init__()
        c_ = c1 // reduction
        self.conv1_1 = Conv(c1, c_, 1, 1)
        self.conv1_12 = Conv(c_, c_, k=3, s=1, g=c_)
        self.conv1_13 = Conv(c_, c_, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # self.conv1_2 = nn.Conv2d(c_ * 2, c_, 1, 1)
        self.conv1_2 = nn.Conv2d(c_ * 2, c_, kernel_size=3, stride=1, padding=1, groups=c_)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = Conv(c1, c_, 1, 1)

        # self.conv1_3 = Conv(c_ * 2, c_, 1, 1)
        self.conv1_3 = Conv(c_ * 2, c_, k=3, s=1, p=1, g=c_)

        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c_, c_, 1, 1)
        self.sigmoid2 = nn.Sigmoid()

        self.conv3 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        b1 = self.conv1_13(self.conv1_12(self.conv1_1(x)))
        b2 = self.conv2(x)
        b3 = self.conv3(x)
        b1 = b1 * self.sigmoid1(self.conv1_2(torch.cat([self.maxpool(b1), self.avgpool(b1)], dim=1)))
        b1 = self.conv1_3(torch.cat([b1, b2], dim=1))
        ch_w = self.sigmoid2(self.fc(self.gpool(b1)))
        b1 = b1 * ch_w
        out = self.outConv(torch.cat([b1, b3], dim=1))
        return out
##### Attention #####



##### Little Attention #####
class LCSAttention(nn.Module):
    def __init__(self, c1, c2, reduction=2):
        super(LCSAttention, self).__init__()
        self.c1 = c1
        c_ = c1 // reduction

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv1_2 = Conv(c_ * 3, c_, 1, 1)

        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.outConv = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        y = torch.split(x, self.c1 // 2, dim=1)
        b1 = y[0]
        b2 = y[1]
        b1 = self.conv1_2(torch.cat([self.maxpool(b1), self.avgpool(b1), b1], dim=1))
        ch_w = self.sigmoid(self.conv1d(self.gpool(b1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
        b1 = b1 * ch_w
        out = self.outConv(torch.cat([b1, b2], dim=1))
        return out
##### Attention #####


##### Attention #####
class CSAttentionM(nn.Module):
    def __init__(self, c1, c2, reduction=2):
        super(CSAttentionM, self).__init__()
        c_ = c1 // reduction
        self.conv1_1 = Conv(c1, c_, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.maxconv = nn.Conv2d(c_, c_, 3, 1, 1, groups=c_)
        self.maxsig = nn.Sigmoid()

        self.avgconv = nn.Conv2d(c_, c_, 3, 1, 1, groups=c_)
        self.avgsig = nn.Sigmoid()

        self.conv1_2 = Conv(c_ * 2, c_, 1, 1)

        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c_, c_, 1, 1)
        self.sigmoid2 = nn.Sigmoid()

        self.conv2 = Conv(c1, c_, 1, 1)

        self.outConv = Conv(c_ * 2, c2, 1, 1)

    def forward(self, x):
        b1 = self.conv1_1(x)
        b2 = self.conv2(x)
        max_w = self.maxsig(self.maxconv(self.maxpool(b1)))
        avg_w = self.avgsig(self.avgconv(self.avgpool(b1)))
        spatial = b1 * avg_w * max_w
        b1 = self.conv1_2(torch.cat([b1, spatial], dim=1))
        ch_w = self.sigmoid2(self.fc(self.gpool(b1)))
        b1 = b1 * ch_w
        out = self.outConv(torch.cat([b1, b2], dim=1))
        return out
##### Attention #####


##### CSCBAM #####
class CSCBAM(nn.Module):
    def __init__(self, c1, c2, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CSCBAM, self).__init__()
        self.c_ = c1 // 2
        # self.conv1 = Conv(c1, c_, 1, 1)
        # self.conv2 = Conv(c1, c_, 1, 1)
        self.ChannelGate = ChannelGate(self.c_, reduction_ratio, pool_types)
        # self.ese = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(c_, c_, 1, 1),
        #     nn.Sigmoid()
        # )
        self.SpatialGate = SpatialGate()
        self.shuffle = ChannelShuffle(channels=4 * self.c_, groups=4)
        self.conv3 = Conv(4 * self.c_, c2, 1, 1)
    def forward(self, x):
        splited = torch.split(x, self.c_, dim=1)
        # x1 = self.conv1(x)
        # x2 = self.conv2(x)
        x1 = splited[0]
        x2 = splited[1]

        x1_1 = self.ChannelGate(x1)
        x1_2 = self.SpatialGate(x1_1)

        out = self.shuffle(torch.cat([x1, x1_1, x1_2, x2], dim=1))
        out = self.conv3(out)
        return out
##### CSCBAM #####


##### Channel and Coordinate Attention #####
class EFCCA(nn.Module):
    def __init__(self, c1, gama=2, b=1):
        super(EFCCA, self).__init__()
        # Channel Attention
        self.c_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((math.log(c1, 2) + b) / gama))
        k = t if t % 2 else t + 1
        k = max(5, k)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=(k - 1) // 2, bias=False)
        self.c_sig = nn.Sigmoid()
        # Spatial Coordinate Attention
        self.s_gap_h = nn.AdaptiveAvgPool2d((None, 1))
        self.s_gap_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.s_sig = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        y = self.c_avg_pool(x) + self.c_max_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = x * self.c_sig(y).expand_as(x)

        # Spatial Coordinate Attention
        h = self.s_gap_h(torch.max(y, 1)[0].unsqueeze(1))
        w = self.s_gap_w(torch.max(y, 1)[0].unsqueeze(1))
        y = y * self.s_sig(self.conv(h * w)).expand_as(y)
        return y
##### Channel and Coordinate Attention #####


##### Channel and Coordinate Attention #####
class EFCCA2(nn.Module):
    def __init__(self, c1, gama=2, b=1):
        super(EFCCA2, self).__init__()
        # Channel Attention
        self.c_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((math.log(c1, 2) + b) / gama))
        k = t if t % 2 else t+1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=(k - 1) // 2, bias=False)
        self.c_sig = nn.Sigmoid()
        # Spatial Coordinate Attention
        self.s_conv = Conv(c1, 1, 1, 1)
        self.s_gap_h = nn.AdaptiveAvgPool2d((None, 1))
        self.s_gap_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
        self.s_sig = nn.Sigmoid()

    def forward(self, x):
        identity = x
        # Channel Attention
        y = self.c_avg_pool(x) + self.c_max_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = x * self.c_sig(y).expand_as(x)

        # Spatial Coordinate Attention
        c_ = self.s_conv(y)
        h = self.s_gap_h(c_)
        w = self.s_gap_w(c_)
        y = y * self.s_sig(self.conv(h * w)).expand_as(y)
        return y + identity
##### Channel and Coordinate Attention #####

##### Channel and Coordinate Attention #####
class EFCCA3(nn.Module):
    def __init__(self, c1, gama=2, b=1):
        super(EFCCA3, self).__init__()
        # Channel Attention
        self.c_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((math.log(c1, 2) + b) / gama))
        k = t if t % 2 else t + 1
        k = max(5, k)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=(k - 1) // 2, bias=False)
        self.c_sig = nn.Sigmoid()
        # Spatial Coordinate Attention
        self.s_gap_h = nn.AdaptiveAvgPool2d((None, 1))
        self.s_gap_w = nn.AdaptiveAvgPool2d((1, None))
        self.cbs = Conv(1, 1, k=7, s=1, p=3)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.s_sig = nn.Sigmoid()

    def forward(self, x):
        identity = x
        # Channel Attention
        y = self.c_avg_pool(x) + self.c_max_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = x * self.c_sig(y).expand_as(x)

        # Spatial Coordinate Attention
        h = self.s_gap_h(torch.max(y, 1)[0].unsqueeze(1))
        w = self.s_gap_w(torch.max(y, 1)[0].unsqueeze(1))
        y = y * self.s_sig(self.conv(self.cbs(h * w))).expand_as(y)
        return y + identity
##### Channel and Coordinate Attention #####


##### More Efficient Channel and Spatial Coordinate Attention #####
class MECSCA(nn.Module):
    def __init__(self, c1):
        super(MECSCA, self).__init__()
        assert not bool(math.sqrt(c1)-int(math.sqrt(c1)))   # c1 has to satisfy c1 = int * int
        # Channel Attention
        self.c_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_max_pool = nn.AdaptiveMaxPool2d(1)
        self.wh = int(math.sqrt(c1))
        t = int(abs((math.log(c1, 2) + 1)))
        k = t if t % 2 else t + 1
        self.c_conv = nn.Conv2d(1, 1, kernel_size=k, stride=1, padding=(k - 1) // 2)
        self.c_sig = nn.Sigmoid()
        # Spatial Coordinate Attention
        self.s_gap_h = nn.AdaptiveAvgPool2d((None, 1))
        self.s_gap_w = nn.AdaptiveAvgPool2d((1, None))
        self.s_conv = nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3)
        self.s_sig = nn.Sigmoid()

    def forward(self, x):
        identity = x
        # Channel Attention
        y = self.c_avg_pool(x) + self.c_max_pool(x)
        b, c, w, h = y.shape
        y = self.c_conv(y.view(b, 1, self.wh, self.wh)).view(b, self.wh * self.wh, 1, 1)
        y = x * self.c_sig(y).expand_as(x)

        # Spatial Coordinate Attention
        cmax = torch.max(y, 1)[0].unsqueeze(1)
        h = self.s_gap_h(cmax).expand_as(cmax)
        w = self.s_gap_w(cmax).expand_as(cmax)

        y = y * self.s_sig(self.s_conv(torch.cat([cmax, h, w], dim=1))).expand_as(y)
        return y
##### Channel and Coordinate Attention #####

##### heatmap #####
class MaskP(nn.Module):
    def __init__(self, c1):
        super(MaskP, self).__init__()
        self.conv1 = Conv(c1, c1, 7, 1, 3)
        self.conv2 = nn.Conv2d(c1, 1, 1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.sig(self.conv2(self.conv1(x)))
        return y
##### heatmap #####

##### More Efficient Channel Attention #####
class Meca(nn.Module):
    def __init__(self, c1):
        super(Meca, self).__init__()
        assert not bool(math.sqrt(c1) - int(math.sqrt(c1)))  # c1 has to satisfy c1 = int * int

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.wh = int(math.sqrt(c1))
        t = int(abs((math.log(c1, 2) + 1)))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv2d(1, 1, kernel_size=k, stride=1, padding=(k - 1) // 2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        b, c, w, h = y.shape
        y = self.conv(y.view(b, 1, self.wh, self.wh)).view(b, self.wh * self.wh, 1, 1)
        y = self.sig(y).expand_as(x) * x
        return y
##### More Efficient Channel Attention #####

##### New Conv Module #####
# Error
class ChainConv(nn.Module):
    def __init__(self, c1, c2, e=0.5, act=True):
        super(ChainConv, self).__init__()
        self.meca = Meca(c1)
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1x1 = Conv(c1, c_, 1, 1, act=act)
        self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.dwconv2 = Conv(2 * c_, c_, 3, 1, 1, g=2 * c_, act=act)
        self.dwconv3 = Conv(2 * c_, c_, 3, 1, 1, g=2 * c_, act=act)
        self.outConv = Conv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        x = self.conv1x1(self.meca(x))
        y = torch.cat([x, self.dwconv1(x)], dim=1)
        y = torch.cat([x, self.dwconv2(y)], dim=1)
        y = torch.cat([x, self.dwconv3(y)], dim=1)
        y = self.outConv(y)
        return y
##### New Conv Module #####

# ##### Enhance Ghost Bottleneck 1 #####
# class EGhost_1(nn.Module):
#     def __init__(self, c1, c2, act=True, e=0.5):
#         super(EGhost_1, self).__init__()
#         c_ = int(c1 * e)
#         act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.conv1x1 = Conv(c1, c_, 1, 1, act=act)
#         self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#         self.dwconv2 = Conv(2 * c_, 2 * c_, 3, 1, 1, g=2 * c_, act=act)
#         self.cs = ChannelShuffle(2 * c1, 2)
#
#     def forward(self, x):
#         y = self.conv1x1(x)
#         y = torch.cat([y, self.dwconv1(y)], dim=1)
#         y = torch.cat([x, self.dwconv2(y)], dim=1)
#         y = self.cs(y)
#         return y
# ##### Enhance Ghost Bottleneck 1 #####

##### Enhance Ghost Bottleneck 3 #####
class EGhost_1(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(EGhost_1, self).__init__()
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1x1 = Conv(c1, c_, 1, 1, act=act)
        self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.dwconv2 = Conv(2 * c_, 2 * c_, 3, 1, 1, g=2 * c_, act=act)
        self.cs = ChannelShuffle(4 * c_, 4)

    def forward(self, x):
        y1 = self.conv1x1(x)
        y2 = self.dwconv1(y1)
        y3 = self.dwconv2(torch.cat([y1, y2], dim=1))
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.cs(y)
        return y
##### Enhance Ghost Bottleneck 3 #####

##### Enhance Ghost Bottleneck Downsampling #####
class EGhost_3(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5, k=3):
        super(EGhost_3, self).__init__()
        self.c = c1
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv11 = Conv(c_, c2//2, 1, 1, act=act)
        self.dwconv1 = Conv(c2//2, c2//2, 3, 2, 1, g=c2//2, act=False)
        self.conv12 = Conv(c2//2, c2//2, 1, 1, act=act)

        self.conv21 = Conv(c_, c2//2, 1, 1, act=act)
        self.max = MP()
        self.conv22 = Conv(c2//2, c2//2, 1, 1, act=act)

        self.cs = ChannelShuffle(c2, 2)

    def forward(self, x):
        y1, y2 = torch.split(x, self.c // 2, dim=1)
        y1 = self.conv12(self.dwconv1(self.conv11(y1)))
        y2 = self.conv22(self.max(self.conv21(y2)))
        y = torch.cat([y2, y1], dim=1)
        y = self.cs(y)
        return y
##### Enhance Ghost Bottleneck Downsampling #####

##### Enhance Ghost Bottleneck 2 #####
class EGhost_2(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5, k=3):
        super(EGhost_2, self).__init__()
        self.c = c1
        self.switch = bool(math.sqrt(c1) - int(math.sqrt(c1)))
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.meca = Meca(c_) if self.switch else Meca(c1)
        self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=False)
        self.dwconv2 = Conv(c_, c_, 3, 1, 1, g=c_, act=False)
        self.conv1 = Conv(c_, c_, 1, 1, act=act)

        self.conv1x1 = Conv(2 * c_, c_ // 2, 1, 1, act=act)
        self.dwconv3 = Conv(c_ // 2, c_ // 2, k, 1, k // 2, g=c_ // 2, act=act)
        # self.outConv = Conv(2 * c_, c2, 1, 1, act=act)
        self.cs = ChannelShuffle(4 * c_, 4)

    def forward(self, x):
        if self.switch:
            y1, y2 = torch.split(x, self.c // 2, dim=1)
            y1 = self.conv1(self.dwconv2(self.dwconv1(self.meca(y1))))
            y1 = self.conv1x1(torch.cat([y2, y1], dim=1))
            y = torch.cat([y2, y1, self.dwconv3(y1)], dim=1)
            # y = self.outConv(y)
            y = self.cs(y)
            return y
        else:
            y1, y2 = torch.split(self.meca(x), self.c // 2, dim=1)
            y1 = self.conv1(self.dwconv2(self.dwconv1(y1)))
            y1 = self.conv1x1(torch.cat([y2, y1], dim=1))
            y = torch.cat([y2, y1, self.dwconv3(y1)], dim=1)
            # y = self.outConv(y)
            y = self.cs(y)
            return y

##### Enhance Ghost Bottleneck 2 #####

##### Enhance Ghost Bottleneck 4 #####
class EGhost_4(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5, k=3):
        super(EGhost_4, self).__init__()
        self.c = c1
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1 = Conv(c_, c1, 1, 1)
        self.dwconv1 = Conv(c1, c1, 3, 2, 1, g=c1, act=act)
        self.conv2 = Conv(c_, c1, 1, 1)
        self.max = MP()

        self.conv1x1 = Conv(2 * c1, c2//2, 1, 1, act=act)
        self.dwconv3 = Conv(c2//2, c2//2, k, 1, k // 2, g=c2//2, act=act)
        # self.outConv = Conv(2 * c_, c2, 1, 1, act=act)
        self.cs = ChannelShuffle(4 * c_, 4)

    def forward(self, x):
        y1, y2 = torch.split(x, self.c // 2, dim=1)
        y1 = self.dwconv1(self.conv1(y1))
        y2 = self.max(self.conv2(y2))
        # y1 = self.conv1(self.dwconv1(y1))
        # y2 = self.conv2(self.max(y2))
        y = self.conv1x1(torch.cat([y2, y1], dim=1))
        y = torch.cat([y, self.dwconv3(y)], dim=1)
        y = self.cs(y)
        return y

##### Enhance Ghost Bottleneck 4 #####

# ##### Enhance Ghost Bottleneck 22 #####
# class EGhost_22(nn.Module):
#     def __init__(self, c1, c2, act=True, e=0.5, k=3):
#         super(EGhost_22, self).__init__()
#         self.c = c1
#         c_ = int(c1 * e)
#         act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         # self.conv1 = Conv(c1, c1, 1, 1, act=act)
#         self.sff = nn.Sequential(
#             SpatialAtt(c_),
#             SpatialAtt(c_),
#             SpatialAtt(c_),
#             SpatialAtt(c_),
#         )
#         # self.sa1 = SpatialAtt(c_)
#         # self.sa2 = SpatialAtt(c_)
#         # self.sa3 = SpatialAtt(c_)
#
#         self.conv1x1 = Conv(2 * c_, c_ // 2, 1, 1, act=act)
#         self.dwconv = Conv(c_ // 2, c_ // 2, k, 1, k // 2, g=c_ // 2, act=act)
#         self.outConv = Conv(2 * c_, c2, 1, 1, act=act)
#
#     def forward(self, x):
#         y1, y2 = torch.split(x, self.c // 2, dim=1)
#         y1 = self.sff(y1)
#         # y1 = self.sa1(y1)
#         # t2 = self.sa2(y1)
#         # t3 = self.sa3(t2)
#         y1 = self.conv1x1(torch.cat([y1, y2], dim=1))
#         y = torch.cat([y1, y2, self.dwconv(y1)], dim=1)
#         y = self.outConv(y)
#         return y
#
# ##### Enhance Ghost Bottleneck 22 #####

# ##### Enhance Ghost Bottleneck 24 #####
# class EGhost_24(nn.Module):
#     def __init__(self, c1, c2, act=True, e=0.5):
#         super(EGhost_24, self).__init__()
#         self.c = c1
#         self.switch = bool(math.sqrt(c1) - int(math.sqrt(c1)))
#         c_ = int(c1 * e)
#         act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.meca = Meca(c_) if self.switch else Meca(c1)
#         self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#         self.dwconv2 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#         self.dwconv3 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#
#         self.conv1x1 = Conv(2 * c_, c_, 1, 1, act=act)
#         self.dwconv4 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#         self.outConv = Conv(2 * c_, c2, 1, 1, act=act)
#
#     def forward(self, x):
#         if self.switch:
#             y1, y2 = torch.split(x, self.c // 2, dim=1)
#             y1 = self.dwconv3(self.dwconv2(self.dwconv1(self.meca(y1))))
#             y1 = self.conv1x1(torch.cat([y2, y1], dim=1))
#             y2 = self.dwconv4(y1)
#             y = torch.cat([y2, y1], dim=1)
#             y = self.outConv(y)
#             return y
#         else:
#             y1, y2 = torch.split(self.meca(x), self.c // 2, dim=1)
#             y1 = self.dwconv3(self.dwconv2(self.dwconv1(y1)))
#             y1 = self.conv1x1(torch.cat([y2, y1], dim=1))
#             y2 = self.dwconv4(y1)
#             y = torch.cat([y2, y1], dim=1)
#             y = self.outConv(y)
#             return y
#
# ##### Enhance Ghost Bottleneck 24 #####
#
# ##### Enhance Ghost Bottleneck 4 #####
# class EGhost_4(nn.Module):
#     def __init__(self, c1, c2, act=True, e=0.5):
#         super(EGhost_4, self).__init__()
#         c_ = int(c1 * e)
#         act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.conv1x1 = Conv(c1, c_, 1, 1, act=act)
#         self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#         self.dwconv2 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#         self.dwconv3 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
#         # self.cs = ChannelShuffle(4 * c_, 4)
#         self.outConv = Conv(4 * c_, c2, 1, 1, act=act)
#
#     def forward(self, x):
#         y1 = self.conv1x1(x)
#         y2 = self.dwconv1(y1)
#         y3 = self.dwconv2(y2)
#         y4 = self.dwconv3(y3)
#         y = torch.cat([y1, y2, y3, y4], dim=1)
#         # y = self.cs(y)
#         y = self.outConv(y)
#         return y
#
# ##### Enhance Ghost Bottleneck 4 #####

##### GhostSPPF #####
class GhostSPPF(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(GhostSPPF, self).__init__()
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1 = Conv(c1, c_, 1, 1, act=act)
        self.mp1 = SP(k=5)
        self.conv2 = Conv(2 * c_, c_, 1, 1, act=act)
        self.mp2 = SP(k=9)
        self.conv3 = Conv(2 * c_, c_, 1, 1, act=act)
        self.mp3 = SP(k=13)
        self.conv4 = Conv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(torch.cat([y, self.mp1(y)], dim=1))
        y = self.conv3(torch.cat([y, self.mp2(y)], dim=1))
        y = self.conv4(torch.cat([y, self.mp3(y)], dim=1))
        return y

##### GhostSPPF #####

##### Fine-Grained Module #####
class FGM(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.25):
        super(FGM, self).__init__()
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1_1 = Conv(c1, c_, 1, 1, act=act)
        self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.dwconv2 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.dwconv3 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.conv1_2 = Conv(4 * c_, 1, 1, act=act)

        self.dwconv2_1 = Conv(c1, c1, 3, 1, 1, g=c_, act=act)
        self.dwconv2_2 = Conv(c1, c1, 3, 1, 1, g=c_, act=act)
        self.conv2_1 = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y1 = self.conv1x1(x)
        y2 = self.dwconv1(y1)
        y3 = self.dwconv2(y2)
        y4 = self.dwconv3(y3)
        out1 = self.conv1_2(torch.cat([y1, y2, y3, y4], dim=1))
        out2 = self.sig(self.conv2_1(self.dwconv2_2(self.dwconv2_1(x)))).expand_as(out1)
        return out1 * out2

##### Fine-Grained Module #####

##### Channel Shuffle #####
class ChannelShuffle(nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups):
        super(ChannelShuffle, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return self.channel_shuffle(x, self.groups)

    @staticmethod
    def channel_shuffle(x, groups):
        batch, channels, height, width = x.size()
        # assert (channels % groups == 0)
        channels_per_group = channels // groups
        x = x.view(batch, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        return x
##### Channel Shuffle #####

##### Spatial Att #####
class SpatialAtt(nn.Module):
    def __init__(self, c1, act=True):
        super(SpatialAtt, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_, 1, 1, act=act)
        self.conv2 = Conv(c_, 1, 1, 1, act=act)
        self.conv3 = Conv(2, 1, 1, 1, 0, act=nn.Sigmoid())
        self.dwconv = Conv(c_, c_, 3, 1, 1, g=c_, act=act)

    def forward(self, x):
        y = self.conv1(x)
        y1 = torch.cat([torch.max(y, dim=1)[0].unsqueeze(1), self.conv2(y)], dim=1)
        y1 = self.conv3(y1).expand_as(y) * y
        y = torch.cat([self.dwconv(y), y1], dim=1)
        return y
##### Spatial Att #####

##### Spatial Att2 #####
class SpatialAtt2(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(SpatialAtt2, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        c_ = int(c1 * e)
        self.conv1 = Conv(c1, c_//2, 1, 1, act=act)
        self.dwconv1 = Conv(c_//2, c_//2, 3, 1, 1, g=c_//2, act=act)
        self.conv2 = Conv(c_, c_//4, 1, 1, act=act)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.avgpool = nn.AvgPool2d(3, 1, 1)
        self.dwconv = Conv(c_//4, c_//4, 3, 1, 1, g=c_//4, act=act)
        self.conv3 = Conv(c_, c_, 1, 1, 0, act=nn.Sigmoid())

        self.conv4 = Conv(c_, c2 // 2, 1, 1, act=act)
        self.dwconv4 = Conv(c2//2, c2//2, 3, 1, 1, g=c2//2, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = torch.cat([y1, self.dwconv1(y1)], dim=1)
        y2 = self.conv2(y1)
        y = self.conv3(torch.cat([y2, self.maxpool(y2), self.dwconv(y2), self.avgpool(y2)], dim=1)) * y1
        y = self.conv4(y)
        y = torch.cat([y, self.dwconv4(y)], dim=1)
        return y
##### Spatial Att2 #####

##### Spatial Feature Extraction Module #####
class SFEM(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(SFEM, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        c_ = int(c1 * e)
        self.conv = Conv(c1, c_, 1, 1, act=act)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.avgpool = nn.AvgPool2d(3, 1, 1)
        self.dwconv = Conv(c_, c_, 3, 1, 1, g=c_, act=act)

    def forward(self, x):
        y = self.conv(x)
        y = torch.cat([y, self.dwconv(y), self.maxpool(y), self.avgpool(y)], dim=1)
        return y
##### Spatial Feature Extraction Module #####

##### Spatial Feature Extraction Module2 #####
class SFEM2(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(SFEM2, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # c_ = int(c1 * e)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        # self.avgpool = nn.AvgPool2d(3, 1, 1)
        self.dwconv = Conv(c1, c1, 3, 1, 1, g=c1, act=act)
        self.conv = Conv(2 * c1, c2, 1, 1, act=act)

    def forward(self, x):
        y = torch.cat([self.dwconv(x), self.maxpool(x)], dim=1)
        y = self.conv(y)
        return y
##### Spatial Feature Extraction Module2 #####

##### Spatial Feature Cross Stage Partial Module #####
class SFCSP(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(SFCSP, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        c_ = int(c1 * e)
        self.conv1 = Conv(c1, c_//2, 1, 1, act=act)
        self.dwconv1 = Conv(c_//2, c_//2, 3, 1, 1, g=c_//2, act=act)

        self.sfem = nn.Sequential(
            SFEM(c_, 2*c_, act, 0.5),
            SFEM(2*c_, 2*c_, act, 0.25),
            SFEM(2 * c_, 2 * c_, act, 0.25),
            SFEM(2 * c_, 2 * c_, act, 0.25),
        )

        self.conv2 = Conv(2 * c_, c_//2, 1, 1, 0, act=act)
        self.dwconv2 = Conv(c_//2, c_//2, 3, 1, 1, g=c_//2, act=act)

        self.conv3 = Conv(2 * c_, c2 // 2, 1, 1, act=act)
        self.dwconv3 = Conv(c2//2, c2//2, 3, 1, 1, g=c2//2, act=act)

    def forward(self, x):
        y = self.conv1(x)
        y = torch.cat([y, self.dwconv1(y)], dim=1)
        y1 = self.sfem(y)
        y1 = self.conv2(y1)
        y = torch.cat([y1, self.dwconv2(y1), y], dim=1)
        y = self.conv3(y)
        y = torch.cat([y, self.dwconv3(y)], dim=1)
        return y
##### Spatial Feature Extraction Module #####

##### Cross information aggregation #####
class CIA(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(CIA, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        c_ = int(c1 * e)
        self.c_ = c_
        self.conv1 = Conv(c1, c_, 1, 1, act=act)
        self.conv2 = Conv(c1, c_, 1, 1, act=act)

        self.dw1 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.dw2 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.dw3 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)
        self.dw4 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)

        self.sf = ChannelShuffle(c_, 2)
        self.dw5 = Conv(c_, c_, 3, 1, 1, g=c_, act=act)

        self.outConv = Conv(4 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        y2_1, y2_2 = torch.split(y2, self.c_ // 2, dim=1)
        y1_1, y1_2 = torch.split(y1, self.c_ // 2, dim=1)
        y21 = self.sf(torch.cat([y2_1, y1_1], dim=1))
        y11 = torch.cat([y2_2, y1_2], dim=1)
        y11 = self.dw2(self.dw1(y11))

        y2_1, y2_2 = torch.split(y21, self.c_ // 2, dim=1)
        y1_1, y1_2 = torch.split(y11, self.c_ // 2, dim=1)
        y21 = torch.cat([y2_1, y1_1], dim=1)
        y11 = torch.cat([y2_2, y1_2], dim=1)
        y11 = self.dw4(self.dw3(y11))
        y21 = self.dw5(y21)

        y = self.outConv(torch.cat([y1, y2, y11, y21], dim=1))
        return y
##### Cross information aggregation #####

##### Mask #####
class Mask(nn.Module):
    def __init__(self, c1, c2, act=True):
        super(Mask, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.conv1 = Conv(c1, c1 // 3, 1, 1, act=act)
        # self.dwconv2 = Conv(c1 // 3, c1 // 3, 3, 1, 1, g=c1 // 3, act=act)
        # self.conv2 = Conv(c1, 1, 1, 1, None, 1, nn.Sigmoid())
        self.conv = nn.Sequential(
            Conv(c1, c1, 3, 1, 1, g=c1, act=act),
            Conv(c1, c1, 3, 1, 1, g=c1, act=act),
            Conv(c1, 1, 1, 1, None, 1, nn.Sigmoid())
        )

    def forward(self, x):
        # y = self.conv1(x)
        # y = self.sig(self.conv2(torch.cat([y, self.dwconv2(y)], dim=1)))
        y = self.conv(x)
        return y
##### Mask #####

##### Mask Fusion #####
class Fusion(nn.Module):
    def __init__(self, c2, operation=None, shortcut=True, act=True):
        super(Fusion, self).__init__()
        self.op = None
        self.shortcut = shortcut
        self.act = None
        if operation == 'up':
            self.op = nn.Upsample(None, scale_factor=2, mode='bilinear')
        elif operation == 'down':
            self.op = Conv(1, 1, 3, 2, None, 1, nn.LeakyReLU(0.1))
        if act:
            self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        mask = x[0]
        feature = x[1]
        if self.op is not None:
            mask = self.op(mask)
        if self.shortcut:
            y = (mask * feature) + feature
        else:
            y = mask * feature
        if self.act:
            y = self.act(y)
        return y
##### Mask Fusion #####

##### SplitFeature #####
class SplitFeature(nn.Module):
    def __init__(self, c2, dim=1, pos=0, num=3):
        super(SplitFeature, self).__init__()
        assert pos < num
        self.pos = pos
        self.num = num
        self.dim = dim
        self.size = c2

    def forward(self, x):
        y = torch.split(x, self.size, self.dim)[self.pos]
        return y
##### SplitFeature #####

#####  #####
class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=3, act=True):
        super(ConvBlock, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1_1 = Conv(c1, c1, (k, 1), 1, None, g=c1, act=act)
        self.conv1_2 = Conv(c1, c1, (1, k), 1, None, g=c1, act=False)
        self.conv2 = Conv(c1, c1, 3, 1, None, g=c1, act=False)
        self.act = act

    def forward(self, x):
        y1 = self.conv1_2(self.conv1_1(x))
        y2 = self.conv2(x)
        y = self.act(y1 + y2)
        return y
#####  #####

##### Simple Conv #####
class SConv(nn.Module):
    def __init__(self, c1, c2, k=3, act=True):
        super(SConv, self).__init__()
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.dw = Conv(c1, c1, k, 1, None, g=c1, act=act)
        self.conv = Conv(c1, c2, 1, 1, None, act=act)

    def forward(self, x):
        return self.conv(self.dw(x))
##### Simple Conv #####


##### Shufflenet B2 #####
class ShuffleBlock2(nn.Module):
    def __init__(self, c1, c2):
        super(ShuffleBlock2, self).__init__()
        c_ = c1 // 2
        self.c = c1
        self.conv1 = Conv(c_, c_, 1, 1, act=nn.ReLU())
        self.dwconv = Conv(c_, c_, 3, 1, 1, g=c_, act=False)
        self.conv2 = Conv(c_, c_, 1, 1, act=nn.ReLU())
        self.sf = ChannelShuffle(c1, 2)

    def forward(self, x):
        shortcut, y = torch.split(x, self.c // 2, dim=1)
        y = self.conv2(self.dwconv(self.conv1(y)))
        y = torch.cat([shortcut, y], dim=1)
        return self.sf(y)
##### Shufflenet B2 #####

##### Shufflenet B1 #####
class ShuffleBlock1(nn.Module):
    def __init__(self, c1, c2):
        super(ShuffleBlock1, self).__init__()
        self.b1 = nn.Sequential(
            Conv(c1, c1, 3, 2, 1, g=c1, act=False),
            Conv(c1, c2 // 2, 1, 1, act=nn.ReLU()),
        )

        self.b2 = nn.Sequential(
            Conv(c1, c2 // 2, 1, 1, act=nn.ReLU()),
            Conv(c2 // 2, c2 // 2, 3, 2, None, g=c2 // 2, act=False),
            Conv(c2 // 2, c2 // 2, 1, 1, act=nn.ReLU()),
        )

        self.sf = ChannelShuffle(c1, 2)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y = torch.cat([y1, y2], dim=1)
        return self.sf(y)
##### Shufflenet B1 #####

##### MobileNet V3 #####
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation(FC+ReLU+FC+Sigmoid)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MobileNet_Block(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se=True):
        super(MobileNet_Block, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(0.1),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(0.1),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                nn.LeakyReLU(0.1),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y
##### MobileNet V3 #####

##### Balanced Channel Attention #####
class BCA(nn.Module):
    def __init__(self, c1):
        super(BCA, self).__init__()
        assert not bool(math.sqrt(c1) - int(math.sqrt(c1)))  # c1 has to satisfy c1 = int * int

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.wh = int(math.sqrt(c1))
        t = int(abs((math.log(c1, 2) + 1)))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv2d(1, 1, kernel_size=k, stride=1, padding=(k - 1) // 2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        b, c, w, h = y.shape
        y = self.conv(y.view(b, 1, self.wh, self.wh)).view(b, self.wh * self.wh, 1, 1)
        y = self.sig(y).expand_as(x) * x
        return y
##### More Efficient Channel Attention #####

##### Downsampling Module #####
class DM(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5, k=3):
        super(DM, self).__init__()
        self.c = c1
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv11 = Conv(c_, c2//2, 1, 1, act=act)
        self.dwconv1 = Conv(c2//2, c2//2, 3, 2, 1, g=c2//2, act=False)
        self.conv12 = Conv(c2//2, c2//2, 1, 1, act=act)

        self.conv21 = Conv(c_, c2//2, 1, 1, act=act)
        self.max = MP()
        self.conv22 = Conv(c2//2, c2//2, 1, 1, act=act)

        self.cs = ChannelShuffle(c2, 2)

    def forward(self, x):
        y1, y2 = torch.split(x, self.c // 2, dim=1)
        y1 = self.conv12(self.dwconv1(self.conv11(y1)))
        y2 = self.conv22(self.max(self.conv21(y2)))
        y = torch.cat([y2, y1], dim=1)
        y = self.cs(y)
        return y
##### Downsampling Module #####

##### Efficient Feature Extraction Module #####
class EFEM(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5, k=3):
        super(EFEM, self).__init__()
        self.c = c1
        self.switch = bool(math.sqrt(c1) - int(math.sqrt(c1)))
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.bca = BCA(c_) if self.switch else BCA(c1)
        self.dwconv1 = Conv(c_, c_, 3, 1, 1, g=c_, act=False)
        self.dwconv2 = Conv(c_, c_, 3, 1, 1, g=c_, act=False)
        self.conv1 = Conv(c_, c_, 1, 1, act=act)

        self.conv1x1 = Conv(2 * c_, c_ // 2, 1, 1, act=act)
        self.dwconv3 = Conv(c_ // 2, c_ // 2, k, 1, k // 2, g=c_ // 2, act=act)
        # self.outConv = Conv(2 * c_, c2, 1, 1, act=act)
        self.cs = ChannelShuffle(4 * c_, 4)

    def forward(self, x):
        if self.switch:
            y1, y2 = torch.split(x, self.c // 2, dim=1)
            y1 = self.conv1(self.dwconv2(self.dwconv1(self.bca(y1))))
            y1 = self.conv1x1(torch.cat([y2, y1], dim=1))
            y = torch.cat([y2, y1, self.dwconv3(y1)], dim=1)
            # y = self.outConv(y)
            y = self.cs(y)
            return y
        else:
            y1, y2 = torch.split(self.bca(x), self.c // 2, dim=1)
            y1 = self.conv1(self.dwconv2(self.dwconv1(y1)))
            y1 = self.conv1x1(torch.cat([y2, y1], dim=1))
            y = torch.cat([y2, y1, self.dwconv3(y1)], dim=1)
            # y = self.outConv(y)
            y = self.cs(y)
            return y

##### Efficient Feature Extraction Module #####


##### Progressive Spatial Pyramid Pooling #####
class PSPP(nn.Module):
    def __init__(self, c1, c2, act=True, e=0.5):
        super(PSPP, self).__init__()
        c_ = int(c1 * e)
        act = nn.ELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1 = Conv(c1, c_, 1, 1, act=act)
        self.mp1 = SP(k=5)
        self.conv2 = Conv(2 * c_, c_, 1, 1, act=act)
        self.mp2 = SP(k=9)
        self.conv3 = Conv(2 * c_, c_, 1, 1, act=act)
        self.mp3 = SP(k=13)
        self.conv4 = Conv(2 * c_, c2, 1, 1, act=act)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(torch.cat([y, self.mp1(y)], dim=1))
        y = self.conv3(torch.cat([y, self.mp2(y)], dim=1))
        y = self.conv4(torch.cat([y, self.mp3(y)], dim=1))
        return y

##### Progressive Spatial Pyramid Pooling #####