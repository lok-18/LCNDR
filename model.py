import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Module_1

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


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
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

class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.ca_1 = CoordAtt(32, 32, 64)
        self.ca_2 = CoordAtt(32, 32, 64)
        self.ca_3 = CoordAtt(32, 32, 64)
        # Maxpooling和Avgpooling还没加

    def forward(self, x):
        x_ca1 = self.ca_1(x)
        x_ca2 = self.ca_2(x)
        x_ca3 = self.ca_3(x)
        out = torch.cat((x_ca1, x_ca2, x_ca3), 1)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SE(nn.Module):
    def __init__(self):
        super(SE, self).__init__()
        self.se = SELayer(32, 2)
        self.conv1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 96, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x_se1 = self.se(x)
        x_se2 = self.se(x)
        x_se3 = self.se(x)
        out = torch.cat((x_se1, x_se2, x_se3), 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        out = torch.cat((out, x), 1)

        return out

# Module_2
class Up_Module(nn.Module):
    def __init__(self):
        super(Up_Module, self).__init__()
        self.conv_up_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu_up_1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv_up_2 = nn.Conv2d(64, 96, 3, 1, 1)
        self.relu_up_2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        # self.conv_up_3 = nn.Conv2d(96, 128, 3, 1, 1)
        # self.Upsample = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=True)  # size直接指定大小 scale_factor为扩大的倍数

    def forward(self, x):
        x = self.conv_up_1(x)
        x = self.relu_up_1(x)
        x = self.conv_up_2(x)
        x = self.relu_up_2(x)
        # x = self.conv_up_3(x)
        # x = self.Upsample(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
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

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 32, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

# Module_3
class Self_Attention(nn.Module):
    def __init__(self, bn=True):
        super(Self_Attention, self).__init__()
        self.conv1 = nn.Conv2d(33, 48, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, stride=2)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.Cv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.cv2 = nn.Conv2d(32, 8, kernel_size=1, stride=1)
        self.cv3 = nn.Conv2d(32, 8, kernel_size=1, stride=1)

    def forward(self, under, over):
        x = torch.cat((under, over), dim=1)
        output = self.relu(self.bn(self.conv1(x)))
        output = self.maxpool(output)
        output = self.relu(self.bn2(self.conv2(output)))
        C = self.Cv1(output)
        C = C.view(C.shape[0] * C.shape[1], C.shape[2] * C.shape[3])
        c1 = self.cv2(output)
        c1 = c1.view(c1.shape[0] * c1.shape[2] * c1.shape[3], 8)
        c2 = self.cv3(output)
        c2 = c2.view(c2.shape[0] * c2.shape[2] * c2.shape[3], 8).t()
        c_1 = torch.mm(c1, c2)  # 矩阵相乘
        sogtmax = torch.nn.Softmax(dim=1)
        c = sogtmax(c_1)
        # c = torch.nn.Softmax(torch.mm(c1, c2), dim=1)
        c = c.view(output.shape[0], c.shape[0], int(c.shape[1] // output.shape[0]))
        c = c.view(c.shape[0] * c.shape[1], c.shape[2])
        attention_map = torch.mm(C, c.t())
        attention_map = attention_map.view(output.shape[0], output.shape[1], output.shape[2] * output.shape[0], output.shape[3] * output.shape[0] )
        attention_map = F.interpolate(attention_map, size=[under.shape[2], under.shape[3]])  # up-sample
        return attention_map

class net_test(nn.Module):
    def __init__(self):
        super(net_test, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.m1 = SE()
        self.m3 = Self_Attention()
        self.conv_2 = nn.Conv2d(224, 1, 3, 1, 1)
        self.act = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(1)
        # self.conv_3 = nn.Conv2d(128, 1, 3, 1, 1)
        self.t = nn.Tanh()
        self.r = nn.PReLU()

    def forward(self, over, under):
        # over
        x_over = self.conv_1(over)
        x_m1_over = self.m1(x_over)
        x_m3_over = self.m3(x_over, under)

        x_m1m2m3_over = torch.cat((x_m1_over, x_m3_over), 1)
        x_over = self.conv_2(x_m1m2m3_over)
        x_over = self.bn(x_over)
        x_over = self.act(x_over)
        x_over = self.t(x_over)
        # x_over = self.r(x_over)
        x_over = torch.mul(x_over, over)

        # under
        x_under = self.conv_1(under)
        x_m1_under = self.m1(x_under)
        x_m3_under = self.m3(x_under, over)

        x_m1m2m3_under = torch.cat((x_m1_under, x_m3_under), 1)
        x_under = self.conv_2(x_m1m2m3_under)
        x_under = self.bn(x_under)
        x_under = self.act(x_under)
        x_under = self.t(x_under)
        # x_under = self.r(x_under)
        x_under = torch.add(x_under, under)

        #last
        x = torch.add(x_over, x_under)

        return x




