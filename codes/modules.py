import torch
import torch.nn as nn


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, is_up=True):
        super().__init__()
        self.is_up = is_up
        if self.is_up:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c+out_c, out_c)

    def forward(self, x, s):
        if self.is_up:
            x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x


class ChannelAttention(nn.Module):   #CA
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):  #SA
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class RCSA(nn.Module):
    def __init__(self, in_channel):
        super(RCSA, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channel)

    def forward(self, x):
        # CA = x.mul(self.ca(x))
        # # 元素级别点对点相乘
        # SA = CA.mul(self.sa(CA))

        x1_ca = x.mul(self.ca(x))
        x1_sa = x1_ca.mul(self.sa(x1_ca))
        x = x + x1_sa
        return x


class RSA(nn.Module):
    def __init__(self, channels, padding=0, groups=1, matmul_norm=True):
        super(RSA, self).__init__()
        self.channels = channels
        self.padding = padding
        self.groups = groups
        self.matmul_norm = matmul_norm
        self._channels = channels//8

        self.conv_query = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_key = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_value = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=groups)

        self.conv_output = Conv2D(in_c=channels, out_c=channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Get query, key, value tensors
        query = self.conv_query(x).view(batch_size, -1, height*width)
        key = self.conv_key(x).view(batch_size, -1, height*width)
        value = self.conv_value(x).view(batch_size, -1, height*width)

        # Apply transpose to swap dimensions for matrix multiplication
        query = query.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels//8)
        value = value.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels)

        # Compute attention map
        attention_map = torch.matmul(query, key)
        if self.matmul_norm:
            attention_map = (self._channels**-.5) * attention_map
        attention_map = torch.softmax(attention_map, dim=-1)

        # Apply attention
        out = torch.matmul(attention_map, value)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)

        # Apply output convolution
        out = self.conv_output(out)
        out = out + x

        return out

