# -*- coding: utf-8 -*-

from torch import nn

class Bottleneck(nn.Module):

    def __init__(self, in_channels, final_out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.in_channels = in_channels
        self.final_out_channels = final_out_channels
        self.stride = stride
        self.downsample = downsample

        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(num_features=self.final_out_channels // 4)
        self.batch_norm2 = nn.BatchNorm2d(num_features=self.final_out_channels // 4)
        self.batch_norm3 = nn.BatchNorm2d(num_features=self.final_out_channels)

        self.conv1 = nn.Conv2d(self.in_channels, self.final_out_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(self.final_out_channels // 4, self.final_out_channels // 4, kernel_size=3,
                               stride=self.stride, padding=1)
        self.conv3 = nn.Conv2d(self.final_out_channels // 4, self.final_out_channels, kernel_size=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)

        if self.downsample:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out
