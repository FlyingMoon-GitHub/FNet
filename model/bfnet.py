# -*- coding: utf-8 -*-

from torch import nn
from torchsummary import *

from model.bottleneck import *

class BFNet(nn.Module):

    def __init__(self, config):
        super(BFNet, self).__init__()

        self.target_size = (3, config['target_size'], config['target_size'])
        self.class_num = config['class_num']
        self.use_cuda = config['use_cuda']

        self.relu = nn.ReLU()

        self.conv1_aux = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1_aux = nn.BatchNorm2d(num_features=64)
        self.max_pooling_aux = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x_aux = self._bottleneck_layers(in_channels=64, final_out_channels=256, loops=3)
        self.conv3_x_aux = self._bottleneck_layers(in_channels=256, final_out_channels=512, stride=2, loops=4)
        self.conv4_x_aux = self._bottleneck_layers(in_channels=512, final_out_channels=1024, stride=2, loops=6)
        self.conv5_x_aux = self._bottleneck_layers(in_channels=1024, final_out_channels=2048, stride=2, loops=3)

        self.assist_aux_2 = self._assistant_layers(in_channels=256, first_kernel_size=7, loops=4)
        self.assist_aux_3 = self._assistant_layers(in_channels=512, first_kernel_size=5, loops=3)
        self.assist_aux_4 = self._assistant_layers(in_channels=1024, first_kernel_size=3, loops=2)
        self.assist_aux_5 = self._assistant_layers(in_channels=2048, first_kernel_size=1, loops=1)

        self.avg_pool_aux = nn.AvgPool2d(kernel_size=16, stride=1)
        self.fc_aux = nn.Linear(in_features=2048, out_features=self.class_num)

        self.conv1_prim = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1_prim = nn.BatchNorm2d(num_features=64)
        self.max_pooling_prim = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x_prim = self._bottleneck_layers(in_channels=64, final_out_channels=256, loops=3)
        self.conv3_x_prim = self._bottleneck_layers(in_channels=256, final_out_channels=512, stride=2, loops=4)
        self.conv4_x_prim = self._bottleneck_layers(in_channels=512, final_out_channels=1024, stride=2, loops=6)
        self.conv5_x_prim = self._bottleneck_layers(in_channels=1024, final_out_channels=2048, stride=2, loops=3)

        self.assist_prim_2 = self._assistant_layers(in_channels=256, first_kernel_size=7, loops=4)
        self.assist_prim_3 = self._assistant_layers(in_channels=512, first_kernel_size=5, loops=3)
        self.assist_prim_4 = self._assistant_layers(in_channels=1024, first_kernel_size=3, loops=2)
        self.assist_prim_5 = self._assistant_layers(in_channels=2048, first_kernel_size=1, loops=1)

        self.avg_pool_prim = nn.AvgPool2d(kernel_size=16, stride=1)
        self.fc_prim = nn.Linear(in_features=2048, out_features=self.class_num)

    def _bottleneck_layers(self, in_channels, final_out_channels, stride=1, loops=1):

        downsample = None
        if stride != 1 or in_channels != final_out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=final_out_channels, kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(final_out_channels)
            )

        layers = []
        layers.append(Bottleneck(in_channels=in_channels, final_out_channels=final_out_channels, stride=stride,
                                 downsample=downsample))
        for _ in range(1, loops):
            layers.append(Bottleneck(in_channels=final_out_channels, final_out_channels=final_out_channels))

        return nn.Sequential(*layers)

    def _assistant_layers(self, in_channels, first_kernel_size, loops=1, certain_channel=256):
        layers = []

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=certain_channel, kernel_size=1))

        layers.append(
            nn.Conv2d(in_channels=certain_channel, out_channels=certain_channel, kernel_size=first_kernel_size))

        for _ in range(loops):
            layers.append(Bottleneck(in_channels=certain_channel, final_out_channels=certain_channel, stride=1))

        layers.append(nn.Conv2d(in_channels=certain_channel, out_channels=in_channels, kernel_size=1,
                                padding=first_kernel_size // 2))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x_aux, x_prim):
        # Auxiliary Pathway
        c_aux_1 = self.conv1_aux(x_aux)
        # print('conv1_aux', c_aux_1.shape)

        c_prim_1 = self.conv1_prim(x_prim)
        # print('conv1_prim', c_prim_1.shape)

        c_aux_2 = self.batch_norm1_aux(c_aux_1)
        c_aux_2 = self.relu(c_aux_2)
        c_aux_2 = self.max_pooling_aux(c_aux_2)
        c_aux_2 = self.conv2_x_aux(c_aux_2)
        # print('conv2_aux', c_aux_2.shape)

        c_prim_2 = self.batch_norm1_prim(c_prim_1)
        c_prim_2 = self.relu(c_prim_2)
        c_prim_2 = self.max_pooling_prim(c_prim_2)
        c_prim_2 = self.conv2_x_prim(c_prim_2)
        # print('conv2_prim', c_prim_2.shape)

        assist_prim_2 = self.assist_prim_2(c_aux_2)
        # print('assist_prim_2', assist_prim_2.shape)
        assist_aux_2 = self.assist_aux_2(c_prim_2)
        # print('assist_aux_2', assist_aux_2.shape)

        c_prim_2 = c_prim_2 + assist_prim_2 * c_prim_2
        c_aux_2 = c_aux_2 + assist_aux_2 * c_aux_2

        c_aux_3 = self.conv3_x_aux(c_aux_2)
        # print('conv3_aux', c_aux_3.shape)

        c_prim_3 = self.conv3_x_prim(c_prim_2)
        # print('conv3_prim', c_prim_3.shape)

        assist_prim_3 = self.assist_prim_3(c_aux_3)
        # print('assist_prim_3', assist_prim_3.shape)
        assist_aux_3 = self.assist_aux_3(c_prim_3)
        # print('assist_aux_3', assist_aux_3.shape)

        c_prim_3 = c_prim_3 + assist_prim_3 * c_prim_3
        c_aux_3 = c_aux_3 + assist_aux_3 * c_aux_3

        c_aux_4 = self.conv4_x_aux(c_aux_3)
        # print('conv4_aux', c_aux_4.shape)

        c_prim_4 = self.conv4_x_prim(c_prim_3)
        # print('conv4_prim', c_prim_4.shape)

        assist_prim_4 = self.assist_prim_4(c_aux_4)
        # print('assist_prim_4', assist_prim_4.shape)
        assist_aux_4 = self.assist_aux_4(c_prim_4)
        # print('assist_aux_4', assist_aux_4.shape)

        c_prim_4 = c_prim_4 + assist_prim_4 * c_prim_4
        c_aux_4 = c_aux_4 + assist_aux_4 * c_aux_4

        c_aux_5 = self.conv5_x_aux(c_aux_4)
        # print('conv5_aux', c_aux_5.shape)

        c_prim_5 = self.conv5_x_prim(c_prim_4)
        # print('conv5_prim', c_prim_5.shape)

        assist_prim_5 = self.assist_prim_5(c_aux_5)
        # print('assist_prim_5', assist_prim_5.shape)
        assist_aux_5 = self.assist_aux_5(c_prim_5)
        # print('assist_aux_5', assist_aux_5.shape)

        c_prim_5 = c_prim_5 + assist_prim_5 * c_prim_5
        c_aux_5 = c_aux_5 + assist_aux_5 * c_aux_5

        aux_out = self.avg_pool_aux(c_aux_5)
        # print('avg_pool_aux', aux_out.shape)
        aux_out = aux_out.view(aux_out.size(0), -1)
        # print('view_aux', aux_out.shape)
        aux_out = self.fc_aux(aux_out)
        # print('fc_aux', aux_out.shape)

        prim_out = self.avg_pool_prim(c_prim_5)
        # print('avg_pool_prim', prim_out.shape)
        prim_out = prim_out.view(prim_out.size(0), -1)
        # print('view_prim', prim_out.shape)
        prim_out = self.fc_prim(prim_out)
        # print('fc_prim', prim_out.shape)

        return aux_out, prim_out

    def summary(self):
        summary(self, [self.target_size, self.target_size], device="cuda" if self.use_cuda else "cpu")
