# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.model_utils import get_activation
import numpy as np

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


from networks.DeScarGan.descargan import conv2d_block, conv2d_bn_block, deconv2d_bn_block, get_activation, deconv2d_block


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, batch_norm=True, activation=ACTIVATION, final='tanh', mc=False):
        super(Generator, self).__init__()

        conv_block = conv2d_bn_block if batch_norm else conv2d_block
        deconv_block = deconv2d_bn_block if batch_norm else deconv2d_block

        max_pool = nn.MaxPool2d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.c_dim = 0

        self.down0 = nn.Sequential(
            conv_block(n_channels + self.c_dim, nf, activation=act),
            conv_block(nf, nf, activation=act)
        )
        self.down1 = nn.Sequential(
            max_pool,
            conv_block(nf, 2 * nf, activation=act),
            conv_block(2 * nf, 2 * nf, activation=act),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2 * nf, 4 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act),

        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4 * nf, 8 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(8 * nf, 8 * nf, activation=act),
        )

        self.up3 = deconv_block(8 * nf, 4 * nf, activation=act)

        self.conv5 = nn.Sequential(
            conv_block(8 * nf, 4 * nf, activation=act),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act),
        )
        self.up2 = deconv_block(4 * nf, 2 * nf, activation=act)
        self.conv6 = nn.Sequential(
            conv_block(4 * nf, 2 * nf, activation=act),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(2 * nf, 2 * nf, activation=act),
        )

        self.up1 = deconv_block(2 * nf, nf, activation=act)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv_block(nf, out_channels, activation=final_layer),
        )

        self.conv7_g = nn.Sequential(
            conv_block(nf, out_channels, activation=final_layer),
        )

        #if NoTanh:
        #    self.conv7_k[-1] = self.conv7_k[-1][:-1]
        #    self.conv7_g[-1] = self.conv7_g[-1][:-1]

    def forward(self, x, alpha=None, method=None):
        if method != 'decode':
            x0 = self.down0(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)   # Dropout
            x3 = self.down3(x2)   # Dropout
            feat = [x0, x1, x2, x3]
            if method == 'encode':
                return feat

        if method == 'decode':
            [x0, x1, x2, x3] = x

        xu3 = self.up3(x3)
        #alpha
        x2 = alpha * x2 + (1 - alpha) * xu3  # alpha means the features from the encoder are connecting, or it is replaced by the features from the decoder
        xu3_ = xu3#.detach()
        cat3 = torch.cat([xu3_, x2], 1)
        x5 = self.conv5(cat3)   # Dropout
        xu2 = self.up2(x5)
        #alpha
        x1 = alpha * x1 + (1 - alpha) * xu2
        xu2_ = xu2#.detach()
        cat2 = torch.cat([xu2_, x1], 1)
        x6 = self.conv6(cat2)   # Dropout

        xu1 = self.up1(x6)
        x70 = self.conv7_k(xu1)
        x71 = self.conv7_g(xu1)

        return {'out0': x70, 'out1': x71}


if __name__ == '__main__':
    g = Generator(n_channels=3, batch_norm=False, final='tanh')
    #from torchsummary import summary
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)

    f = g(torch.rand(1, 3, 256, 256), method='encode', alpha=0)
    for i in f:
        print(i.shape)

    out = g(f, method='decode', alpha=0)