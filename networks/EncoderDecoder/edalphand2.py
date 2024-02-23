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
import functools

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


from networks.DeScarGan.descargan import get_activation


def conv2d_block(in_channels, out_channels, kernel=3, momentum=0.01, norm_type='instance', activation=ACTIVATION):
    if norm_type != 'none':
        norm_layer = get_norm_layer(norm_type)(out_channels, momentum=momentum)
    else:
        norm_layer = Identity()

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        norm_layer,
        activation(),
    )

def Identity():
    return nn.Identity()

def deconv2d_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                   norm_type='instance', activation=ACTIVATION):
    if norm_type != 'none':
        norm_layer = get_norm_layer(norm_type)(out_channels, momentum=momentum)
    else:
        norm_layer = Identity()

    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        norm_layer,
        activation(),
    )


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, activation=ACTIVATION, norm_type='instance',
                 final='tanh', mc=False):
        super(Generator, self).__init__()
        conv_block = conv2d_block
        deconv_block = deconv2d_block

        max_pool = nn.MaxPool2d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.c_dim = 0

        self.down0 = nn.Sequential(
            conv_block(n_channels + self.c_dim, nf, activation=act, norm_type=norm_type),
            conv_block(nf, nf, activation=act, norm_type=norm_type),
        )
        self.down1 = nn.Sequential(
            max_pool,
            conv_block(nf, 2 * nf, activation=act, norm_type=norm_type),
            conv_block(2 * nf, 2 * nf, activation=act, norm_type=norm_type),
        )
        self.down2 = nn.Sequential(
            max_pool,
            conv_block(2 * nf, 4 * nf, activation=act, norm_type=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act, norm_type=norm_type),

        )
        self.down3 = nn.Sequential(
            max_pool,
            conv_block(4 * nf, 8 * nf, activation=act, norm_type=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(8 * nf, 8 * nf, activation=act, norm_type=norm_type),
        )

        self.up3 = deconv_block(8 * nf, 4 * nf, activation=act, norm_type=norm_type)

        self.conv5 = nn.Sequential(
            conv_block(8 * nf, 4 * nf, activation=act, norm_type=norm_type),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv_block(4 * nf, 4 * nf, activation=act, norm_type=norm_type),
        )
        self.up2 = deconv_block(4 * nf, 2 * nf, activation=act, norm_type=norm_type)
        self.conv6 = nn.Sequential(
            conv_block(4 * nf, 2 * nf, activation=act, norm_type=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv_block(2 * nf, 2 * nf, activation=act, norm_type=norm_type),
        )

        self.up1 = deconv_block(2 * nf, nf, activation=act, norm_type=norm_type)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv_block(nf, out_channels, activation=final_layer, norm_type=norm_type),
        )

        self.conv7_g = nn.Sequential(
            conv_block(nf, out_channels, activation=final_layer, norm_type=norm_type),
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
    g = Generator(n_channels=3, final='tanh', norm_type='none')
    #from torchsummary import summary
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)

    f = g(torch.rand(1, 3, 256, 256), method='encode', alpha=0)
    for i in f:
        print(i.shape)

    out = g(f, method='decode', alpha=0)