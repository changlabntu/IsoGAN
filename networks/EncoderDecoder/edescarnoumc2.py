import torch
import torch.nn as nn

from networks.DeScarGan.descargan import conv2d_block, conv2d_bn_block, deconv2d_bn_block, get_activation, deconv2d_block


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, batch_norm=True, activation=nn.ReLU, final='tanh', mc=False):
        super(Generator, self).__init__()

        conv_block = conv2d_bn_block if batch_norm else conv2d_block
        deconv_block = deconv2d_bn_block if batch_norm else deconv2d_block

        max_pool = nn.MaxPool2d(2)
        act = activation
        self.c_dim = 0

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

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

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)

    def forward(self, x, a=None, method=None):
        if method != 'decode':
            x0 = self.down0(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            feat = [x0, x1, x2, x3]
            if method == 'encode':
                return feat

        if method == 'decode':
            [x0, x1, x2, x3] = x

        xu3 = self.up3(x3)
        xu2 = self.up2(xu3)
        xu1 = self.up1(xu2)

        x70 = self.conv7_k(xu1)
        x71 = self.conv7_g(xu1)

        return {'out0': x70, 'out1': x71, 'z': x3}


if __name__ == '__main__':
    g = Generator(n_channels=3, nf=64, batch_norm=False, final='tanh')
    #out = g(torch.rand(1, 3, 256, 256), a=None)
    from utils.data_utils import print_num_of_parameters
    print_num_of_parameters(g)
