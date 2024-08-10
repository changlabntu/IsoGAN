import torch
import torch.nn as nn
from networks.resunet.modules import ResidualConv, Upsample


class ResUnet(nn.Module):
    def __init__(self, channel, nf=16):
        super(ResUnet, self).__init__()

        filters = [nf*2, nf*4, nf*8, nf*16]

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 1, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 1, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 1, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 1, 1, 1),
            nn.Sigmoid(),
        )

        self.pool23d = Pooling23D()

    def forward(self, x, method=None):
        # Encode
        if method != 'decode':
            x = x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
            x1 = self.input_layer(x) + self.input_skip(x)
            print(x1.shape)
            x1 = self.pool23d(x1)
            x2 = self.residual_conv_1(x1)
            print(x2.shape)
            x2 = self.pool23d(x2)
            x3 = self.residual_conv_2(x2)
            print(x3.shape)
            # Bridge
            x3 = self.pool23d(x3)
            x4 = self.bridge(x3)
            print(x4.shape)

            feat = [x.permute(1, 2, 3, 0).unsqueeze(0) for x in (x1, x2, x3, x4)]
            if method == 'encode':
                return feat

        if method == 'decode':
            x1, x2, x3, x4 = x

        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output


class Pooling23D(nn.Module):
    def __init__(self):
        super(Pooling23D, self).__init__()
        self.max3_pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        # Input shape: (Z, C, X, Y)
        x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
        x = self.max3_pool(x)
        x = x.squeeze(0).permute(3, 0, 1, 2)  # (Z, C, X, Y)
        return x


if __name__ == '__main__':
    model = ResUnet(1)
    x = torch.randn(1, 1, 128, 128, 64)
    z = model(x, method='encode')
    out = model(z, method='decode')