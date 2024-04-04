import torch
from models.MRPretrained import MRPretrained
import torch.nn as nn


class MRPretrainedSiamese(MRPretrained):
    def __init__(self, *args, **kwargs):
        super(MRPretrainedSiamese, self).__init__(*args, **kwargs)

    def chain_multiply(self, x):
        x = 1 - x.unsqueeze(1).unsqueeze(2)
        return torch.chain_matmul(*x)

    def forward(self, x):  # (1, 3, 224, 224, 23)
        # dummies
        out = None  # output of the model
        features = None  # features we want for further analysis
        # reshape
        x0 = x[0]
        x1 = x[1]

        B = x0.shape[0]
        x0 = x0.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x0 = x0.reshape(B * x0.shape[1], x0.shape[2], x0.shape[3], x0.shape[4])  # (B*23, 3, 224, 224)
        x1 = x1.permute(0, 4, 1, 2, 3)  # (B, 23, 3, 224, 224)
        x1 = x1.reshape(B * x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4])  # (B*23, 3, 224, 224)
        # features
        x0 = self.features(x0)  # (B*23, 512, 7, 7)
        x1 = self.features(x1)  # (B*23, 512, 7, 7)

        x0 = x0.reshape(B, 23, x0.shape[1], x0.shape[2], x0.shape[3])
        x1 = x1.reshape(B, 23, x1.shape[1], x1.shape[2], x1.shape[3])  # (B, 23, 512, 7, 7)

        x0 = x0.permute(0, 2, 3, 4, 1)  # (B, 512, 7, 7, 23)
        x1 = x1.permute(0, 2, 3, 4, 1)

        if self.fuse == 'cat0':  # max-pooling across the slices
            x0 = torch.mean(x0, dim=(2, 3))  # (B, 512, 1, 1, 23)
            x1 = torch.mean(x1, dim=(2, 3))
            x0, _ = torch.max(x0, 2)
            x1, _ = torch.max(x1, 2)
            out = self.classifier_cat0(torch.cat([x0, x1], 1).unsqueeze(2).unsqueeze(3))  # (Classes)
            out = out[:, :, 0, 0]

        if self.fuse == 'max2':  # max-pooling across the slices
            x0 = torch.mean(x0, dim=(2, 3))  # (B, 512, 1, 1, 23)
            x1 = torch.mean(x1, dim=(2, 3))
            x0, _ = torch.max(x0, 2)
            x1, _ = torch.max(x1, 2)
            out = self.classifier(x0.unsqueeze(2).unsqueeze(3) - x1.unsqueeze(2).unsqueeze(3))  # (Classes)
            out = out[:, :, 0, 0]

        if self.fuse == 'max3':  # max-pooling across the slices
            x0 = nn.AdaptiveMaxPool3d((1, 1, 23))(x0)  # (B, 512, 1, 1, 23)
            x1 = nn.AdaptiveMaxPool3d((1, 1, 23))(x1)
            x0, _ = torch.max(x0, 4)
            x1, _ = torch.max(x1, 4)
            out = self.classifier(x0 - x1)  # (Classes)
            out = out[:, :, 0, 0]

        if self.fuse == 'minmax':  # max-pooling across the slices
            x0 = torch.mean(x0, dim=(2, 3))  # (B, 512, 1, 1, 23)
            x1 = torch.mean(x1, dim=(2, 3))
            x0, _ = torch.max(x0, 2)
            x1, _ = torch.max(x1, 2)
            out = self.classifier(x0.unsqueeze(2).unsqueeze(3) - x1.unsqueeze(2).unsqueeze(3))  # (Classes)
            out = out[:, :, 0, 0]

        if self.fuse == 'chain0':
            x0 = torch.mean(x0, dim=(2, 3))  # (B, 512, 1, 1, 23)
            x1 = torch.mean(x1, dim=(2, 3))
            # (B, 512, 23)
            x0 = x0.permute(0, 2, 1)
            x1 = x1.permute(0, 2, 1)
            x0 = x0.reshape(x0.shape[0] * x0.shape[1], x0.shape[2])
            x1 = x1.reshape(x1.shape[0] * x1.shape[1], x1.shape[2])
            #x0, _ = torch.max(x0, 2)
            #x1, _ = torch.max(x1, 2)
            x0 = self.classifier(x0.unsqueeze(2).unsqueeze(3))  # (Classes)
            x1 = self.classifier(x1.unsqueeze(2).unsqueeze(3))  # (Classes)
            x0 = nn.Softmax(dim=1)(x0)
            x1 = nn.Softmax(dim=1)(x1)
            x0 = 1 - x0[:, 1, 0, 0]
            x1 = 1 - x1[:, 1, 0, 0]  # (B, 1)
            x0 = x0.reshape(B, 23)
            x1 = x1.reshape(B, 23)
            x0 = torch.cat([torch.chain_matmul(*x0[i, :].unsqueeze(1).unsqueeze(2)) for i in range(B)])
            x1 = torch.cat([torch.chain_matmul(*x1[i, :].unsqueeze(1).unsqueeze(2)) for i in range(B)])
            out = torch.cat([x0, x1], 1)

        return out, [x0, x1]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.backbone = 'vgg11'
    parser.pretrained = False
    parser.n_classes = 2
    parser.fuse = 'cat'

    mr1 = MRPretrained(parser)
    out1 = mr1(torch.rand(4, 3, 224, 224, 23))