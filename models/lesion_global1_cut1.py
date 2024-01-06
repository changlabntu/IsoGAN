import torch, copy
import torch.nn as nn
import torchvision
import torch.optim as optim
from math import log10
import time, os
import pytorch_lightning as pl
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.metrics_classification import CrossEntropyLoss, GetAUC
from utils.data_utils import *
from models.base import BaseModel, combine, VGGLoss
import pandas as pd
import numpy as np
from models.helper_oai import OaiSubjects, classify_easy_3d, swap_by_labels
from models.helper import reshape_3d, tile_like
from networks.networks_cut import Normalize, init_net, PatchNCELoss
from models.IsoGAN import PatchSampleF

class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, num_classes=2):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, 512))

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        print(batch_size)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))
        centers_batch = self.centers.gather(0, targets_expand)  # centers batch

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        print(dist)
        # for each anchor, find the hardest positive and negative
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max())  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min())  # mask[i]==0: negative samples of sample i

        for x in dist_ap:
            print(x.shape)
        for x in dist_an:
            print(x.shape)
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)  # normalize data by batch size
        return loss, prec


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)
        # First, modify the hyper-parameters if necessary
        # Initialize the networks
        self.hparams.final = 'none'
        self.net_g, self.net_d = self.set_networks()
        self.hparams.final = 'none'
        self.net_gY, _ = self.set_networks()
        self.classifier = nn.Conv2d(256, 2, 1, stride=1, padding=0).cuda()

        # update names of the models for optimization
        self.netg_names = {'net_g': 'net_g', 'net_gY': 'net_gY', 'netF': 'netF'}
        self.netd_names = {'net_d': 'net_d'}

        self.oai = OaiSubjects(self.hparams.dataset)

        if hparams.lbvgg > 0:
            self.VGGloss = VGGLoss().cuda()

        # CUT NCE
        self.featDown = nn.MaxPool2d(kernel_size=self.hparams.fDown)  # extra pooling to increase field of view

        netF = PatchSampleF(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[], nc=self.hparams.c_mlp)
        self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
        feature_shapes = [x * self.hparams.ngf for x in [1, 2, 4, 8]]
        self.netF.create_mlp(feature_shapes)

        if self.hparams.fWhich == None:  # which layer of the feature map to be considered in CUT
            self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

        print(self.hparams.fWhich)

        self.criterionNCE = []
        for nce_layer in range(4):  # self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device))

        # global contrastive
        self.batch = self.hparams.batch_size
        self.pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
        #self.pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.triple = nn.TripletMarginLoss()
        if self.hparams.projection > 0:
            self.center = CenterLoss(feat_dim=self.hparams.projection)
        else:
            self.center = CenterLoss(feat_dim=self.hparams.ngf * 8)
        self.tripletcenter = TripletCenterLoss()

        if self.hparams.projection > 0:
            self.net_g.projection = nn.Linear(self.hparams.ngf * 8, self.hparams.projection).cuda()

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument("--projection", dest='projection', type=int, default=0)
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument("--fDown", dest='fDown', type=int, default=1)
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        parser.add_argument("--adv", dest='adv', type=float, default=1)
        parser.add_argument("--lbvgg", dest='lbvgg', type=float, default=0)
        parser.add_argument("--alpha", dest='alpha', type=int,
                            help='ending epoch for decaying skip connection, 0 for no decaying', default=0)
        return parent_parser

    def generation(self, batch):
        z_init = np.random.randint(7)
        batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + 16]
        batch['img'][1] = batch['img'][1][:, :, :, :, z_init:z_init + 16]

        if self.hparams.load3d:  # if working on 3D input, bring the Z dimension to the first and combine with batch
            batch['img'] = reshape_3d(batch['img'])

        self.oriX = batch['img'][0]
        self.oriY = batch['img'][1]

        # decaying skip connection
        if self.hparams.alpha > 0:  # if decaying
            alpha = 1 - self.epoch / self.hparams.alpha
            if alpha < 0:
                alpha = 0
        else:
            alpha = 0  # if always disconnected

        # generating a mask by sigmoid to locate the lesions, turn out its the best way for now
        outXz = self.net_g(self.oriX, alpha=1, method='encode')
        outX = self.net_g(outXz, alpha=1, method='decode')
        self.imgXY = nn.Sigmoid()(outX['out0'])  # mask 0 - 1
        self.imgXY = combine(self.imgXY, self.oriX, method='mul')  # i am using masking (0-1) here

        #
        outYz = self.net_g(self.oriY, alpha=alpha, method='encode')
        outY = self.net_gY(outYz, alpha=alpha, method='decode')
        self.imgYY = nn.Tanh()(outY['out0'])  # -1 ~ 1, real img

        # global contrastive
        # use last layer
        self.outXz = outXz[-1]
        self.outYz = outYz[-1]

        (B, C, X, Y) = self.outXz.shape
        self.outXz = self.outXz.view(B//self.batch, self.batch, C, X, Y)
        self.outXz = self.outXz.permute(1, 2, 3, 4, 0)
        self.outXz = self.pool(self.outXz)[:, :, 0, 0, 0]
        if self.hparams.projection > 0:
            self.outXz = self.net_g.projection(self.outXz)

        self.outYz = self.outYz.view(B//self.batch, self.batch, C, X, Y)
        self.outYz = self.outYz.permute(1, 2, 3, 4, 0)
        self.outYz = self.pool(self.outYz)[:, :, 0, 0, 0]
        if self.hparams.projection > 0:
            self.outYz = self.net_g.projection(self.outYz)

    def backward_g(self):
        loss_dict = dict()

        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_d, truth=True)

        # L1(XY, Y)
        loss_l1 = self.add_loss_l1(a=self.imgXY, b=self.oriY)

        loss_l1Y = self.add_loss_l1(a=self.imgYY, b=self.oriY)

        loss_ga = axy  # * 0.5 + axx * 0.5

        loss_g = loss_ga * self.hparams.adv + loss_l1 * self.hparams.lamb + loss_l1Y * self.hparams.lamb

        if self.hparams.lbvgg > 0:
            loss_gvgg = self.VGGloss(torch.cat([self.imgXY] * 3, 1), torch.cat([self.oriY] * 3, 1))
            loss_g += loss_gvgg * self.hparams.lbvgg
        else:
            loss_gvgg = 0

        # CUT NCE_loss
        if self.hparams.lbNCE > 0:
            # (Y, YY) (XY, YY) (Y, XY)
            feat_q = self.net_g(self.oriY, method='encode')
            feat_k = self.net_g(self.imgXY, method='encode')

            feat_q = [self.featDown(f) for f in feat_q]
            feat_k = [self.featDown(f) for f in feat_k]

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                                None)  # get source patches by random id
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
        else:
            loss_nce = 0

        loss_g += loss_nce * self.hparams.lbNCE

        loss_dict['loss_nce'] = loss_nce
        loss_dict['loss_l1'] = loss_l1
        loss_dict['loss_l1Y'] = loss_l1Y

        # global contrastive
        loss_t = 0
        loss_t += self.triple(self.outYz[:1, ::], self.outYz[1:, ::], self.outXz[:1, ::])
        loss_t += self.triple(self.outYz[1:, ::], self.outYz[:1, ::], self.outXz[1:, ::])
        loss_center = self.center(torch.cat([f for f in [self.outXz, self.outYz]], dim=0), torch.FloatTensor([0, 0, 1, 1]).cuda())
        loss_g += loss_t + loss_center

        loss_dict['loss_t'] = loss_t
        loss_dict['loss_center'] = loss_center

        loss_dict['sum'] = loss_g

        return loss_dict

    def backward_d(self):
        # ADV(XY)-
        axy = self.add_loss_adv(a=self.imgXY.detach(), net_d=self.net_d, truth=False)

        # ADV(XX)-
        # axx, _ = self.add_loss_adv_classify3d(a=self.imgXX, net_d=self.net_dX, truth_adv=False, truth_classify=False)
        ay = self.add_loss_adv(a=self.oriY.detach(), net_d=self.net_d, truth=True)

        # adversarial of xy (-) and y (+)
        loss_da = axy * 0.5 + ay * 0.5  # axy * 0.25 + axx * 0.25 + ax * 0.25 + ay * 0.25
        # classify x (+) vs y (-)
        loss_d = loss_da * self.hparams.adv

        return {'sum': loss_d, 'da': loss_da}


# python train.py --alpha 0 --jsn womac4 --prj global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16/ --models lesion_global1_cut1 --netG edalphand --split moaks --dataset womac4 --lbvgg 0 --lbNCE 4 --nm 01 --fDown 4 --use_mlp --fWhich 0 0 1 1 -b 2 --ngf 32 --projection 32 --env runpod --n_epochs 200 --lr_policy cosine