from models.base import BaseModel, combine
import copy
import torch
import torch.nn as nn
import tifffile as tiff
import os
from dotenv import load_dotenv
from networks.networks_cut import Normalize, init_net, PatchNCELoss
import numpy as np
load_dotenv('.env')


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feature_shapes):
        for mlp_id, feat in enumerate(feature_shapes):
            input_nc = feat
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # if len(self.gpu_ids) > 0:
            # mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        # print(len(feats))
        # print([x.shape for x in feats])
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            # B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            # (B, C, H, W)
            # feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            feat = feat.permute(0, 2, 3, 1)  # (B, H*W, C)
            feat_reshape = feat.view(feat.shape[0], feat.shape[1] * feat.shape[2], feat.shape[3])
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    # patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])  # (random order of range(H*W))
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device) # first N patches
                    # patch_id = torch.from_numpy(patch_id).type(torch.long).to(feat.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)  # Channel (1, 128, 256, 256, 256) > (256, 256, 256, 256, 256)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        # print([x.shape for x in return_feats]) # (B * num_patches, 256) * level of features
        return return_feats, return_ids

class GAN(BaseModel):
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        self.net_g, self.net_d = self.set_networks()
        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dXo = copy.deepcopy(self.net_d)
        self.net_dXw = copy.deepcopy(self.net_d)
        self.net_dYo = copy.deepcopy(self.net_d)
        self.net_dYw = copy.deepcopy(self.net_d)

        # save model names
        self.netg_names = {'net_gXY': 'netGXY', 'net_gYX': 'netGYX', 'netF': 'netF', 'netFB': 'netFB'}
        self.netd_names = {'net_dXw': 'netDXw', 'net_dYw': 'netDYw', 'net_dXo': 'netDXo', 'net_dYo': 'netDYo'}

        if self.hparams.downZ > 0:
            self.upz = nn.Upsample(size=(self.hparams.cropsize, self.hparams.cropsize), mode='bilinear').cuda()

        # CUT NCE
        netF = PatchSampleF(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[],
                            nc=self.hparams.c_mlp)
        self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
        feature_shapes = [x * self.hparams.ngf for x in [1, 2, 4, 8]]
        self.netF.create_mlp(feature_shapes)

        netFB = PatchSampleF(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[],
                             nc=self.hparams.c_mlp)
        self.netFB = init_net(netFB, init_type='normal', init_gain=0.02, gpu_ids=[])
        feature_shapes = [x * self.hparams.ngf for x in [1, 2, 4, 8]]
        self.netFB.create_mlp(feature_shapes)

        if self.hparams.fWhich == None:  # which layer of the feature map to be considered in CUT
            self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

        print(self.hparams.fWhich)

        self.criterionNCE = []
        for nce_layer in range(4):  # self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # CUT NCE
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=float, default=0.5)
        parser.add_argument("--downZ", type=int, default=0)
        return parent_parser

    def test_method(self, net_g, x):
        output, output1 = net_g(torch.cat((x[0], x[1]), 1), a=None)
        return output1

    def generation(self, batch):  # 0
        # zyweak_zyorisb%xyweak_xyorisb
        img = batch['img']

        self.oriXw = img[0]
        self.oriXo = img[1]

        #print((self.oriXw.min(), self.oriXw.max()))
        #print((self.oriXo.min(), self.oriXo.max()))

        if self.hparams.downZ > 0:
            self.oriXw = self.oriXw[:, :, ::self.hparams.downZ, :]
            self.oriXo = self.oriXo[:, :, ::self.hparams.downZ, :]
            self.oriXw = self.upz(self.oriXw)
            self.oriXo = self.upz(self.oriXo)

        self.oriYw = img[2]
        self.oriYo = img[3]

        outXY = self.net_gXY(torch.cat([img[0], img[1]], 1), a=None)
        outYX = self.net_gYX(torch.cat([img[2], img[3]], 1), a=None)
        self.imgXYw, self.imgXYo = outXY['out0'], outXY['out1']
        self.imgYXw, self.imgYXo = outYX['out0'], outYX['out1']

        outXYX = self.net_gYX(torch.cat([self.imgXYw, self.imgXYo], 1), a=None)
        outYXY = self.net_gXY(torch.cat([self.imgYXw, self.imgYXo], 1), a=None)
        self.imgXYXw, self.imgXYXo = outXYX['out0'], outXYX['out1']
        self.imgYXYw, self.imgYXYo = outYXY['out0'], outYXY['out1']

        if self.hparams.lambI > 0:
            outidtX = self.net_gYX(torch.cat([img[0], img[1]], 1), a=None)
            outidtY = self.net_gXY(torch.cat([img[2], img[3]], 1), a=None)
            self.idt_Xw, self.idt_Xo = outidtX['out0'], outidtX['out1']
            self.idt_Yw, self.idt_Yo = outidtY['out0'], outidtY['out1']

    def backward_g(self):
        loss_g = 0
        # ADV(XYw)+
        loss_g += self.add_loss_adv(a=self.imgXYw, net_d=self.net_dYw, truth=True)
        # ADV(YXw)+
        loss_g += self.add_loss_adv(a=self.imgYXw, net_d=self.net_dXw, truth=True)
        # ADV(XYo)+
        loss_g += self.add_loss_adv(a=self.imgXYo, net_d=self.net_dYo, truth=True)
        # ADV(YXo)+
        loss_g += self.add_loss_adv(a=self.imgYXo, net_d=self.net_dXo, truth=True)

        # Cyclic(XYXw, Xw)
        loss_g += self.add_loss_l1(a=self.imgXYXw, b=self.oriXw) * self.hparams.lamb
        # Cyclic(YXYw, Yw)
        loss_g += self.add_loss_l1(a=self.imgYXYw, b=self.oriYw) * self.hparams.lamb
        # Cyclic(XYXo, Xo)
        loss_g += self.add_loss_l1(a=self.imgXYXo, b=self.oriXo) * self.hparams.lamb
        # Cyclic(YXYo, Yo)
        loss_g += self.add_loss_l1(a=self.imgYXYo, b=self.oriYo) * self.hparams.lamb

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            # Identity(idt_Xw, Xw)
            loss_g += self.add_loss_l1(a=self.idt_Xw, b=self.oriXw) * self.hparams.lambI
            # Identity(idt_Yw, Yw)
            loss_g += self.add_loss_l1(a=self.idt_Yw, b=self.oriYw) * self.hparams.lambI
            # Identity(idt_Xo, Xo)
            loss_g += self.add_loss_l1(a=self.idt_Xo, b=self.oriXo) * self.hparams.lambI
            # Identity(idt_Yo, Yo)
            loss_g += self.add_loss_l1(a=self.idt_Yo, b=self.oriYo) * self.hparams.lambI

        # CUT NCE_loss
        if self.hparams.lbNCE > 0:
            # (Y, YX)
            feat_q = self.net_gYX(torch.cat([self.oriYw, self.oriYo], 1), method='encode')
            feat_k = self.net_gYX(torch.cat([self.imgYXw, self.imgYXo], 1), method='encode')

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                                None)  # get source patches by random id
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
            loss_g += loss_nce

            # (X, XYX)
            feat_q = self.net_gXY(torch.cat([self.oriXw, self.oriXo], 1), method='encode')
            feat_k = self.net_gXY(torch.cat([self.imgXYXw, self.imgXYXo], 1), method='encode')

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                                None)  # get source patches by random id
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
            loss_g += loss_nce
        else:
            loss_nce = 0

        return {'sum': loss_g, 'loss_g': loss_g, 'loss_nce': loss_nce}

    def backward_d(self):
        loss_d = 0
        # ADV(XY)-
        loss_d += self.add_loss_adv(a=self.imgXYw, net_d=self.net_dYw, truth=False)
        loss_d += self.add_loss_adv(a=self.imgXYo, net_d=self.net_dYo, truth=False)

        # ADV(YX)-
        loss_d += self.add_loss_adv(a=self.imgYXw, net_d=self.net_dXw, truth=False)
        loss_d += self.add_loss_adv(a=self.imgYXo, net_d=self.net_dXo, truth=False)

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriYw, net_d=self.net_dYw, truth=True)
        loss_d += self.add_loss_adv(a=self.oriYo, net_d=self.net_dYo, truth=True)

        # ADV(X)+
        loss_d += self.add_loss_adv(a=self.oriXw, net_d=self.net_dXw, truth=True)
        loss_d += self.add_loss_adv(a=self.oriXo, net_d=self.net_dXo, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --jsn wnwp3d --prj cyc4_1024/cut0nomcNofdownMLP --models cyc4_cut -b 16 --direction zyft0_zyori%xyft0_xyori --trd 2000 --nm 11 --netG edescarnoumc --split all --env runpod --dataset_mode PairedSlices --use_mlp