from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
from models.base import VGGLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss


class PatchSampleF3D(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF3D, self).__init__()
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
            # (B, C, H, W, Z)
            # feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            feat = feat.permute(0, 2, 3, 4, 1)  # (B, H*W*Z, C)
            feat_reshape = feat.reshape(feat.shape[0], feat.shape[1] * feat.shape[2] * feat.shape[3], feat.shape[4])
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
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()
        self.hparams.final = 'tanh'
        self.net_gback, self.net_dzy = self.set_networks()
        self.net_dzx = copy.deepcopy(self.net_dzy)

        # save model names
        self.netg_names = {'net_g': 'net_g', 'net_gback': 'net_gback'}
        self.netd_names = {'net_d': 'net_d', 'net_dzy': 'net_dzy', 'net_dzx': 'net_dzx'}
        #self.netg_names = {'net_gy': 'net_gy'}
        #self.netd_names = {'net_dy': 'net_dy'}

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropsize), mode='trilinear')

        self.uprate = (hparams.cropsize // hparams.cropz)
        print('uprate: ' + str(self.uprate))

        # CUT NCE
        if not self.hparams.nocut:
            netF = PatchSampleF3D(use_mlp=self.hparams.use_mlp, init_type='normal', init_gain=0.02, gpu_ids=[],
                                nc=self.hparams.c_mlp)
            self.netF = init_net(netF, init_type='normal', init_gain=0.02, gpu_ids=[])
            feature_shapes = [x * self.hparams.ngf for x in [1, 2, 4, 8]]
            self.netF.create_mlp(feature_shapes)

            if self.hparams.fWhich == None:  # which layer of the feature map to be considered in CUT
                self.hparams.fWhich = [1 for i in range(len(feature_shapes))]

            print(self.hparams.fWhich)

            self.criterionNCE = []
            for nce_layer in range(4):  # self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt=hparams))  # .to(self.device))

            self.netg_names['netF'] = 'netF'

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        #parser.add_argument("--nocyc", action='store_true')
        parser.add_argument("--nocut", action='store_true')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        parser.add_argument('--transpose', action='store_true', help='transpose to the yz plane')
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original

        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)

        if self.hparams.transpose:
            self.Xup = self.Xup.permute(0, 1, 3, 4, 2)

        self.goutz = self.net_g(self.Xup, method='encode')
        self.XupX = self.net_g(self.goutz[-1], method='decode')['out0']

        #self.XupXback = self.net_gback(self.XupX)['out0']

    def get_xy_plane(self, x):
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]

    def adv_loss_six_way(self, x, net_d, truth):
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                       net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                       net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                       net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                       net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],  # (Z, C, X, Y)
                                       net_d=net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 3, 2, 0)[:, :, :, :, 0],  # (Z, C, Y, X)
                                       net_d=net_d, truth=truth)
        loss = loss / 6
        return loss

    def adv_loss_six_way_y(self, x, truth):
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                        net_d=self.net_dzy, truth=truth)
        loss += self.add_loss_adv(a=x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                        net_d=self.net_dzy, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                        net_d=self.net_dzx, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                        net_d=self.net_dzx, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],  # (Z, C, X, Y)
                                        net_d=self.net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 3, 2, 0)[:, :, :, :, 0],  # (Z, C, Y, X)
                                        net_d=self.net_d, truth=truth)
        loss = loss / 6
        return loss

    def backward_g(self):
        loss_g = 0

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)
        loss_g += axx

        loss_l1 = self.add_loss_l1(a=self.XupX[:, :, :, :, ::self.uprate*4], b=self.Xup[:, :, :, :, ::self.uprate*4]) * self.hparams.lamb
        loss_g += loss_l1

        #gback = self.adv_loss_six_way_y(self.XupXback, truth=True)
        #loss_g += gback
        # Cyclic(XYX, X)
        #if self.hparams.lamb > 0:
        #    loss_g += self.add_loss_l1(a=self.XupXback, b=self.Xup) * self.hparams.lamb

        if not self.hparams.nocut:
            # (X, XupX)
            #self.goutz = self.net_g(self.Xup, method='encode')
            feat_q = self.goutz
            feat_k = self.net_g(self.XupX, method='encode')

            feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches,
                                                None)  # get source patches by random id
            feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)  # use the ids for query target

            total_nce_loss = 0.0
            for f_q, f_k, crit, f_w in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.hparams.fWhich):
                loss = crit(f_q, f_k) * f_w
                total_nce_loss += loss.mean()
            loss_nce = total_nce_loss / 4
            loss_g += loss_nce

        return {'sum': loss_g, 'gxx': axx, 'nce': loss_nce}

    def backward_d(self):
        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)

        # ADV(X)+
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)

        # ADV dyy
        #dyy = self.adv_loss_six_way_y(self.XupXback, truth=False)
        #dy = self.adv_loss_six_way_y(self.oriX, truth=True)

        loss_d = dxx + dx #+ dyy + dy

        return {'sum': loss_d, 'dxx_x': dxx + dx}#, 'dyy': dyy + dy}


# USAGE
# python train.py --jsn cyc_imorphics --prj IsoScope0/0 --models IsoScope0 --cropz 16 --cropsize 128 --netG ed023d --env t09 --adv 1 --rotate --ngf 64 --direction xyori --nm 11 --dataset longdent