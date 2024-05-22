from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
from models.base import VGGLoss
from networks.networks_cut import Normalize, init_net, PatchNCELoss


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        self.hparams.final = 'tanh'
        self.net_g, self.net_d = self.set_networks()
        self.hparams.final = 'tanh'

        # save model names
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()

        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropz * hparams.uprate), mode='trilinear')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        parser.add_argument("--uprate", type=int, default=4)
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--lbNCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument("--c_mlp", dest='c_mlp', type=int, default=256, help='channel of mlp')
        parser.add_argument('--fWhich', nargs='+', help='which layers to have NCE loss', type=int, default=None)
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        x = batch['img'][0]
        x = x.permute(0, 1, 3, 4, 2)  # (move the axis required enhaccement to the last dim)
        # batch  (B, C, X, Y, Z)  (1, 1, 240, 240, 155)
        # then should be cropped to (1, 1, 128, 128, 128)

        z_init = np.random.randint(x.shape[4] - self.hparams.cropz)
        x_init = np.random.randint(x.shape[2] - self.hparams.crop)
        y_init = np.random.randint(x.shape[3] - self.hparams.crop)
        x = x[:, :, x_init:x_init + self.hparams.crop, y_init:y_init + self.hparams.crop, z_init:z_init + self.hparams.cropz]

        # (original)

        # downsample
        x = x[:, :, :, :, ::uprate] # (we are using ::4 here, but nn.upsample is also possible)

        # upsample
        self.oriX = x  # (B, C, X, Y, Z) # original
        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)

        # (upsample finished)

        self.goutz = self.net_g(self.Xup, method='encode')
        self.XupX = self.net_g(self.goutz[-1], method='decode')['out0']


    def get_xy_plane(self, x):  # (B, C, X, Y, Z)
        return x.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]  # (Z, C, X, Y, B)

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
        loss_dict = {}

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)
        loss_l1 = self.add_loss_l1(a=self.XupX[:, :, :, :, ::self.hparams.uprate], b=self.oriX[:, :, :, :, :]) * self.hparams.lamb

        loss_dict['axx'] = axx
        loss_g += axx
        loss_dict['l1'] = loss_l1
        loss_g += loss_l1

        loss_dict['sum'] = loss_g

        return loss_dict

    def backward_d(self):
        loss_d = 0
        loss_dict = {}

        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        # ADV(X)+
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)

        loss_dict['dxx_x'] = dxx + dx
        loss_d += dxx + dx

        loss_dict['sum'] = loss_d

        return loss_dict


# USAGE
# python train.py --jsn cyc_imorphics --prj IsoScope0/0 --models IsoScope0 --cropz 16 --cropsize 128 --netG ed023d --env t09 --adv 1 --rotate --ngf 64 --direction xyori --nm 11 --dataset longdent
