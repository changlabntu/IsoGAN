from models.base import BaseModel
import copy
import torch
import numpy as np
import torch.nn as nn
from models.base import VGGLoss

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

        if self.hparams.cropz > 0:
            self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropz * hparams.uprate), mode='trilinear')
        else:
            self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, 32 * hparams.uprate), mode='trilinear')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        parser.add_argument("--uprate", type=int, default=4)
        parser.add_argument("--nocyc", action='store_true')
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        if self.hparams.cropz > 0:
            z_init = np.random.randint(batch['img'][0].shape[4] - self.hparams.cropz)
            batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]
        #batch['img'][1] = batch['img'][1][:, :, :, :, z_init:z_init + self.hparams.cropz]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original
        #self.oriY = batch['img'][1]  # (B, C, X, Y, Z) # original

        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)
        #self.Yup = self.upsample(self.oriY)  # (B, C, X, Y, Z)

        goutz = self.net_g(self.Xup, method='encode')
        self.XupX = self.net_g(goutz[-1], method='decode')['out0']

        if not self.hparams.nocyc:
            self.XupXback = self.net_gback(self.XupX)['out0']

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

        if not self.hparams.nocyc:
            gback = self.adv_loss_six_way_y(self.XupXback, truth=True)
            loss_dict['gback'] = gback
            loss_g += gback
            if self.hparams.lamb > 0:
                loss_g += self.add_loss_l1(a=self.XupXback, b=self.Xup) * self.hparams.lamb

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

        # ADV dyy
        if not self.hparams.nocyc:
            dyy = self.adv_loss_six_way_y(self.XupXback, truth=False)
            dy = self.adv_loss_six_way_y(self.oriX, truth=True)

            loss_dict['dyy'] = dyy + dy
            loss_d += dyy + dy

        loss_dict['sum'] = loss_d

        return loss_dict


# USAGE
# python train.py --jsn cyc_imorphics --prj IsoScope0/0 --models IsoScope0 --cropz 16 --cropsize 128 --netG ed023d --env t09 --adv 1 --rotate --ngf 64 --direction xyori --nm 11 --dataset longdent
