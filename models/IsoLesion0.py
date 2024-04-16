from models.base import BaseModel, combine
import copy
import torch
import numpy as np


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, eval_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, eval_loader, checkpoints)

        self.net_g, self.net_d = self.set_networks()

        # save model names
        self.netg_names = {'net_g': 'net_g'}
        self.netd_names = {'net_d': 'net_d'}

        # interpolation network
        self.upsample = torch.nn.Upsample(size=(hparams.cropsize, hparams.cropsize, hparams.cropz * 8))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        z_init = np.random.randint(23 - self.hparams.cropz)
        batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original

        self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)

        gout = self.net_g(self.Xup)

        self.XupX = gout['out0']  # (B, C, X, Y, Z)
        #self.XupY = gout['out1']  # (B, C, X, Y, Z)

    def adv_loss_six_way(self, x, truth):
        loss = 0
        loss += self.add_loss_adv(a=x.permute(2, 1, 4, 3, 0)[:, :, :, :, 0],  # (X, C, Z, Y)
                                       net_d=self.net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(2, 1, 3, 4, 0)[:, :, :, :, 0],  # (X, C, Y, Z)
                                       net_d=self.net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 4, 2, 0)[:, :, :, :, 0],  # (Y, C, Z, X)
                                       net_d=self.net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(3, 1, 2, 4, 0)[:, :, :, :, 0],  # (Y, C, X, Z)
                                       net_d=self.net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0],  # (Z, C, X, Y)
                                       net_d=self.net_d, truth=truth)
        loss += self.add_loss_adv(a=x.permute(4, 1, 3, 2, 0)[:, :, :, :, 0],  # (Z, C, Y, X)
                                       net_d=self.net_d, truth=truth)
        loss = loss / 6
        return loss

    def backward_g(self):
        loss_g_gan = 0
        #loss_g_gan += self.adv_loss_six_way(self.XupY, truth=True)
        loss_g_gan += self.adv_loss_six_way(self.XupX, truth=True)

        loss_g_l1 = self.add_loss_l1(a=self.XupX[:, :, :, :, ::8], b=self.oriX[:, :, :, :, :]) * self.hparams.lamb

        loss_g = loss_g_gan + loss_g_l1

        return {'sum': loss_g, 'loss_g_gan': loss_g_gan, 'loss_g_l1': loss_g_l1}

    def backward_d(self):
        loss_d = 0

        loss_d_gan = self.adv_loss_six_way(self.XupX, truth=False)

        loss_d += loss_d_gan

        # ADV(X)+
        Xup = self.Xup.permute(4, 1, 2, 3, 0)[::1, :, :, :, 0]  # (Z, C, X, Y) # xy plane
        loss_d += self.add_loss_adv(a=Xup, net_d=self.net_d, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn cyc_imorphics --prj cyc_oai3d/0/ --models cyc_oai3d --cropz 16 --cropsize 128 --netG ed03d
