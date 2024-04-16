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
        self.hparams.netG = 'edescarnoumc'
        self.hparams.final = 'none'
        self.net_gy, self.net_dy = self.set_networks()

        # save model names
        self.netg_names = {'net_g': 'net_g', 'net_gy': 'net_gy'}
        self.netd_names = {'net_d': 'net_d', 'net_dy': 'net_dy'}
        #self.netg_names = {'net_gy': 'net_gy'}
        #self.netd_names = {'net_dy': 'net_dy'}

        self.VGGLoss = VGGLoss()

        # Finally, initialize the optimizers and scheduler
        self.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        # coefficient for the identify loss
        parser.add_argument("--lambI", type=int, default=0.5)
        parser.add_argument("--lbvgg", dest='lbvgg', type=float, default=1)
        return parent_parser

    def test_method(self, net_g, img):
        output = net_g(img[0])
        #output = combine(output, x[0], method='mul')
        return output[0]

    def generation(self, batch):
        z_init = np.random.randint(23 - self.hparams.cropz)
        batch['img'][0] = batch['img'][0][:, :, :, :, z_init:z_init + self.hparams.cropz]
        batch['img'][1] = batch['img'][1][:, :, :, :, z_init:z_init + self.hparams.cropz]

        self.oriX = batch['img'][0]  # (B, C, X, Y, Z) # original
        self.oriY = batch['img'][1]  # (B, C, X, Y, Z) # original

        #self.Xup = self.upsample(self.oriX)  # (B, C, X, Y, Z)
        #self.Yup = self.upsample(self.oriY)  # (B, C, X, Y, Z)

        goutz = self.net_g(self.oriX, method='encode')
        self.XupX = self.net_g(goutz[-1], method='decode')['out0']

        goutzpermute = [x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0] for x in goutz]
        self.imgXY = self.net_gy(goutzpermute, method='decode')['out0']
        self.imgXY = nn.Sigmoid()(self.imgXY)  # 0-1
        #self.imgXY = combine(self.imgXY, self.get_xy_plane(self.oriX), method='mul')

        oriXxy = self.get_xy_plane(self.oriX) # -1 to 1
        imgXY = torch.mul(self.imgXY, (oriXxy + 1) / 2)  # 0-1
        self.imgXY = imgXY * 2 - 1  # -1 to 1

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

    def backward_g(self):
        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_dy, truth=True)

        loss_l1 = self.add_loss_l1(a=self.XupX[:, :, :, :, ::8], b=self.oriX[:, :, :, :, :]) * self.hparams.lamb
        loss_l1y = self.add_loss_l1(a=self.imgXY, b=self.get_xy_plane(self.oriY)) * self.hparams.lamb

        loss_gvgg = self.VGGLoss(torch.cat([self.imgXY] * 3, 1), torch.cat([self.get_xy_plane(self.oriY)] * 3, 1))

        loss_g = axx + axy + loss_l1 + loss_l1y + loss_gvgg * self.hparams.lbvgg

        return {'sum': loss_g, 'gxx': axx, 'gxy': axy, 'l1': loss_l1, 'l1Y': loss_l1y, 'vgg': loss_gvgg}

    def backward_d(self):
        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        dxy = self.add_loss_adv(a=self.imgXY, net_d=self.net_dy, truth=False)

        # ADV(X)+
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)
        dy = self.add_loss_adv(a=self.get_xy_plane(self.oriY), net_d=self.net_dy, truth=True)

        loss_d = dxx + dxy + dx + dy

        return {'sum': loss_d, 'dxx_x': dxx + dx, 'dxy_y': dxy + dy}


# USAGE
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn cyc_imorphics --prj cyc_oai3d/0/ --models cyc_oai3d --cropz 16 --cropsize 128 --netG ed03d
