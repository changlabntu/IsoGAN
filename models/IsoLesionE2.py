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

        self.hparams.final = 'none'
        self.net_g, self.net_d = self.set_networks()
        self.hparams.final = 'none'
        self.net_gB, _ = self.set_networks()
        self.hparams.netG = 'edescarnoumc'
        self.hparams.final = 'none'
        self.net_gy, self.net_dy = self.set_networks()
        #self.net_dymask = copy.deepcopy(self.net_dy)

        # save model names
        self.netg_names = {'net_g': 'net_g', 'net_gy': 'net_gy', 'net_gB': 'net_gB'}
        self.netd_names = {'net_d': 'net_d', 'net_dy': 'net_dy'}#, 'net_dymask': 'net_dymask'}

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
        goutzdecode = self.net_g(goutz[-1], method='decode')
        self.XupX = nn.Tanh()(goutzdecode['out0'])  # X interpolated

        goutzpermute = [x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0] for x in goutz]
        self.imgXY = self.net_gy(goutzpermute, method='decode')['out0']
        self.imgXYmask = nn.Sigmoid()(self.imgXY)  # 0-1
        #self.imgXY = combine(self.imgXY, self.get_xy_plane(self.oriX), method='mul')

        oriXxy = self.get_xy_plane(self.oriX)  # -1 to 1
        imgXY = torch.mul(self.imgXYmask, (oriXxy + 1) / 2)  # 0-1  # (MASK X IMAGE)
        self.imgXY = imgXY * 2 - 1  # -1 to 1

        # Y Interpolated
        self.YupY = nn.Tanh()(self.net_g(self.oriY)['out0'])

        # XY masking
        goutzdecodeB = self.net_gB(goutz[-1], method='decode')
        self.XupYmask = nn.Sigmoid()(goutzdecodeB['out0'])
        XupY = torch.mul(self.XupYmask, (self.XupX + 1) / 2)  # 0-1
        self.XupY = XupY * 2 - 1  # -1 to 1

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
        dict_g = {}
        loss_g = 0

        axx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=True)
        axy = self.add_loss_adv(a=self.imgXY, net_d=self.net_dy, truth=True)

        loss_l1 = self.add_loss_l1(a=self.XupX[:, :, :, :, ::8], b=self.oriX[:, :, :, :, :]) * self.hparams.lamb

        loss_l1y = self.add_loss_l1(a=self.imgXY, b=self.get_xy_plane(self.oriY)) * self.hparams.lamb

        loss_gvgg = self.VGGLoss(torch.cat([self.imgXY] * 3, 1), torch.cat([self.get_xy_plane(self.oriY)] * 3, 1))

        loss_g += axx + axy + loss_l1 + loss_l1y + loss_gvgg * self.hparams.lbvgg
        dict_g['gxx'] = axx
        dict_g['gxy'] = axy
        dict_g['l1'] = loss_l1
        dict_g['l1Y'] = loss_l1y
        dict_g['vgg'] = loss_gvgg

        # l1y3d
        #loss_l1y3d = self.add_loss_l1(a=self.XupY[:, :, :, :, ::8],
        #                              b=self.oriY[:, :, :, :, :]) * self.hparams.lamb  # OPTION 1
        loss_l1y3d = self.add_loss_l1(a=self.XupY[:, :, :, :, :],
                                      b=self.YupY[:, :, :, :, :]) * self.hparams.lamb  # OPTION 2
        dict_g['l1Y3d'] = loss_l1y3d
        loss_g += loss_l1y3d

        #axy3d = self.adv_loss_six_way(self.XupYmask, net_d=self.net_dymask, truth=True)
        #loss_g += axy3d

        dict_g['sum'] = loss_g
        return dict_g

    def backward_d(self):
        dict_d = {}
        loss_d = 0

        dxx = self.adv_loss_six_way(self.XupX, net_d=self.net_d, truth=False)
        dxy = self.add_loss_adv(a=self.imgXY, net_d=self.net_dy, truth=False)

        # ADV(X)+
        dx = self.add_loss_adv(a=self.get_xy_plane(self.oriX), net_d=self.net_d, truth=True)
        dy = self.add_loss_adv(a=self.get_xy_plane(self.oriY), net_d=self.net_dy, truth=True)
        dict_d['dxx_x'] = dxx + dx
        dict_d['dxy_y'] = dxy + dy
        loss_d += dxx + dxy + dx + dy

        #dxy3d = self.adv_loss_six_way(self.XupYmask, net_d=self.net_dymask, truth=False)
        #dxy2d = self.add_loss_adv(a=self.imgXYmask, net_d=self.net_dymask, truth=True)

        #dict_d['dxy3d'] = dxy3d + dxy2d
        #loss_d += dxy3d + dxy2d

        dict_d['sum'] = loss_d

        return dict_d


# USAGE
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --jsn cyc_imorphics --prj cyc_oai3d/0/ --models cyc_oai3d --cropz 16 --cropsize 128 --netG ed03d
