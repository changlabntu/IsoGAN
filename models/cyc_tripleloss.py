from models.base import BaseModel, combine
import copy
import torch
import torch.nn as nn


class GAN(BaseModel):
    """
    There is a lot of patterned noise and other failures when using lightning
    """
    def __init__(self, hparams, train_loader, test_loader, checkpoints):
        BaseModel.__init__(self, hparams, train_loader, test_loader, checkpoints)

        self.net_g, self.net_d = self.set_networks()

        self.net_gXY = self.net_g
        self.net_gYX = copy.deepcopy(self.net_g)

        self.net_dX = self.net_d
        self.net_dY = copy.deepcopy(self.net_d)
        self.triple = nn.TripletMarginLoss()

        # save model names
        self.netg_names = {'net_gXY': 'net_gXY', 'net_gYX': 'net_gYX'}
        self.netd_names = {'net_dX': 'netDX', 'net_dY': 'netDY'}

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
        img = batch['img']
        self.batch = self.hparams.batch_size
        self.pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))

        self.oriX = img[0]
        self.oriY = img[1]

        self.imgXY = self.net_gXY(self.oriX, alpha=1)['out0']
        self.imgYX = self.net_gYX(self.oriY, alpha=1)['out0']

        if self.hparams.lamb > 0:
            self.imgXYX = self.net_gYX(self.imgXY, alpha=1)['out0']
            self.imgYXY = self.net_gXY(self.imgYX, alpha=1)['out0']

        if self.hparams.lambI > 0:
            self.idt_X = self.net_gYX(self.oriX, alpha=1)['out0']
            self.idt_Y = self.net_gXY(self.oriY, alpha=1)['out0']

        outXz = self.net_gXY(self.oriX, alpha=1, method='encode')
        outYz = self.net_gYX(self.oriY, alpha=1, method='encode')
        self.outXz = outXz[-1]
        # print('gen ouXz[-1]', self.outXz.shape)
        # print('gennartion outXz[-1]', self.outXz.shape)#torch.Size([1, 512, 16, 16])
        self.outYz = outYz[-1]
        # print('gen ouYz[-1]', self.outYz.shape)
        # assert 0
        # print('gennartion outYz[-1]', self.outYz.shape)#torch.Size([1, 512, 16, 16])
        self.outXz = self.classify(self.outXz)
        # print('gennartion outXz classify', self.outXz.shape)#torch.Size([1, 512])
        self.outYz = self.classify(self.outYz)

    def classify(self, f):
        (B, C, X, Y) = f.shape
        f = f.view(B//self.batch, self.batch, C, X, Y)
        f = f.permute(1, 2, 3, 4, 0)
        f = self.pool(f)[:, :, 0, 0, 0]
        return f

    def backward_g(self):
        loss_dict = dict()
        loss_g = 0
        # ADV(XY)+
        loss_g += self.add_loss_adv(a=self.imgXY, net_d=self.net_dY, truth=True)
        # ADV(YX)+
        loss_g += self.add_loss_adv(a=self.imgYX, net_d=self.net_dX, truth=True)

        # Cyclic(XYX, X)
        if self.hparams.lamb > 0:
            loss_g += self.add_loss_l1(a=self.imgXYX, b=self.oriX) * self.hparams.lamb
            # Cyclic(YXY, Y)
            loss_g += self.add_loss_l1(a=self.imgYXY, b=self.oriY) * self.hparams.lamb

        # Identity(idt_X, X)
        if self.hparams.lambI > 0:
            loss_g += self.add_loss_l1(a=self.idt_X, b=self.oriX) * self.hparams.lambI
            # Identity(idt_Y, Y)
            loss_g += self.add_loss_l1(a=self.idt_Y, b=self.oriY) * self.hparams.lambI

        loss_t = 0
        loss_t += self.triple(self.outYz[:1, ::], self.outYz[1:, ::], self.outXz[:1, ::])
        loss_t += self.triple(self.outYz[1:, ::], self.outYz[:1, ::], self.outXz[1:, ::])

        loss_g = loss_t * 15 + loss_g*0.6
        loss_dict['sum'] = loss_g
        loss_dict['loss_g'] = loss_g
        loss_dict['loss_t'] = loss_t

        # return {'sum': loss_g, 'loss_g': loss_g, 'loss_t': loss_t}
        return loss_dict


    def backward_d(self):
        loss_d = 0
        # ADV(XY)-
        loss_d += self.add_loss_adv(a=self.imgXY, net_d=self.net_dY, truth=False)

        # ADV(YX)-
        loss_d += self.add_loss_adv(a=self.imgYX, net_d=self.net_dX, truth=False)

        # ADV(Y)+
        loss_d += self.add_loss_adv(a=self.oriY, net_d=self.net_dY, truth=True)

        # ADV(X)+
        loss_d += self.add_loss_adv(a=self.oriX, net_d=self.net_dX, truth=True)

        return {'sum': loss_d, 'loss_d': loss_d}


# USAGE
# CUDA_VISIBLE_DEVICES=0 python train.py --prj train --jsn wbc --save_logs cyc_tripleloss/edalphand_nm01_0223 --nm 01 --netG edalphand --models cyc_tripleloss




import torch
anchor_embed = torch.randn(10, 512)
positive_embed = torch.randn(10, 512)
negative_embeds = torch.randn(10, 512)

positive_distances = torch.norm(anchor_embed - positive_embed, p=2, dim=1)

# Expand dimensions to allow broadcasting
anchor_embed_exp = anchor_embed.unsqueeze(1)
negative_embeds_exp = negative_embeds.unsqueeze(0)

# Calculate distances between anchors and all negatives
negative_distances = torch.norm(anchor_embed_exp - negative_embeds_exp, p=2, dim=2)

# For each anchor-positive pair, find the hardest negative
# The hardest negative is the one with the smallest distance to the anchor
# that is still greater than the distance to the positive
negative_distances[negative_distances <= positive_distances.unsqueeze(1)] = float('inf')
hardest_negative_indices = torch.argmin(negative_distances, dim=1)
hardest_negatives = negative_embeds[hardest_negative_indices]