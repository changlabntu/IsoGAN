import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


def get_one_out(x0, model):
    x0 = x0.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

    out_all = model(x0.cuda(), method='encode')[-1].cpu().detach()
    out_all = model(out_all.cuda(), method='decode')['out0'].cpu().detach()

    out_all = out_all.numpy()[0, 0, :, :, :]
    return out_all


def reverse_log(x):
    return np.power(10, x)


def test_IsoScope():
    dataset = 'Dayu1'

    # no cyc
    if 1:
        model = torch.load(
            '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXXcyc0/ngf32tryB/checkpoints/net_gback_model_epoch_800.pth',
            map_location=torch.device('cpu')).cuda()  # .eval() # newly ran
        # with cyc
    else:
        model = torch.load(
        '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXXcyc0/ngf32tryB/checkpoints/net_g_model_epoch_900.pth',
        map_location=torch.device('cpu')).cuda()  # .eval() # newly ran

    model = model.eval()

    x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyori.tif')

    trd = 4240
    x0[x0 >= trd] = trd
    x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2

    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()

    # x0 (B, C, Z, X, Y) original
    # stack = x0[:, :, 256-16:256, 256:256+512, 256:256+512]
    # x0 (B, C, Z, Y, X) input
    # yx = model(stack.permute(0, 1, 2, 4, 3).cuda())['out0'].cpu().detach().permute(0, 1, 2, 4, 3)

    tiff.imwrite('stack.tif', stack.squeeze().numpy())
    tiff.imwrite('stackyx.tif', yx.squeeze().numpy())


#dx = 512; dy = 512; dz = 16; sx = 0; sy = 0; sz = 0; ox = 0; oy = 0; oz = 0
def testing_over_roi(x0, roi_name, model, dx=512, dy=512, dz=16, sx=0, sy=0, sz=0, ox=256, oy=256, oz=0):
    stepx = dx - ox
    stepy = dy - oy
    stepz = dz - oz
    for z in range(0 + sz, x0.shape[2] - dz + sz + 1, stepz)[:]:
        for x in range(0 + sx, x0.shape[3] - dx + sx + 1, stepx)[:]:
            for y in range(0 + sy, x0.shape[4] - dy + sy + 1, stepy)[:]:
                stack = x0[:, :, z:z + dz, x:x + dx, y:y + dy]
                yx = model(stack.permute(0, 1, 2, 4, 3).cuda())['out0'].cpu().detach().permute(0, 1, 2, 4, 3)

                # normalize yx to the mean and std of stack
                yx = yx - yx.mean()
                yx = yx / yx.std()
                yx = yx * stack.std() + stack.mean()
                print('stack', stack.mean(), stack.std(), 'yx', yx.mean(), yx.std())

                for zz in range(dz):
                    filename = roi_name + '_' + str(x).zfill(5) + str(y).zfill(5) + str(z + zz).zfill(5) + '.tif'
                    tiff.imwrite(destination + 'xyori/' + filename, stack.squeeze()[zz, 64:-64, 64:-64].numpy())
                    tiff.imwrite(destination + 'xygan/' + filename, yx.squeeze()[zz, 64:-64, 64:-64].numpy())


if __name__ == '__main__':
    dataset = 'DPM4X'
    model = torch.load(
        '/media/ExtHDD01/logs/' + 'Dayu1' + '/IsoScopeXXcyc0/ngf32tryB/checkpoints/net_gback_model_epoch_800.pth',
        map_location=torch.device('cpu')).cuda()  # .eval() # newly ran

    model = model.eval()

    destination = '/media/ExtHDD01/Dataset/paired_images/DPM4X/cycout/'

    for roi_name in ['3-2ROI000', '3-2ROI002', '3-2ROI006', '3-2ROI008'][1:]:
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/' + roi_name + '.tif')
        trd = 4240
        x0[x0 >= trd] = trd
        x0 = x0 / x0.max()
        x0 = (x0 - 0.5) * 2
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()

        testing_over_roi(x0, roi_name, model)







