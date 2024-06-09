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


#dx = 512; dy = 512; dz = 16; sx = 0; sy = 0; sz = 0; ox = 0; oy = 0; oz = 0
def testing_over_roi(x0, x1, roi_name, model, dx=512, dy=512, dz=16, sx=0, sy=0, sz=0, ox=256, oy=256, oz=0):

    stepx = dx - ox
    stepy = dy - oy
    stepz = dz - oz
    for z in range(0 + sz, x0.shape[2] - dz + sz + 1, stepz)[:]:
        for x in range(0 + sx, x0.shape[3] - dx + sx + 1, stepx)[:]:
            for y in range(0 + sy, x0.shape[4] - dy + sy + 1, stepy)[:]:

                stackx0 = x0[:, :, z:z + dz, x:x + dx, y:y + dy].cuda()
                stackx1 = x1[:, :, z:z + dz, x:x + dx, y:y + dy].cuda()

                for zz in range(stackx0.shape[2]):
                    patchx0 = stackx0[:, :, zz, :, :]
                    patchx1 = stackx1[:, :, zz, :, :]
                    out = model(torch.cat([patchx1, patchx0], 1))['out1']

                    if patchx0.mean().item() > -0.6:
                        filename =  str(x).zfill(5) + str(y).zfill(5) + str(z + zz).zfill(5) + '.tif'
                        #filename = str(patchx0.mean().item()) + '.tif'
                        tiff.imwrite(destination + 'xyori/' + filename, patchx0.squeeze()[64:-64, 64:-64].cpu().detach().numpy().astype(np.float32))
                        tiff.imwrite(destination + 'xygan/' + filename, out.squeeze()[64:-64, 64:-64].cpu().detach().numpy().astype(np.float32))


def trd_and_rescale(x0, trd):
    x0[x0 >= trd] = trd
    x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    return x0


if __name__ == '__main__':
    dataset = 'Fly0B'
    if 0:
        model = torch.load(
            '/media/ExtHDD01/logs/' + dataset + '/cyc4_1024/cutFoo/checkpoints/netGYX_model_epoch_180.pth',
            map_location=torch.device('cpu')).cuda()  # .eval()
    else:
        model = torch.load(
            '/media/ExtHDD01/logs/' + dataset + '/cyc4_1024/cutF/checkpoints/netGYX_model_epoch_200.pth',
            map_location=torch.device('cpu')).cuda()  # .eval()

    model = model.eval()

    destination = '/media/ExtHDD01/Dataset/paired_images/' + dataset + '/cycout/'

    for roi_name in ['xyori'][:]:
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/' + roi_name + '.tif')
        x0 = trd_and_rescale(x0, 2000)

        x1 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/' + 'xyft0' + '.tif')
        x1 = trd_and_rescale(x1, 2000)
        #testing_over_roi(x0, roi_name, model)

    patchx0 = x0[:, :, 250,-512:,:512].cuda()
    patchx1 = x1[:, :, 250,-512:,:512].cuda()
    out = model(torch.cat([patchx1, patchx0], 1))
    imagesc(patchx0[:, :, 64:-64, 64:-64].squeeze().detach().cpu())
    imagesc(out['out1'][:, :, 64:-64, 64:-64].squeeze().detach().cpu())

    testing_over_roi(x0, x1, roi_name='', model=model)