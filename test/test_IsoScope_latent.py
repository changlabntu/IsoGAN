import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


def get_one_encode(x0, model):
    # x0 (B, C, D, H, W)
    x0 = x0.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

    out_all = model(x0.cuda(), method='encode')[-1].cpu().detach()
    #print(out_all.shape)
    #out_all = model(out_all.cuda(), method='decode')['out0'].cpu().detach()

    #out_all = out_all.numpy()[0, 0, :, :, :]
    return out_all

def get_one_decode(out_all, model):
    out_all = model(out_all.cuda(), method='decode')['out0'].cpu().detach()
    out_all = out_all.numpy()[0, 0, :, :, :]
    return out_all

def test_IsoScope():

    dataset = 'Dayu1'

    # no cyc
    if 1:
        model = torch.load(
            '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXXcyc0b/ngf32lb10/checkpoints/net_g_model_epoch_1700.pth',
            map_location=torch.device('cpu')).cuda()  # .eval() # newly ran
        # with cyc
    else:
        model = torch.load(
            '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXXcyc0/ngf32tryB/checkpoints/net_g_model_epoch_900.pth',
            map_location=torch.device('cpu')).cuda()  # .eval() # newly ran

    x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyori.tif')

    uprate = 8
    if uprate > 1:
        upsample = torch.nn.Upsample(scale_factor=(uprate, 1, 1), mode='trilinear')

    trd = 424
    x0[x0 >= trd] = trd
    x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2

    dx = 256
    dy = 256
    dz = 32

    crop = True
    if crop:
        ox = 128
        oy = 128
        oz = 16
    else:
        ox = 0
        oy = 0
        oz = 16

    sx = 0
    sy = 0
    sz = 0

    stepx = dx - ox
    stepy = dy - oy
    stepz = dz - oz

    all_z = []
    all_zg = []
    for z in range(0 + sz, x0.shape[0] + sz, stepz)[8:9]:
        all_x = []
        all_xg = []
        for x in range(0 + sx, x0.shape[1] + sx, stepx)[3:4]:
            all_y = []
            all_yg = []

            # initialize a row
            patchz_next = None
            out_next = None

            for y in range(0 + sy, x0.shape[2] + sy, stepy)[0:5]:

                patch = x0[z:z + dz, x:x + dx, y:y + dy]
                patch_next = x0[z:z + dz, x:x + dx, y + stepy:y + dy + stepy]

                patch = torch.from_numpy(patch)
                patch_next = torch.from_numpy(patch_next)

                patch = upsample(patch.unsqueeze(0).unsqueeze(0)).squeeze()
                patch_next = upsample(patch_next.unsqueeze(0).unsqueeze(0)).squeeze()

                all_y.append(patch[oz // 2 * uprate:-oz // 2 * uprate, ox // 2:-ox // 2, oy // 2:-oy // 2])

                if patchz_next is not None:
                    patchz = patchz_next
                else:
                    patchz = get_one_encode(patch, model) # (B, C, H, W, D)
                patchz_next = get_one_encode(patch_next, model)

                overlapz = (patchz_next[:, :, :, :16, :] + patchz[:, :, :, -16:, :]) / 2
                patchz[:, :, :, -16:, :] = overlapz
                patchz_next[:, :, :, :16, :] = overlapz

                if out_next is not None:
                    out = out_next
                else:
                    out = get_one_decode(patchz, model)
                    out = np.transpose(out, (2, 0, 1))

                out_next = get_one_decode(patchz_next, model)
                out_next = np.transpose(out_next, (2, 0, 1))

                overlap = (out_next[:, :, :128] + out[:, :, -128:]) / 2
                out[:, :, -128:] = overlap
                out_next[:, :, :128] = overlap

                out_all = out
                # adjust the mean and std of out_all to match the input
                #out_all = out_all - out_all.mean()
                #out_all = out_all / out_all.std()
                #out_all = out_all * patch.std().numpy()
                #out_all = out_all + patch.mean().numpy()
                if crop:
                    all_yg.append(out_all[oz//2*uprate:-oz//2*uprate, ox//2:-ox//2, oy//2:-oy//2])
                else:
                    all_yg.append(out_all[oz // 2 * uprate:-oz // 2 * uprate, :, :])
                    #all_yg.append(out_all)

            all_y = np.concatenate(all_y, 2)
            all_x.append(all_y)
            all_yg = np.concatenate(all_yg, 2)
            all_xg.append(all_yg)
        all_x = np.concatenate(all_x, 1)
        all_z.append(all_x)
        all_xg = np.concatenate(all_xg, 1)
        all_zg.append(all_xg)
    all_z = np.concatenate(all_z, 0)
    all_zg = np.concatenate(all_zg, 0)

    tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xx.tif',  (all_z))
    tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xy.tif',  (all_zg))


def reverse_log(x):
    return np.power(10, x)


