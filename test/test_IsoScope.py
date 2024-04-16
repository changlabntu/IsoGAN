import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


def get_one_out(x0, model):
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

    out_all = model(x0.cuda(), method='encode')[-1].cpu().detach()
    out_all = model(out_all.cuda(), method='decode')['out0'].cpu().detach()

    out_all = out_all.numpy()[0, 0, :, :, :]
    return out_all


def test_IsoScope():

    dataset = 'x2404g102'

    model = torch.load('/media/ExtHDD01/logs/' + dataset + '/IsoScope0/0/checkpoints/net_g_model_epoch_20.pth',
                       map_location=torch.device('cpu')).cuda()#.eval() # newly ran
    x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset +'/xyori.tif')

    #x0[x0 >= 50] = 50
    x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2

    dz = 32
    dx = 256
    dy = 256

    stepz = 32 - 4
    stepx = 256 - 32
    stepy = 256 - 32

    all_z = []
    all_zg = []
    for z in range(0, x0.shape[0] - dz, stepz)[:2]:
        all_x = []
        all_xg = []
        for x in range(0, x0.shape[1] - dx, stepx)[:2]:
            all_y = []
            all_yg = []
            for y in range(0, x0.shape[2] - dy, stepy)[:2]:
                print(z, x, y)

                patch = x0[z:z + dz, x:x + dx, y:y + dy]
                #patch = patch / patch.max()
                #patch = (patch - 0.5) * 2
                #patch = x0[:128, 16 + dx:16 + 32 + dx, dx:dx + 512] / 1
                out_all = get_one_out(patch, model)
                out_all = np.transpose(out_all, (2, 0, 1))
                tiff.imwrite('xy.tif', out_all[:, :, :])

                all_y.append(patch[2:-2, 16:-16, 16:-16])
                all_yg.append(out_all[16:-16, 16:-16, 16:-16])

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
                #out_all = get_one_out(patch, model)
                #tiff.imwrite('patch_' + str(z) + '_' + str(x) + '_' + str(y) + '.tif', out_all)


    #print(out_all.shape)

    #upsample = torch.nn.Upsample(scale_factor=(1, 1, 8))
    #tiff.imwrite('x0.tif', upsample(x0)[0, 0, :, :, :].numpy())
    #tiff.imwrite('xy.tif', out_all[:, :, :])

    tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xx.tif', all_z)
    tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xy.tif', all_zg)

if 0:
    for epoch in range(20, 201, 20):
        model = torch.load('/media/ExtHDD01/logs/x2404g102/X3cyc_ngf64/checkpoints/net_gXY_model_epoch_' + str(epoch) + '.pth',
                             map_location=torch.device('cpu')).cuda()

        dataset = 'x2404g102'
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyori.tif')

        x0 = x0 / x0.max()
        x0 = (x0 - 0.5) * 2

        x0 = x0[:128, :, :2048]
        upsample = torch.nn.Upsample(scale_factor=(3, 1, 1))
        x0u = upsample(torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float())

        #for xi in range(x0u.shape[1]):

        if 1:
            slice = x0u[:, :, :, 2349, :]
            out = model(slice.cuda())['out0'].cpu().detach().numpy()[0, 0, :, :]
            tiff.imwrite('outimg/original.tif', slice.squeeze()[32:-32, 32:-32].numpy())
            tiff.imwrite('outimg/X364_' + str(epoch).zfill(3) + '.tif', out[32:-32, 32:-32])
        #imagesc(slice.squeeze()[32:-32, 32:-32])
        #imagesc(out[32:-32, 32:-32])

        for i in range(x0u.shape[3]):
            slice = x0u[:, :, :, i, :]
            out = model(slice.cuda())['out0'].cpu().detach().numpy()[0, 0, :, :]
            tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/x2404g102/xy/' + str(i).zfill(4) + '.tif', out[32:-32, 32:-32])
            tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/x2404g102/xx/' + str(i).zfill(4) + '.tif', slice.squeeze()[32:-32, 32:-32].numpy())