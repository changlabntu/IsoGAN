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




def test_IsoScope(**kwargs):
    if 1:
        dataset = 'Fly0B'
        trd = 5000
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyori.tif')
        x0[x0 >= trd] = trd
        x0 = x0 / x0.max()
        x0 = (x0 - 0.5) * 2

        prj = '/IsoScopeXXcutcyc/ngf32lb10/'
        epoch = 300
        uprate = 1890 // 300

        model = torch.load('/media/ExtHDD01/logs/' + dataset + prj + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                map_location=torch.device('cpu')).cuda()#.eval() # newly ran

        upsample = torch.nn.Upsample(size=(512, 64, 512), mode='trilinear')
        # patch
        patch = x0[-(512*300//1890):, -64:, :512]
    elif 0:
        dataset = 'Fly0B'
        trd = 5000
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyori.tif')
        x0[x0 >= trd] = trd
        x0 = x0 / x0.max()
        x0 = (x0 - 0.5) * 2

        prj = '/IsoScopeXXcut/ngf32lb10transposeTry3/'
        epoch = 1000
        uprate = 1890 // 300

        model = torch.load('/media/ExtHDD01/logs/' + dataset + prj + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                map_location=torch.device('cpu')).cuda()#.eval() # newly ran

        upsample = torch.nn.Upsample(size=(256, 128, 256), mode='trilinear')
        # patch
        patch = x0[-(256*300//1890):, -128:, :256]
    elif 0:
        dataset = 'BraTSReg'
        trd = 5000
        x_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/train/t1norm/*.tif'))
        x0 = tiff.imread(x_list[0])

        prj = '/IsoScopeXX/cyc0lb1skip4ndf32Try2/'
        epoch = 340
        uprate = 1

        model = torch.load('/media/ExtHDD01/logs/' + dataset + prj + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                map_location=torch.device('cpu')).cuda()#.eval() # newly ran

        # patch
        patch = x0[-128:, 56:-56, 56:-56]

        upsample = torch.nn.Upsample(size=(patch.shape[0]*uprate, patch.shape[1], patch.shape[2]), mode='trilinear')

    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
    patch = upsample(patch).squeeze()

    #patch = patch.permute(1, 2, 0)

    out = get_one_out(patch, model)
    #out = np.transpose(out, (2, 0, 1))
    out = np.transpose(out, (2, 0, 1))

    tiff.imwrite('patch.tif', patch.numpy())
    tiff.imwrite('xy.tif', out)

    ox = 128
    oy = 128
    oz = 8

    dx = 256
    dy = 256
    dz = 16

    sx = 128
    sy = 128
    sz = 8

    stepx = dx - ox
    stepy = dy - oy
    stepz = dz - oz

    all_z = []
    all_zg = []
    for z in range(0 + sz, x0.shape[0] - dz + sz, stepz)[3:9]:
        all_x = []
        all_xg = []
        for x in range(0 + sx, x0.shape[1] - dx + sx, stepx)[3:9]:
            all_y = []
            all_yg = []
            for y in range(0 + sy, x0.shape[2] - dy + sy, stepy)[3:9]:
                print(z, x, y)

                patch = x0[z:z + dz, x:x + dx, y:y + dy]
                #all_y.append(patch[oz // 2:-oz // 2, ox // 2:-ox // 2, oy // 2:-oy // 2])

                patch = torch.from_numpy(patch)
                if uprate > 1:
                    patch = upsample(patch.unsqueeze(0).unsqueeze(0)).squeeze()
                all_y.append(patch[oz // 2 * uprate:-oz // 2 * uprate, ox // 2:-ox // 2, oy // 2:-oy // 2])
                #patch = patch / patch.max()
                #patch = (patch - 0.5) * 2
                #patch = x0[:128, 16 + dx:16 + 32 + dx, dx:dx + 512] / 1

                out_all = []
                for mc in range(1):
                    out = get_one_out(patch, model)
                    out = np.transpose(out, (2, 0, 1))
                    out_all.append(out)
                out_all = np.stack(out_all, 0)
                out_all = np.mean(out_all, 0)

                # adjust the mean and std of out_all to match the input
                out_all = out_all - out_all.mean()
                out_all = out_all / out_all.std()
                out_all = out_all * patch.std().numpy()
                out_all = out_all + patch.mean().numpy()

                all_yg.append(out_all[oz//2*uprate:-oz//2*uprate, ox//2:-ox//2, oy//2:-oy//2])

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


