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


dataset = 'Dayu1'

# no cyc
if 0:
    model = torch.load('/media/ExtHDD01/logs/' + dataset + '/IsoScopeXXcyc0b/ngf32lb10cut0/checkpoints/net_g_model_epoch_1900.pth',
                      map_location=torch.device('cpu')).cuda()#.eval()
elif 0:
    # with cyc
    model = torch.load(
    '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXXcyc0/ngf32tryB/checkpoints/net_g_model_epoch_900.pth',
    map_location=torch.device('cpu')).cuda()  # .eval() # newly ran
elif 0:  # THIS WORKS 1700
    model = torch.load(
        '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXY/ngf32ndf32lb10skip4/checkpoints/net_g_model_epoch_1700.pth',
        map_location=torch.device('cpu')).cuda()  # .eval()
elif 0:  # BUT NOT RUN0
    model = torch.load(
        '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXYnocyc/ngf32ndf32lb10skip4/checkpoints/net_g_model_epoch_500.pth',
        map_location=torch.device('cpu')).cuda()  # .eval()
else:
    model = torch.load(
        '/media/ExtHDD01/logs/' + dataset + '/IsoScopeXYftr0/ngf32ndf32lb10skip4exp0nocyc/checkpoints/net_g_model_epoch_500.pth',
        map_location=torch.device('cpu')).cuda()  # .eval()

x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyori.tif')

#uprate = 8
#if uprate > 1:
upsample = torch.nn.Upsample(size=(256, 256, 256), mode='trilinear')


trd = (100, 424)
x0[x0 <= trd[0]] = trd[0]
x0[x0 >= trd[1]] = trd[1]
if 0:
    x0 = (x0 - x0.min()) / (x0.max() - x0.min())
    #x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2
else:
    xx=np.log10(x0+1);xx=np.divide((xx-xx.mean()), xx.std());
    trd = 6
    xx[xx<=-trd]=-trd;xx[xx>=trd]=trd;xx=xx/trd
    x0 = xx

ox = 128
oy = 128
oz = 20

dx = 256
dy = 256
dz = 40

sx = 128
sy = 128
sz = 20

stepx = dx - ox
stepy = dy - oy
stepz = dz - oz

all_z = []
all_zg = []
for z in range(0 + sz, x0.shape[0] - dz + sz, stepz)[3:6]:
    all_x = []
    all_xg = []
    for x in range(0 + sx, x0.shape[1] - dx + sx, stepx)[5:6]:
        all_y = []
        all_yg = []
        for y in range(0 + sy, x0.shape[2] - dy + sy, stepy)[:6]:
            print(z, x, y)

            patch = x0[z:z + dz, x:x + dx, y:y + dy]
            #all_y.append(patch[oz // 2:-oz // 2, ox // 2:-ox // 2, oy // 2:-oy // 2])

            patch = torch.from_numpy(patch)
            patch = upsample(patch.unsqueeze(0).unsqueeze(0)).squeeze()
            print(patch.shape)

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

            all_y.append(patch[64:-64, ox // 2:-ox // 2, oy // 2:-oy // 2])

            # adjust the mean and std of out_all to match the input
            #aout_all = out_all - out_all.mean()
            #out_all = out_all / out_all.std()
            #out_all = out_all * patch.std().numpy()
            #out_all = out_all + patch.mean().numpy()

            all_yg.append(out_all[64:-64, ox//2:-ox//2, oy//2:-oy//2])

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

all_z = np.transpose(all_z, (1, 0, 2))
all_zg = np.transpose(all_zg, (1, 0, 2))

tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xx.tif',  (all_z.astype(np.float32)))
tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xy.tif',  (all_zg.astype(np.float32)))


if 0:
    patch = tiff.imread('patch.tif') # (X, Z, Y)
    patch = np.transpose(patch, (1, 0, 2))
    patch = torch.from_numpy(patch).float().cuda()
    out = get_one_out(patch, model)
    out = np.transpose(out, (2, 0, 1))
    out = np.transpose(out, (1, 0, 2))
    tiff.imwrite('out.tif', out)