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

    #out_all = model(x0)['out0'].cpu().detach()

    out_all = out_all.numpy()[0, 0, :, :, :]
    return out_all



def test_IsoScope(x0, **kwargs):
    dataset = kwargs['dataset']
    trd = kwargs['trd']
    prj = kwargs['prj']
    epoch = kwargs['epoch']
    #uprate = kwargs['uprate']

    x0[x0 >= trd] = trd
    x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2

    model = torch.load('/media/ExtHDD01/logs/' + dataset + prj + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
            map_location=torch.device('cpu')).cuda()#.eval() # newly ran
    upsample = torch.nn.Upsample(size=kwargs['upsample_params']['size'], mode='trilinear')
    patch = x0[kwargs['patch_range']['start_dim0']:kwargs['patch_range']['end_dim0'],
            kwargs['patch_range']['start_dim1']:kwargs['patch_range']['end_dim1'],
            kwargs['patch_range']['start_dim2']:kwargs['patch_range']['end_dim2']]


    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
    patch = upsample(patch).squeeze()

    #patch = patch.permute(1, 2, 0)

    out = get_one_out(patch, model)
    #out = np.transpose(out, (2, 0, 1))
    out = np.transpose(out, (2, 0, 1))

    tiff.imwrite('patch.tif', patch.numpy())
    tiff.imwrite('xy.tif', out)


def reverse_log(x):
    return np.power(10, x)


if 1:
    kwargs = {
        "dataset": 'Fly0B',
        "trd": 5000,
        "prj": '/IsoScopeXXcut/ngf32lb10/',
        #"prj": '/IsoScopeXY/ngf32lb10skip4/',
        "epoch": 2800,
        "uprate": 1890 // 300,
        "upsample_params": {'size': (512, 64, 512)},
        "patch_range": {'start_dim0': -(512 * 300 // 1890), 'end_dim0': None,
                        'start_dim1': -64, 'end_dim1': None,
                        'start_dim2': 0, 'end_dim2': 512}
    }
    x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/xyori.tif')
elif 0:
    kwargs = {
    "dataset": 'BraTSReg',
    "trd": 5000,
    "prj": '/IsoScopeXX/cyc0lb1skip4ndf32Try2/',
    "epoch": 340,
    "uprate": 1,
    "upsample_params": {'size': (128, 128, 128)},
    "patch_range": {'start_dim0': -128, 'end_dim0': None, 'start_dim1': 64, 'end_dim1': -64, 'start_dim2': 64, 'end_dim2': -64},
    }
    x_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/train/t1norm/*.tif'))
    x0 = tiff.imread(x_list[0])
elif 0:
    kwargs = {
    "dataset": 'womac4',
    "trd": 2000,
    #"prj": '/IsoScopeXX/cyc0lb1skip4ndf32/',
    "prj": '/IsoScopeXY/ngf32lb10skip4nocut/',
    "epoch": 120,
    "uprate": 8,
    "upsample_params": {'size': (23*8, 384, 384)},
    "patch_range": {'start_dim0': None, 'end_dim0': None, 'start_dim1': None, 'end_dim1': None, 'start_dim2': None, 'end_dim2': None},
    }
    x_list = sorted(glob.glob('/media/ExtHDD01/oai_diffusion_interpolated/original/a2d/*.tif'))
    x0 = tiff.imread(x_list[0])

test_IsoScope(x0, **kwargs)
