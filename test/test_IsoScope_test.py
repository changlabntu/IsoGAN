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

    #out_all = model(x0.cuda(), method='encode')[-1].cpu().detach()
    #out_all = model(out_all.cuda(), method='decode')['out0'].cpu().detach()

    out_all = model(x0)['out0'].cpu().detach()

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
            map_location=torch.device('cpu'))#.cuda()#.eval() # newly ran
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

    return out, patch.numpy()


def reverse_log(x):
    return np.power(10, x)

def get_args(option):
    if option == 'Fly0B':
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
    elif option == 'BraTSReg':
        kwargs = {
        "dataset": 'BraTSReg',
        "trd": 4000,
        "prj": '/IsoScopeXX/cyc0lb1skip4ndf32Try2/',
        "epoch": 340,
        "uprate": 1,
        "upsample_params": {'size': (128, 128, 128)},
        "patch_range": {'start_dim0': -128, 'end_dim0': None, 'start_dim1': 64, 'end_dim1': -64, 'start_dim2': 64, 'end_dim2': -64},
        }
        x_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/train/t1norm/*.tif'))
        x0 = tiff.imread(x_list[0])
    elif option == 'womac4':
        kwargs = {
        "dataset": 'womac4',
        "trd": 2000,
        #"prj": '/IsoScopeXX/cyc0lb1skip4ndf32/',
        "prj": '/IsoScopeXY/ngf32lb10skip4nocut/',
        "epoch": 400,
        "uprate": 8,
        "upsample_params": {'size': (23*8, 384, 384)},
        "patch_range": {'start_dim0': None, 'end_dim0': None, 'start_dim1': None, 'end_dim1': None, 'start_dim2': None, 'end_dim2': None},
        }
        x_list = sorted(glob.glob('/media/ExtHDD01/oai_diffusion_interpolated/original/a2d/*.tif'))
        x0 = tiff.imread(x_list[0])
    elif option == 'weikun060524':
        kwargs = {
            "dataset": 'weikun060524',
            "trd": 5000,
            "prj": '/IsoScopeXY16X/ngf32lb10skip2nocyc/',
            #"prj": '/IsoScopeXY/ngf32lb10skip4/',
            "epoch": 3500,#2300,
            "uprate": 8,
            "upsample_params": {'size': (384, 128, 384)},
            "patch_range": {'start_dim0': -(384 // 16), 'end_dim0': None,
                            'start_dim1': 1024, 'end_dim1': 1024+128,
                            'start_dim2': 1024, 'end_dim2': 1024+384}
        }
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/roiAx2.tif')
    return x0, kwargs


def test_microscopy_volumne(kwargs):
    destination = '/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/cycout/'

    H = 64

    dz, dx, dy = (6 * H // 16, 2 * H, 6 * H)
    sz, sx, sy = (3 * H // 16, 1 * H, 3 * H)

    for iz in range(0, x0.shape[0], sz)[:]:
        for ix in [1536]:#range(0, x0.shape[1], sx):
            for iy in range(0, x0.shape[2], sy)[:]:
                print(iz, ix, iy)
                print(iz+dz, ix+dx, iy+dy)
                kwargs['patch_range'] = {'start_dim0': iz, 'end_dim0': iz + dz,
                                         'start_dim1': ix, 'end_dim1': ix + dx,
                                         'start_dim2': iy, 'end_dim2': iy + dy}
                out, patch = test_IsoScope(x0, **kwargs)
                tiff.imwrite(destination + 'xy/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out)
                tiff.imwrite(destination + 'ori/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', patch)


    # combine
    def get_weight(method):
        weight = np.ones((256, 128, 256))
        weight[:, :, :64] = np.linspace(0, 1, 64)
        weight[:, :, -64:] = np.linspace(1, 0, 64)
        if method == 'row':
            return weight
        if method == 'cross':
            weight = np.multiply(np.transpose(weight, (2, 1, 0)), weight)
            return weight


    w = get_weight('cross')


    source = '/media/ExtHDD01/Dataset/paired_images/weikun060524/cycout/xy/'
    for ix in [1536]:#range(0, x0.shape[1], sx):
        one_column = []
        for iz in range(0, x0.shape[0], sz)[:-2]:
            one_row = []
            for iy in range(0, x0.shape[2], sy):
                #print(iz, ix, iy)
                x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
                cropped = x[64:-64, :, 64:-64]
                cropped = np.multiply(cropped, w)
                if len(one_row) > 0:
                    one_row[-1][:, :, -64:] = one_row[-1][:, :, -64:] + cropped[:, :, :64]
                    one_row.append(cropped[:, :, 64:])
                else:
                    one_row.append(cropped)
            one_row = np.concatenate(one_row, axis=2)
            one_row = np.transpose(one_row, (1, 0, 2))
            print(one_row.shape)

            if len(one_column) > 0:
                one_column[-1][:, -64:, :] = one_column[-1][:, -64:, :] + one_row[:, :64, :]
                one_column.append(one_row[:, 64:, :])
            else:
                one_column.append(one_row)
        one_column = np.concatenate(one_column, axis=1)
    #tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/weikun060524/cycout/temp.tif', np.transpose(one_row, (1, 0, 2)).astype(np.float32))

    tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/weikun060524/cycout/xy.tif', one_column[8:-8, :, :].astype(np.float32))



x0, kwargs = get_args(option='womac4')
out, patch = test_IsoScope(x0, **kwargs)
tiff.imwrite('patch.tif', patch)
tiff.imwrite('xy.tif', out)