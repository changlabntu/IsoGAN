import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import time
import shutil
from tqdm import tqdm


def get_one_out(x0, model):
    x0 = x0.unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)

    if gpu:
        out_all = model(x0.cuda(), method='encode')[-1].cpu().detach()
        out_all = model(out_all.cuda(), method='decode')['out0'].cpu().detach()

        #out = model(out_all.cuda(), method='decode')
        #out0 = out['out0'].cpu().detach()
        #out1 = out['out1'].cpu().detach()

        #mask_to_01 = torch.multiply((out0 + 1) / 2, (out1 + 1) / 2)
        #out_all = (mask_to_01 - 0.5) * 2 * 0.5 + out0 * 0.5

    else:
        out_all = model(x0)['out0'].cpu().detach()

    out_all = out_all.numpy()[0, 0, :, :, :]
    return out_all


def test_IsoScope(x0, model, **kwargs):
    out_all = []
    for m in range(mc):

        tini = time.time()
        patch = x0[:, :, kwargs['patch_range']['start_dim0']:kwargs['patch_range']['end_dim0'],
                kwargs['patch_range']['start_dim1']:kwargs['patch_range']['end_dim1'],
                kwargs['patch_range']['start_dim2']:kwargs['patch_range']['end_dim2']]
        #patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()

        # normalize patch  to [-1, 1]
        #patch = (patch - patch.min()) / (patch.max() - patch.min())
        #patch = patch * 2 - 1

        if gpu:
            patch = patch.cuda()
        patch = upsample(patch).squeeze()

        #patch = patch.permute(1, 2, 0)
        out = get_one_out(patch, model)
        #out = np.transpose(out, (2, 0, 1))
        out = np.transpose(out, (2, 0, 1))
        if gpu:
            patch = patch.cpu().detach()
        print('Time:', time.time() - tini)

        out_all.append(out)

    out_all = np.stack(out_all, axis=3)

    return out_all, patch.numpy()


def reverse_log(x):
    return np.power(10, x)


def get_args(option):
    if option == 'Dayu1':
        kwargs = {
            "dataset": 'Dayu1',
            "trd": 424,
            #"prj": '/IsoScopeXY/ngf32ndf32lb10skip4/',
            #"epoch": 1700,
            "prj": '/IsoScopeXYftr0/ngf32ndf32lb10skip4exp0nocyc/',
            "epoch": 1000,
            "uprate": 6,
            #"upsample_params": {'size': (384+32, 128, 384+32)},
            #"patch_range": {'start_dim0': -((384+32) * 300 // 1890), 'end_dim0': None,
            #                'start_dim1': -128, 'end_dim1': None,
            #               'start_dim2': 0, 'end_dim2': (384+32)}
            "upsample_params": {'size': (256, 256, 256)},
            "patch_range": {'start_dim0': 56*3, 'end_dim0': 56*3+40,
                            'start_dim1': 230, 'end_dim1': 230+256,
                            'start_dim2': 384, 'end_dim2': 384+256}
            # "upsample_params": {'size': (256, 256, 256)},
            # "patch_range": {'start_dim0': 80, 'end_dim0': 120,
            #                'start_dim1': 512, 'end_dim1': 512 + 256,
            #                'start_dim2': 512, 'end_dim2': 512 + 256}
        }
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/xyori.tif')
    elif option == 'Fly0B':
        kwargs = {
            "dataset": 'Fly0B',
            "trd": 2000,
            "prj": '/IsoScopeXXcut/ngf32lb10/',
            #"prj": '/IsoScopeXY/ngf32lb10skip4/',
            "epoch": 5000,
            "uprate": 1890 // 300,
            #"upsample_params": {'size': (384+32, 128, 384+32)},
            #"patch_range": {'start_dim0': -((384+32) * 300 // 1890), 'end_dim0': None,
            #                'start_dim1': -128, 'end_dim1': None,
            #               'start_dim2': 0, 'end_dim2': (384+32)}
            "upsample_params": {'size': (384, 128, 384)},
            "patch_range": {'start_dim0': -(384 * 300 // 1890), 'end_dim0': None,
                            'start_dim1': -128, 'end_dim1': None,
                            'start_dim2': 0, 'end_dim2': 384}
        }
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/xyori.tif')
    elif option == 'BraTSReg':
        kwargs = {
        "dataset": 'BraTSReg',
        "trd": 4000,
        "prj": '/IsoScopeXX/cyc0lb1skip4ndf32Try2/',
        "epoch": 320,
        "uprate": 1,
        "upsample_params": {'size': (128, 128, 128)},
        "patch_range": {'start_dim0': -128, 'end_dim0': None, 'start_dim1': 64, 'end_dim1': -64, 'start_dim2': 64, 'end_dim2': -64},
        }
        x_list = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/train/t1norm/*.tif'))
        x0 = tiff.imread(x_list[0])
    elif option == 'womac4':
        kwargs = {
        "dataset": 'womac4',
        "trd": 800,
        #"prj": '/IsoScopeXX/cyc0lb1skip4ndf32/', #baseline
        #"epoch": 320,
        #"prj": '/IsoScopeXX/cyc0lb1skip4ndf32nomc/',    #  no mc seems better
        #"epoch": 500,
        "prj": '/IsoScopeXXoai/bnormnomclr01/',
        "epoch": 240,
        "uprate": 8,
        "upsample_params": {'size': (23*8, 384, 384)},
        "patch_range": {'start_dim0': None, 'end_dim0': None, 'start_dim1': None, 'end_dim1': None, 'start_dim2': None, 'end_dim2': None},
        }
        #womac4_list = ['9081362_06.tif']
        #x_list = sorted(glob.glob('/media/ExtHDD01/oai_diffusion_interpolated/original/a2d/*.tif'));x0 = tiff.imread(x_list[14])
        x_list = sorted(glob.glob('/media/ghc/GHc_data2/OAI_extracted/womac4min0/Processed/norm/*.tif'));x0 = tiff.imread(x_list[140])

    elif option == 'weikun060524':
        kwargs = {
            "dataset": 'weikun060524',
            "trd": 5000,
            #"prj": '/IsoScopeXY16X/ngf32ndf32lb10skip2',
            "prj": '/IsoScopeXY16X/ngf32ndf32lb10skip2nocyc',
            "epoch": 3000,#2300,
            "uprate": 8,
            "upsample_params": {'size': (384, 128, 384)},
            "patch_range": {'start_dim0': -(384 // 16), 'end_dim0': None,
                            'start_dim1': 1024, 'end_dim1': 1024+128,
                            'start_dim2': 1024, 'end_dim2': 1024+384}
        }
        x0 = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/roiAx2.tif')
    return x0, kwargs


def assemble_microscopy_volumne(kwargs, zrange, xrange, yrange, source):
    for ix in tqdm(xrange):
        one_column = []
        for iz in zrange:
            one_row = []
            for iy in yrange:
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
        one_column = np.concatenate(one_column, axis=1).astype(np.float32)
        #one_column[8:-8, :, :].astype(np.float32)
    tiff.imwrite(source[:-1] + '.tif', one_column)


def test_microscopy_volumne(kwargs, zrange, xrange, yrange, destination):
    for ix in xrange:#range(0, x0.shape[2], sz)[:]:
        for iz in zrange:#[1536]:#range(0, x0.shape[3], sx):
            for iy in yrange:#range(0, x0.shape[4], sy)[:]:
                print(iz, ix, iy, iz+dz, ix+dx, iy+dy)
                kwargs['patch_range'] = {'start_dim0': iz, 'end_dim0': iz + dz,
                                         'start_dim1': ix, 'end_dim1': ix + dx,
                                         'start_dim2': iy, 'end_dim2': iy + dy}

                out_all, patch = test_IsoScope(x0, model, **kwargs)

                tiff.imwrite(destination + 'xy/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out_all.mean(axis=3).astype(np.float32))
                tiff.imwrite(destination + 'ori/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', patch)

                if mc > 1:
                    tiff.imwrite(destination + 'xyvar/' + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif', out_all.std(axis=3).astype(np.float32))


def get_model(kwargs, gpu):
    dataset = kwargs['dataset']
    prj = kwargs['prj']
    epoch = kwargs['epoch']
    model = torch.load(
        '/media/ExtHDD01/logs/' + dataset + prj + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
        map_location=torch.device('cpu'))  # .cuda()#.eval()
    upsample = torch.nn.Upsample(size=kwargs['upsample_params']['size'], mode='trilinear')
    if gpu:
        model = model.cuda()
        upsample = upsample.cuda()
    return model, upsample


def norm_x0(x0, kwargs):
    trd = kwargs['trd']
    x0[x0 >= trd] = trd
    x0 = x0 / x0.max()
    x0 = (x0 - 0.5) * 2
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
    return x0


def recreate_volume_folder(destination):
    # remove and recreate the folder
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)
    os.makedirs(destination + 'xy/')
    os.makedirs(destination + 'ori/')
    if mc > 1:
        os.makedirs(destination + 'xyvar/')


def view_two_other_direction(x):
    return np.concatenate([np.transpose(x, (2, 1, 0)), np.transpose(x, (1, 2, 0))], 2)


def slice_for_ganout():
    rois = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"] + '/cycout/xy/*.tif'))

    for roi in tqdm(rois[:]):
        xy = tiff.imread(roi)
        ori = tiff.imread(roi.replace('/xy/', '/ori/'))

        xy = xy[64:-64, 32:-32, 64:-64]
        ori = ori[64:-64, 32:-32, 64:-64]

        if xy.mean() >= -0.5:
            for ix in range(xy.shape[1]):
                tiff.imwrite(roi.replace('/xy/', '/ganxy/')[:-4] + '_' + str(ix).zfill(3) + '.tif', xy[:, ix, :])
                tiff.imwrite(roi.replace('/xy/', '/ganori/')[:-4] + '_' + str(ix).zfill(3) + '.tif', ori[:, ix, :])


def get_weight(size, method='cross'):
    #weight = np.ones(kwargs["upsample_params"]['size'])
    weight = np.ones(size)
    weight[:, :, :H] = np.linspace(0, 1, H)
    weight[:, :, -H:] = np.linspace(1, 0, H)
    if method == 'row':
        return weight
    if method == 'cross':
        weight = np.multiply(np.transpose(weight, (2, 1, 0)), weight)
        return weight


option = 'womac4'
expand = False
gpu = False
norm_exp = False
tilt = False
mc = 1
x0, kwargs = get_args(option=option)
destination = '/media/ExtHDD01/Dataset/paired_images/' + kwargs["dataset"]

if norm_exp:
    x0[x0 <= 100] = 100
    x0[x0 >= 424] = 424
    xx=np.log10(x0+1);xx=np.divide((xx-xx.mean()), xx.std());
    trd = 6
    xx[xx<=-trd]=-trd;xx[xx>=trd]=trd;xx=xx/trd
    x0 = xx
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
elif 0:
    x0 = norm_x0(x0, kwargs)

x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()

#x0 = x0[:, :, :, :, :]
#kwargs['upsample_params']['size'] = ((23)*8, 384, 384)

# repeat
if expand:
    x0 = torch.cat([(torch.flip(x0, (2,))[:, :, -8:, :, :]), x0, torch.flip(x0, (2,))[:, :, :8, :, :]], 2)
    # expand dim 1
    x0 = torch.cat([(torch.flip(x0, (3,))[:, :, :, -64:, :]), x0, torch.flip(x0, (3,))[:, :, :, :64, :]], 3)
    kwargs['upsample_params']['size'] = ((23+16)*8, 512, 384)


if tilt:
    if 1:
        import torchvision.transforms as transforms
        x0 = x0.squeeze().unsqueeze(1)
        to_tilt = transforms.RandomRotation((-45, -44), interpolation=transforms.functional.InterpolationMode.BILINEAR,
                                                expand=False, center=None, fill=-1)
        back_tilt = transforms.RandomRotation((45, 46), interpolation=transforms.functional.InterpolationMode.BILINEAR,
                                                expand=False, center=None, fill=-1)
        x0 = to_tilt(x0)
        x0 = x0.squeeze().unsqueeze(0).unsqueeze(1)
    #x0 = torch.flip(x0, (2,))


# single test
model, upsample = get_model(kwargs, gpu)
out, patch = test_IsoScope(x0, model, **kwargs)
out = out.mean(axis=3)

if expand:
    out = out[8*8:-8*8, 64:-64, :]
    patch = patch[8*8:-8*8, 64:-64, :]

if tilt:
    out = back_tilt(torch.from_numpy(out).unsqueeze(1)).squeeze().numpy()
    patch = back_tilt(torch.from_numpy(patch).unsqueeze(1)).squeeze().numpy()
    #out = out[::-1, :, :]

if option == 'womac4':
    tiff.imwrite(destination + '/patch.tif', view_two_other_direction(patch))
    tiff.imwrite(destination + '/xy.tif', view_two_other_direction(out))
else:
    tiff.imwrite(destination + '/patch.tif', np.transpose(patch, (1, 0, 2)))
    tiff.imwrite(destination + '/xy.tif', np.transpose(out, (1, 0, 2)))

# Fly0B
if 0:
    recreate_volume_folder(destination + '/cycout/')
    H = 64
    dz, dx, dy = (6 * H // kwargs['uprate'], 2 * H, 6 * H)
    sz, sx, sy = (3 * H // kwargs['uprate'], 1 * H, 3 * H)
    zrange = range(0, x0.shape[2], sz)[-7:-2]
    xrange = [448]#range(0, x0.shape[3], sx)[:-2]
    yrange = range(0, x0.shape[4], sy)[:-2]

    test_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange, destination=destination + '/cycout/')
    assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=[448], yrange=yrange, source=destination + '/cycout/xy/')
    assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange, source=destination + '/cycout/ori/')


# weikun060524
if 0:
    recreate_volume_folder(destination + '/cycout/')
    H = 64
    dz, dx, dy = (6 * H // 16, 2 * H, 6 * H)
    sz, sx, sy = (3 * H // 16, 1 * H, 3 * H)

    zrange = range(0, x0.shape[2], sz)[:-2]
    xrange = [1536]
    yrange = range(0, x0.shape[4], sy)[:]

    test_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange, destination=destination + '/cycout/')
    assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=[1536], yrange=yrange, source=destination + '/cycout/xy/')
    assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange, source=destination + '/cycout/ori/')


# Dayu1
# "upsample_params": {'size': (256, 256, 256)},
if 0: # processing volume
    recreate_volume_folder(destination + '/cycout/')
    H = 64
    #dz, dx, dy = (6 * H // kwargs['uprate'], 2 * H, 6 * H)
    #sz, sx, sy = (3 * H // kwargs['uprate'], 1 * H, 3 * H)
    dz, dx, dy = (40, 256, 256)
    sz, sx, sy = (20, 128, 128)
    zrange = range(0, x0.shape[2], sz)[4:12]
    xrange = [448]#range(0, x0.shape[3], sx)[:-2]
    yrange = range(0, x0.shape[4], sy)[:-2]

    test_microscopy_volumne(kwargs, zrange=zrange, xrange=xrange, yrange=yrange, destination=destination + '/cycout/')
    w = get_weight((128, 256, 128), method='cross')
    assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=[448], yrange=yrange, source=destination + '/cycout/xy/')
    assemble_microscopy_volumne(kwargs, zrange=zrange, xrange=[448], yrange=yrange, source=destination + '/cycout/ori/')


