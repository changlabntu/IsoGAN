import os, glob
import tifffile as tiff
import numpy as np
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def read_2d_tif_to_3d(xlist):
    """
    Read a list of 2D tif file and convert it to a 3D numpy array
    """
    #xlist = sorted(glob.glob(source + '/*'))
    x = [tiff.imread(x) for x in xlist]
    x = np.stack(x, 0)
    return x


def tif_to_patches(npys, **kwargs):
    """
    Convert a tif file to a folder of patches
    """
    (dz, dx, dy) = kwargs['dh']  # (64, 256, 256)
    (sz, sx, sy) = kwargs['step']

    for i in range(len(npys)):
        os.makedirs(root + kwargs['destination'][i], exist_ok=True)

    for i in range(len(npys)):
        if kwargs['trd'][i] is not None:
            npys[i][npys[i] < kwargs['trd'][i][0]] = kwargs['trd'][i][0]
            npys[i][npys[i] > kwargs['trd'][i][1]] = kwargs['trd'][i][1]

        if kwargs['norm'][i] is not None:
            if kwargs['norm'][i] == 'zrescale':
                npys[i] = z_rescale(npys[i], trd=kwargs['zrescale'])
            elif kwargs['norm'][i] == '01':
                npys[i] = (npys[i] - npys[i].min()) / (npys[i].max() - npys[i].min())
            elif kwargs['norm'][i] == '11':
                npys[i] = (npys[i] - npys[i].min()) / (npys[i].max() - npys[i].min())
                npys[i] = (npys[i] - 0.5) * 2


    for z in range(npys[0].shape[0] // sz):
        for x in range(npys[0].shape[1] // sx):
            for y in range(npys[0].shape[2] // sy):
                    print(z, x, y)
                    volumes = []
                    for i in range(0, len(npys)):
                        volumes.append(npys[i][z * dz : (z+1) * dz, x * dx : (x+1) * dx, y * dy : (y+1) * dy])

                    if volumes[0].shape == (dz, dx, dy):
                        if volumes[0].mean() > kwargs['ftr']:
                            if kwargs['permute'] is not None:
                                for i in range(1, len(npys)):
                                    volumes[i] = np.transpose(volumes[i], kwargs['permute'])
                            for i in range(0, len(npys)):
                                for s in range(volumes[0].shape[0]):
                                    patch = volumes[i][s, ::]
                                    #if kwargs['norm'] is not None:
                                    patch = patch.astype(np.float32)
                                    tiff.imwrite(root + kwargs['destination'][i] + kwargs['prefix'] + str(x).zfill(3) + str(y).zfill(3) + str(z).zfill(3) +
                                                '_' + str(s).zfill(4) + '.tif', patch)
                                    #tiff.imwrite(root + kwargs['destination'][i] + str(volumes[0].mean())[:7] + '_' + str(s).zfill(4) + '.tif', patch)


#def main(source, destination, dh, step, permute, trds, norm, prefix, ftr):
#    for i, (s, d) in enumerate(zip(source, destination)):
#        input = {'source': s + '.tif', 'destination': 'train/' + d + '/',
#                 'permute': permute, 'dh': dh, 'step': step, 'trd': trds[i], 'norm': norm, 'prefix': prefix, 'ftr': ftr}
#        tif_to_patches(**input)


def resampling(source, destination, scale=None, size=None):
    """
    Resampling a tif file
    """
    x = tiff.imread(source)

    if scale is not None:
        up = nn.Upsample(scale_factor=scale, mode='trilinear')
    if size is not None:
        for i in range(len(size)):
            if size[i] == -1:
                size[i] = x.shape[i]
        up = nn.Upsample(size=size, mode='trilinear')

    dtype = x.dtype
    x = x.astype(np.float32)
    out = up(torch.from_numpy(x).unsqueeze(0).unsqueeze(0))
    tiff.imwrite(destination, out[0, 0, :, :, :].numpy().astype(dtype))


def z_rescale(xx, trd=6):
    xx=np.log10(xx+1);xx=np.divide((xx-xx.mean()), xx.std());
    xx[xx<=-trd]=-trd;xx[xx>=trd]=trd;xx=xx/trd;
    return xx


def get_average(source, destination):
    """
    Get the average of a folder of tif files
    """
    #l = sorted(glob.glob(source + '*'))
    #all = [tiff.imread(x) for x in l]
    #all = np.stack(all, 3)

if 1:
    #root = '/workspace/Data/DPM4X/'
    #suffix = ''
    #for s in ['3-2ROI000', '3-2ROI002', '3-2ROI006', '3-2ROI008']:
    #    main(source=[s + suffix],
    #         destination=['xyori512' + suffix],
    #         dh=(32, 512, 512), step=(32, 512, 512), permute=None, trds=[424], norm='11', prefix=s.split('.')[0] + '-')

    #root = '/workspace/Data/DPM4X/'
    #root = '/media/ExtHDD01/Dataset/paired_images/DPM4X/'
    root = '/home/ubuntu/Data/Dataset/paired_images/DPM4X/'
    suffix = ''
    for s in ['ori/3-2ROI00' + str(x) for x in range(10)]:##, '3-2ROI002', '3-2ROI006', '3-2ROI008']:
        npy0 = tiff.imread(root + s + '.tif')
        npy1 = tiff.imread((root + s + '.tif').replace('/ori/', '/ft0/'))
        tif_to_patches([npy0, npy1],
                       destination=['oripatch/', 'ft0patch/'],
                       dh=(32, 256, 256), step=(32, 256, 256), permute=None,
                       trd=((100, 424), (0, 4)), norm=('zrescale', '11'),
                       prefix=s.split('.')[0].split('/')[-1] + '-', ftr=0, zrescale=6)

if 0:
    root = '/workspace/Data/x2404g102/'
    resampling(source=root + 'xyori.tif',
               destination=root + 'xyzori.tif',
               size=[201 * 3, -1, -1])
    suffix = ''
    main(source=['xyzori' + suffix],
         destination=['zyori' + suffix],
         dh=(256, 64, 256), step=(256, 64, 256), permute=(1, 0, 2), trds=[None])
    main(source=['xyzori' + suffix],
         destination=['xyori' + suffix],
         dh=(64, 256, 256), step=(64, 256, 256), permute=None,  trds=[None])


if 0:
    resampling(source=root + 'xyft0.tif',
               destination=root + 'xyzft0.tif',
               size=[201 * 8, -1, -1])
    suffix = ''
    main(source=['xyzft0' + suffix, 'xyzori' + suffix],
         destination=['xyft0' + suffix, 'xyori' + suffix],
         dh=(64, 256, 256), step=(64, 256, 256), permute=None,  trds=[5, 20000])


if 0:
    root = '/workspace/Data/Dayu1/'
    suffix = ''
    main(source=['xyori' + suffix],
         destination=['xyori' + suffix],
         dh=(32, 256, 256), step=(32, 256, 256), permute=None, trds=[424], norm='11')


if 0:
    #root = '/workspace/Data/Fly0B/'
    root = '/media/ExtHDD01/Dataset/paired_images/Fly0B/'
    suffix = ''
    #main(source=['xyori' + suffix],
    #     destination=['xyoriftr' + suffix],
    #     dh=(32, 512, 512), step=(32, 512, 512), permute=None, trds=[2000], norm='11', prefix='', ftr=-1)

    #resampling(source=root + 'xyori.tif',
    #           destination=root + 'xyzori8.tif',
    #           size=[301 * 8, -1, -1])
    npy = tiff.imread(root + 'xyzft0.tif')
    npy = npy[-512:, -512:, :1024]
    max_val = 8 # 2000
    npy[npy > max_val] = max_val
    npy = (npy - 0) / (max_val - 0)
    npy = (npy - 0.5) * 2
    npy = np.transpose(npy, (2, 0, 1))
    npy = npy.astype(np.float32)
    for i in range(npy.shape[0]):
        tiff.imwrite('/media/ExtHDD01/Dataset/paired_images/Fly0B/cycout/yzft0/' + str(i).zfill(3) + '.tif', npy[i, ::])
    #tif_to_patches(npy,
    #     destination='zyori8/',
    #     dh=(32, 512, 512), step=(32, 512, 512), permute=(1, 0, 2), trd=2000, norm='11', prefix='', ftr=-1)

if 0:
    #root = '/workspace/Data/Fly0B/'
    root = '/media/ExtHDD01/Dataset/paired_images/Fly0B/'
    suffix = ''
    npy = tiff.imread(root + 'xyori' + '.tif')
    #main(source=['xyori' + suffix],
    #     destination=['xyoriftr' + suffix],
    #     dh=(32, 512, 512), step=(32, 512, 512), permute=None, trds=[2000], norm='11', prefix='', ftr=-1)

    tif_to_patches([npy],
                   destination=('temp/', ),
                   dh=(32, 512, 512), step=(32, 512, 512), permute=None,
                   trd=((0, 2000), ), norm=('11', ), prefix='', ftr=-100)

if 0:
    root = '/workspace/Data/Fly0B/'
    #root = '/media/ExtHDD01/Dataset/paired_images/Fly0B/'
    suffix = ''
    main(source=['xyori' + suffix],
         destination=['xyoriftr' + suffix],
         dh=(32, 512, 512), step=(32, 512, 512), permute=None, trds=[2000], norm='11', prefix='', ftr=-1)

if 0:
    root = '/media/ExtHDD01/BRC/JY_20240605/'
    xlist = sorted(glob.glob(root + 'ExMa2_ROI1_10um/*.tif'))

    for i in range(0, len(xlist), 32):
        print(i)
        npy = read_2d_tif_to_3d(xlist[i:i+32])
        npy[npy > 5400] = 5400
        npy = (npy - 0) / (5400 - 0)
        npy = (npy - 0.5) * 2
        tif_to_patches(npy,
                       destination='temp/',
                       dh=(32, 512, 512), step=(32, 512, 512), permute=None,
                       trd=[5400], norm=None, prefix=str(i).zfill(3), ftr=-1, read_2d=True)

if 0:
    root = '/media/ExtHDD01/Dataset/paired_images/weikun060524/'
    suffix = ''
    for s in ['roiAx2', 'roiBx2']:
        npy = tiff.imread(root + s + '.tif')
        tif_to_patches(npy,
                       destination='temp/',
                       dh=(32, 512, 512), step=(32, 512, 512), permute=None,
                       trd=(0, 5100), norm='11', prefix=s, ftr=-1000, read_2d=False)

if 0:
    root = '/media/ExtHDD01/Dataset/paired_images/DPM4X/'
    x = tiff.imread(root + 'ori/3-2ROI008.tif')
    x[x<=100] = 100
    x[x>=424] = 424
    xx=np.log10(x+1);xx=np.divide((xx-xx.mean()), xx.std());
    trd = 6
    xx[xx<=-trd]=-trd;xx[xx>=trd]=trd;xx=xx/trd;
    plt.hist(xx.flatten(), bins=50);plt.show()
    tiff.imwrite(root + 'temp.tif', xx)

if 0:
    root = '/media/ExtHDD01/Dataset/paired_images/womac4/full/ap/'
    x_list = sorted(glob.glob(root + '*.tif'))
    for x in x_list:
        y = tiff.imread(x)
        y = np.log10(y + 1); y = np.divide((y - y.mean()), y.std())
        trd = 3
        y[y <= -trd] = -trd;
        y[y >= trd] = trd;
        y = y / trd;
        #plt.hist(y.flatten(), bins=50)
        #plt.show()
        print(x.replace('/ap/', '/apexp/'))
        tiff.imwrite(x.replace('/ap/', '/apexp/'), y)
