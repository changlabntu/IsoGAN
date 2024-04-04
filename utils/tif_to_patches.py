import os, glob
import tifffile as tiff
import numpy as np
import argparse
import torch
import torch.nn as nn


def tif_to_patches(**kwargs):
    """
    Convert a tif file to a folder of patches
    """
    source = kwargs['source']
    destination = kwargs['destination']
    permute = kwargs['permute']
    (dz, dx, dy) = kwargs['dh']  # (64, 256, 256)
    (sz, sx, sy) = kwargs['step']
    trd = kwargs['trd']

    os.makedirs(root + destination, exist_ok=True)
    npy = tiff.imread(root + source)  # (Z, X, Y)

    if trd is not None:
        npy[npy > trd] = trd

    for z in range(npy.shape[0] // sz):
        for x in range(npy.shape[1] // sx):
            for y in range(npy.shape[2] // sy):
                    volume = npy[z * dz : (z+1) * dz, x * dx : (x+1) * dx, y * dy : (y+1) * dy]
                    print(volume.shape)
                    if volume.shape == (dz, dx, dy):
                        if permute is not None:
                            volume = np.transpose(volume, permute)
                        for s in range(volume.shape[0]):
                            patch = volume[s, ::]
                            tiff.imsave(root + destination + str(x).zfill(3) + str(y).zfill(3) + str(z).zfill(3) +
                                        '_' + str(s).zfill(4) + '.tif', patch)


def main(source, destination, dh, step, permute, trds):
    for i, (s, d) in enumerate(zip(source, destination)):
        input = {'source': s + '.tif', 'destination': 'full/' + d + '/',
                 'permute': permute, 'dh': dh, 'step': step, 'trd': trds[i]}
        tif_to_patches(**input)


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


def get_average(source, destination):
    """
    Get the average of a folder of tif files
    """
    #l = sorted(glob.glob(source + '*'))
    #all = [tiff.imread(x) for x in l]
    #all = np.stack(all, 3)

#resampling(source=root + 'xyori.tif',
#           destination=root + 'xyzorix6.tif',
#           size=[50, -1, -1])

root = '/workspace/Data/paired_images/longone/'
#root = '/media/ExtHDD01/Dataset/paired_images/longone/'

if 1:
    resampling(source=root + 'xyori.tif',
               destination=root + 'xyzori.tif',
               size=[301 * 6, -1, -1])
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
    slices = sorted(glob.glob('/media/ExtHDD01/BRC/3DIntestine/SCFA_SNCA/Dendrite/*'))

    cropped = []
    for z in range(300, 800):
        print(z)
        x = tiff.imread(slices[z])
        x = x[5500-1024:5500+1024, 3700-1024:3700+1024]
        cropped.append(x)