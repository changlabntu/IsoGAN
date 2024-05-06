from __future__ import print_function
import argparse, json
import os, glob, sys
from utils.data_utils import imagesc
import torch
import torchvision
from dotenv import load_dotenv
import torchvision.transforms as transforms
import tifffile as tiff
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from dataloader.data_multi import RawTif
import torch.nn as nn


def resampling(x, scale=None, size=None):
    """
    Resampling a tif file
    """
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
    return out[0, 0, :, :, :].numpy().astype(dtype)

def forward_test():
    if 0:
        source = '/media/ExtHDD01/Dataset/paired_images/longone/'
        net = torch.load('/media/ExtHDD01/logs/longone/WorkingGAN/cyc_ngf32/checkpoints/net_gXY_model_epoch_200.pth', map_location='cpu').cuda()
        o = tiff.imread('/media/ExtHDD01/Dataset/paired_images/longone/xyzori.tif')
        w = tiff.imread('/media/ExtHDD01/Dataset/paired_images/longone/xyzori.tif')
    else:
        source = '/media/ExtHDD01/Dataset/paired_images/Human/'
        net = torch.load('/media/ExtHDD01/logs/Human/cyc4_1024/cyc4/checkpoints/netGXY_model_epoch_80.pth',
                         map_location='cpu').cuda()
        o = tiff.imread('/media/ExtHDD01/Dataset/paired_images/Human/xyzori.tif')
        w = tiff.imread('/media/ExtHDD01/Dataset/paired_images/Human/xyzori.tif')

    o = o[500:800, 512:-512, 512:-512]
    w = w[500:800, 512:-512, 512:-512]

    #o[o >= 20000] = 20000
    #w[w >= 5] = 5#w[w >= 0.24] = 0.24

    #o = resampling(o, scale=None, size=(1200, 1024, 1024))
    #w = resampling(w, scale=None, size=(1200, 1024, 1024))

    o = o[-o.shape[0] // 256 * 256:, :, :]
    w = w[-w.shape[0] // 256 * 256:, :, :]

    o = (o - o.min()) / (o.max() - o.min())
    o = (o - 0.5) / 0.5
    w = (w - w.min()) / (w.max() - w.min())
    w = (w - 0.5) / 0.5

    o = torch.from_numpy(o)
    w = torch.from_numpy(w)
    o = o.type(torch.FloatTensor)
    w = w.type(torch.FloatTensor)

    oall = []
    wall = []

    for i in range(o.shape[1]):
        print(i)
        #out = net(o[:, i, :].unsqueeze(0).unsqueeze(0).cuda())
        out = net(torch.cat([w[:, i, :].unsqueeze(0).unsqueeze(0), o[:, i, :].unsqueeze(0).unsqueeze(0)], 1).cuda())
        oout = out['out1'].detach().cpu().numpy()[0, 0, ::]
        wout = out['out0'].detach().cpu().numpy()[0, 0, ::]
        oall.append(oout)
        wall.append(wout)


    wall = np.stack(wall, 1)
    oall = np.stack(oall, 1)

    tiff.imwrite(source + 'wall.tif', wall)
    tiff.imwrite(source + 'oall.tif', oall)

    tiff.imwrite(source + 'oo.tif', o.numpy())
    tiff.imwrite(source + 'ww.tif', w.numpy())


def backward_test():
    dataset = 'Human'
    use_pseudo = True
    source = '/media/ExtHDD01/Dataset/paired_images/' + dataset + '/'
    if dataset == 'longone':
        net = torch.load('/media/ExtHDD01/logs/longone/WorkingGAN/cyc_ngf32/checkpoints/net_gXY_model_epoch_200.pth',
                         map_location='cpu').cuda()
    if dataset  == 'Human':
        net = torch.load('/media/ExtHDD01/logs/Human/cyc4_1024/cyc4/checkpoints/netGYX_model_epoch_80.pth',
                         map_location='cpu').cuda()

    ori = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyori.tif')
    if use_pseudo:
        w = tiff.imread('/media/ExtHDD01/Dataset/paired_images/' + dataset + '/xyft0.tif')

    omax, omin = ori.max(), ori.min()

    def process_one_ori(o):
        o = ori
        o = (o - o.min()) / (o.max() - o.min())
        o = (o - 0.5) / 0.5
        o = torch.from_numpy(o)
        o = o.type(torch.FloatTensor)
        if use_pseudo:
            w = (w - w.min()) / (w.max() - w.min())
            w = (w - 0.5) / 0.5
            w = torch.from_numpy(w)
            w = w.type(torch.FloatTensor)

        oall = []
        if use_pseudo:
            wall = []

        for i in range(o.shape[0]):
            print(i)

            if use_pseudo:
                out = net(torch.cat([w[i, :, :].unsqueeze(0).unsqueeze(0), o[i, :, :].unsqueeze(0).unsqueeze(0)], 1).cuda())
                oout = out['out1'].detach().cpu().numpy()[0, 0, ::]
            else:
                out = net(o[i, :, :].unsqueeze(0).unsqueeze(0).cuda())
                oout = out['out0'].detach().cpu().numpy()[0, 0, ::]

            oall.append(oout)
            if use_pseudo:
                wout = out['out0'].detach().cpu().numpy()[0, 0, ::]
                wall.append(wout)

        oall = np.stack(oall, 0)

        oall = (oall + 1) / 2
        oall = oall * (omax - omin) + omin
        oall = oall.astype(np.uint16)
        #tiff.imwrite(source + 'for_latent_yx.tif', oall[:, 64:-64, 64:-64])

        o = (o + 1) / 2
        o = o * (omax - omin) + omin
        o = o.numpy().astype(np.uint16)
        #tiff.imwrite(source + 'for_latent_original.tif', o[:, 64:-64, 64:-64])

        if use_pseudo:
            tiff.imwrite(source + 'wall.tif', wall)
            tiff.imwrite(source + 'ww.tif', w.numpy())
            wall = np.stack(wall, 0)

            return oall, wall, o, w
