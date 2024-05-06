import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def test_d2t():
    # model = torch.load('/media/ExtHDD01/logs/womac4/cyc/0/checkpoints/net_gYX_model_epoch_100.pth', map_location=torch.device('cpu'))
    g = torch.load('/media/ExtHDD01/logs/t2d/cascade0d2t/checkpoints/net_g_model_epoch_170.pth').cuda()
    gLR = torch.load('/media/ExtHDD01/logs/t2d/cascade0d2t/checkpoints/net_gLR_model_epoch_170.pth').cuda()
    dlist = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/d/*'))
    tlist = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/tres/*'))
    sys.modules['model'] = networks
    tseg = torch.load('/home/ghc/Dropbox/TheSource/scripts/WorkingGan/submodels/tse_seg_0615.pth').cuda()

    all = []
    for i in range(200, 201):
        d = tiff.imread(dlist[i])
        t = tiff.imread(tlist[i])

        (x, y0) = (d, t)
        x = x / x.max()
        x = (x - 0.5) * 2
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().cuda()

        xLR = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(x)
        yLR = gLR(xLR)['out0']
        yLRHR = torch.nn.Upsample(scale_factor=2.0, mode='bilinear')(yLR)
        y = g(torch.cat([x, yLRHR], 1))['out0']

        imagesc(x[0, 0, :, :].detach().cpu())
        imagesc(y[0, 0, :, :].detach().cpu())
        imagesc(y0)

def test_t2d():
    # model = torch.load('/media/ExtHDD01/logs/womac4/cyc/0/checkpoints/net_gYX_model_epoch_100.pth', map_location=torch.device('cpu'))

    g = torch.load('/media/ExtHDD01/logs/t2d/1/checkpoints/net_g_model_epoch_100.pth').cuda()

    g = torch.load('/media/ExtHDD01/logs/t2d/cascade0/checkpoints/net_g_model_epoch_100.pth').cuda()
    gLR = torch.load('/media/ExtHDD01/logs/t2d/cascade0/checkpoints/net_gLR_model_epoch_100.pth').cuda()
    dlist = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/d/*'))
    tlist = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/t2d/full/t/*'))
    sys.modules['model'] = networks
    tseg = torch.load('/home/ghc/Dropbox/TheSource/scripts/WorkingGan/submodels/tse_seg_0615.pth').cuda()

    all = []
    for i in range(200, 201):
        d = tiff.imread(dlist[i])
        t = tiff.imread(tlist[i])

        (x, y0) = (t, d)

        x = x / x.max()
        x = (x - 0.5) * 2
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().cuda()

        try:
            xLR = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(x)
            yLR = gLR(xLR)['out0']
            yLRHR = torch.nn.Upsample(scale_factor=2.0, mode='bilinear')(yLR)
            y = g(torch.cat([x, yLRHR], 1))['out0']
        except:
            y = g(x)['out0']

        imagesc(x[0, 0, :, :].detach().cpu())
        imagesc(y[0, 0, :, :].detach().cpu())
        #imagesc(y0)


def cyc_imorphics3d():
    upsample = torch.nn.Upsample(size=(384, 384, 23 * 8))
    model = torch.load('/media/ExtHDD01/logs/womac43d/cyc_oai3d_1/23d_rotate/checkpoints/net_g_model_epoch_60.pth',
                       map_location=torch.device('cpu'))#.eval()
    #model = torch.load('/media/ExtHDD01/logs/womac43d/cyc_oai3d_2/0/checkpoints/net_g_model_epoch_40.pth',
    #                   map_location=torch.device('cpu')).eval()
    #model = torch.load('/media/ExtHDD01/logs/womac4/cyc/0/checkpoints/net_gYX_model_epoch_100.pth',
    #                   map_location=torch.device('cpu'))
    #model = torch.load('/media/ExtHDD01/logs/womac43d/2024/cyc_oai3d_1/23d_rotate/checkpoints/net_g_model_epoch_40.pth',
    #                   map_location=torch.device('cpu'))
    model = torch.load('/media/ExtHDD01/logs/womac4/oaicyc/cyc_oai3d_1/23d_rotate_ngf32/checkpoints/net_g_model_epoch_200.pth',
                       map_location=torch.device('cpu'))#.eval() # newly ran
    #model = torch.load('/media/ExtHDD01/logs/womac43d/cyc_oai3d_1_cut/23d_rotate/checkpoints/net_g_model_epoch_70.pth',
    #                   map_location=torch.device('cpu'))#.eval() # CUT, 23d, rotate

    l = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac43d/full/ab/*'))
    x0 = tiff.imread(l[0])
    x0 = x0 / x0.max()
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
    x0 = upsample(x0)
    x0 = (x0 - 0.5) * 2

    #aprime = model(x0.permute(0, 1, 3, 2, 4))['out0'].cpu().detach().permute(0, 1, 3, 2, 4)
    a = model(x0)['out0'].cpu().detach()

    #a = (a + aprime) / 2

    #tiff.imwrite('a.tif', a[:, 0, :, :].permute(1, 0, 2).numpy())
    tiff.imwrite('enhanced.tif', a[:, 0, :, :].numpy())
    tiff.imwrite('before.tif', x0[:, 0, :, :].numpy())
    # imagesc(o[0,0,:,:].numpy())

def cyc_imorphics3d_fid():

    original = False

    upsample = torch.nn.Upsample(size=(384, 384, 23 * 8))

    #model = torch.load('/media/ExtHDD01/logs/womac43d/cyc_oai3d_1/23d_rotate/checkpoints/net_g_model_epoch_60.pth',
    #                   map_location=torch.device('cpu'))#.eval() # CUT, 23d, rotate
    #tag = '23d_rotate_60/'
    model = torch.load('/media/ExtHDD01/logs/womac4/IsoLesion/Cvgg10/checkpoints/net_g_model_epoch_160.pth',
                       map_location=torch.device('cpu'))#.eval() # newly ran
    tag = 'IsoLesion_Cvgg10_e160/'

    if original:
        tag = 'original/'

    destination = '/media/ExtHDD01/oaiout/' + tag
    # delete and create folders
    os.makedirs(destination + '/xy', exist_ok=True)
    os.makedirs(destination + '/yz', exist_ok=True)
    os.makedirs(destination + '/xz', exist_ok=True)

    l = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac43d/full/ab/*'))

    for i in range(2, 3):
        filename = l[i].split('/')[-1].split('.')[0]
        x0 = tiff.imread(l[i])
        x0 = x0 / x0.max()
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
        x0 = upsample(x0)
        x0 = (x0 - 0.5) * 2

        if original:
            out_all = x0.numpy()[0, 0, :, :, :]
        else:
            out_all = model(x0)['out0'].cpu().detach()
            out_all = out_all.numpy()[0, 0, :, :, :]

        # reslice
        for z in range(out_all.shape[2]):
            tiff.imwrite(destination + 'xy/'+ filename + '_' +str(z).zfill(3) + '.tif', out_all[:, :, z])
        for x in range(out_all.shape[0]):
            tiff.imwrite(destination + 'yz/'+ filename + '_' +str(x).zfill(3) + '.tif', out_all[x, :, :])
        for y in range(out_all.shape[1]):
            tiff.imwrite(destination + 'xz/'+ filename + '_' +str(y).zfill(3) + '.tif', out_all[:, y, :])

if 0:
    net_gan = torch.load('/media/ExtHDD01/logs/womac4/IsoLesion/Cvgg10/checkpoints/net_gy_model_epoch_200.pth',
                       map_location=torch.device('cpu')).cuda()#.eval() # newly ran
    l = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac4/full/a/*'))

    def get_xy(ax):
        mask = net_gan(ax.cuda())['out0'].detach().cpu()
        mask = nn.Sigmoid()(mask)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        ax = torch.multiply(mask, ax.detach().cpu())
        return ax, mask

    for i in range(40, 41):
        x = tiff.imread(l[i])
        x = x / x.max()
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().cuda()
        mask_all = []
        for mc in range(20):
            ax, mask = get_xy(x)
            mask_all.append(mask.detach().cpu())
        mask_all = torch.cat(mask_all, 0)
        mask_all = torch.mean(mask_all, 0).squeeze(0)

        mean_diff = x.detach().cpu() - torch.multiply(mask_all, x.detach().cpu())
        print(mean_diff.min(), mean_diff.max())

    imagesc(mean_diff[0, 0, ::])


def lesion_using_gY_after_interpolation():

    prj = '/IsoLesion/DshareZngf48mc/'
    epoch = 400

    net = torch.load('/media/ExtHDD01/logs/womac4' + prj + 'checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                       map_location=torch.device('cpu')).cuda()#.eval() # newly ran
    net_y = torch.load('/media/ExtHDD01/logs/womac4' + prj + 'checkpoints/net_gy_model_epoch_' + str(epoch) + '.pth',
                       map_location=torch.device('cpu')).cuda()#.eval() # newly ran
    #l = sorted(glob.glob('/media/ExtHDD01/oaiout/IsoLesion_DshareZngf48_e200/xy/*'))

    destination = '/media/ExtHDD01/oaiout/IsoLesion/DshareZngf48mc/'

    l = sorted(glob.glob(destination + 'xy/*'))

    os.makedirs(destination + 'diffmean', exist_ok=True)
    os.makedirs(destination + 'diffsig', exist_ok=True)

    def get_xy(ax):
        z = net(ax.unsqueeze(4).cuda(), method='encode')
        zpermute = [x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0] for x in z]
        mask = net_y(zpermute, method='decode')['out0'].detach().cpu()
        mask = nn.Sigmoid()(mask)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        ax = torch.multiply(mask, ax.detach().cpu())
        return ax, mask

    #for i in range(313, 314):
    #for i in [496]:#range(496, 497):
    for i in range(184*2, 184*3):  # range(496, 497):
        x = tiff.imread(l[i])
        #x = (x - x.min()) / (x.max() - x.min())
        x = (x + 1) / 2
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().cuda()
        mask_all = []
        diff_all = []
        for mc in range(100):
            ax, mask = get_xy(x)
            mask_all.append(mask.detach().cpu())

            diff = x.detach().cpu() - torch.multiply(mask, x.detach().cpu())
            diff_all.append(diff.detach().cpu())

        mask_all = torch.cat(mask_all, 0)
        diff_all = torch.cat(diff_all, 0)

        mask_mean = torch.mean(mask_all, 0).squeeze(0)
        mask_var = torch.var(mask_all, 0).squeeze(0)
        diff_mean = (x.detach().cpu() - torch.multiply(mask_mean, x.detach().cpu())).squeeze()
        diff_sig = torch.div(diff_all.mean(0), diff_all.var(0) + 0.01).squeeze()

        #imagesc(mean_diff[0, 0, ::])
        #imagesc(diff_all.mean(0).squeeze())
        #imagesc(diff_sig)
        tiff.imwrite(destination + 'diffmean/' + l[i].split('/')[-1], diff_mean.numpy())
        tiff.imwrite(destination + 'diffsig/' + l[i].split('/')[-1], diff_sig.numpy())


def IsoLesion_interpolate():

    upsample = torch.nn.Upsample(size=(384, 384, 23 * 8))

    if 1:
        suffix = 'X'
        l = sorted(glob.glob('/media/ExtHDD01/Dataset/paired_images/womac43d/full/ab/*'))
        irange = range(2, 3)
    else:
        suffix = 'diff'
        l = sorted(glob.glob('/media/ExtHDD01/oai_isotropic_out/IsoLesion/DshareZngf48mc/fromdiffusion/*'))
        irange = range(0, 1)

    prj = '/IsoLesionF/0/'
    epoch = 400

    net = torch.load('/media/ExtHDD01/logs/womac4' + prj + 'checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                       map_location=torch.device('cpu'))#.eval() # newly ran
    netB = torch.load('/media/ExtHDD01/logs/womac4' + prj + 'checkpoints/net_gB_model_epoch_' + str(epoch) + '.pth',
                       map_location=torch.device('cpu'))#.eval() # newly ran

    destination = '/media/ExtHDD01/oai_isotropic_out/' + prj
    # delete and create folders


    os.makedirs(destination + 'xy' + suffix, exist_ok=True)
    os.makedirs(destination + 'yz' + suffix, exist_ok=True)
    os.makedirs(destination + 'xz' + suffix, exist_ok=True)


    for i in tqdm(irange):
        filename = l[i].split('/')[-1].split('.')[0]
        x0 = tiff.imread(l[i])
        x0 = x0 / x0.max()
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
        #x0 = upsample(x0)
        x0 = (x0 - 0.5) * 2

        outz = net(x0, method='encode')

        out_all = net(outz[-1], method='decode')['out0']
        #out_all = nn.Sigmoid()(out_all)
        out_all = nn.Tanh()(out_all)
        out_all = out_all.detach().cpu()
        out_all0 = out_all.numpy()[0, 0, :, :, :]

        out_all = netB(outz[-1], method='decode')['out0']
        out_all = nn.Sigmoid()(out_all)
        #out_all = nn.Tanh()(out_all)
        out_all = out_all.detach().cpu()
        out_all1 = out_all.numpy()[0, 0, :, :, :]
        out_all0 = (out_all0 + 1) / 2
        out_all = out_all0 - np.multiply(out_all0, out_all1)

        # reslice
        for z in range(out_all.shape[2]):
            tiff.imwrite(destination + 'xy' + suffix + '/' + filename + '_' + str(z).zfill(3) + '.tif', out_all[:, :, z])
        for x in range(out_all.shape[0]):
            tiff.imwrite(destination + 'yz'+ suffix + '/' +  filename + '_' +str(x).zfill(3) + '.tif', out_all[x, :, :])
        for y in range(out_all.shape[1]):
            tiff.imwrite(destination + 'xz'+ suffix + '/' +  filename + '_' +str(y).zfill(3) + '.tif', out_all[:, y, :])


def draw_difference():
    root = '/media/ExtHDD01/oai_isotropic_out/IsoLesion/DshareZngf48mc/'
    xy = tiff.imread(root + 'xy.tif')
    xydiff = tiff.imread(root + 'xydiff.tif')

    diff = xy - xydiff
    diff[diff < 0] = 0
    tiff.imwrite(root + 'xydiffpos.tif', diff)

def remove_last_after_dash(str):
    return str[:str.rfind('_')]

def norm_11(x):
    x = (x - x.min()) / (x.max() - x.min())
    x = (x - 0.5) * 2
    return x


def test_IsoLesion_interpolate(net, x0):
    outz = net(x0, method='encode')
    out_all = net(outz[-1], method='decode')['out0']
    # out_all = nn.Sigmoid()(out_all)
    #out_all = nn.Tanh()(out_all)
    out_all = out_all.detach().cpu()
    out_all0 = out_all.numpy()[0, 0, :, :, :]
    return out_all0


def overall():
    upsample = torch.nn.Upsample(size=(384, 384, 23 * 8))

    prj = '/IsoLesion/DshareZngf48mc/'
    epoch = 400

    net = torch.load('/media/ExtHDD01/logs/womac4' + prj + 'checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                       map_location=torch.device('cpu'))#.eval() # newly ran


    root_raw = '/media/ExtHDD01/Dataset/paired_images/womac4/val/a/'
    root_dppm = '/media/ExtHDD01/oai_diffusion_result/aA/results/test/0/0/Out/'

    subjects = sorted(list(set([remove_last_after_dash(x.split('/')[-1]) for x in sorted(glob.glob(root_dppm + '*'))])))

    destination = '/media/ExtHDD01/oai_isotropic_out/Finalzied/'

    for s in subjects[0:10]:
        raw_list = sorted(glob.glob(root_raw + s + '*.tif'))
        dppm_list = sorted(glob.glob(root_dppm + s + '*.tif'))

        raw = np.stack([tiff.imread(x) for x in raw_list], 0)
        dppm = np.stack([tiff.imread(x) for x in dppm_list], 0)

        #raw = tiff.imread('/media/ExtHDD01/oai_isotropic_out/IsoLesion/DshareZngf48mc/fromdiffusion/out1.tif')

        raw = norm_11(raw)
        dppm = norm_11(dppm)

        if 0:
            raw = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
            rawup = test_IsoLesion_interpolate(net, raw)
            rawup = np.transpose(rawup, (2, 0, 1))
            tiff.imwrite(destination + 'xup/' + s + '.tif', rawup)

        dppm = torch.from_numpy(dppm).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
        dppmup = test_IsoLesion_interpolate(net, dppm)
        dppmup = np.transpose(dppmup, (2, 0, 1))
        tiff.imwrite(destination + 'dppmup/' + s + '.tif', dppmup)

