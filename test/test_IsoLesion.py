import torch
import tifffile as tiff
from utils.data_utils import imagesc
import os, glob, sys
import networks, models
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import skimage.measure


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


    def seg_bone(source, destination, filter_area=False):
        os.makedirs(destination, exist_ok=True)
        flist = [x.split('/')[-1] for x in sorted(glob.glob(source + '*'))]

        #x = tiff.imread('/media/ExtHDD01/oai_diffusion_interpolated/DshareZngf48mc_0504/a2d/9000099_03.tif')
        for f in flist[:]:
            print(f)
            x = tiff.imread(source + f)
            x = (x - x.min()) / (x.max() - x.min())
            x = (x - 0.5) / 0.5
            x = torch.from_numpy(x).unsqueeze(1).float()

            bone = []
            for z in range(x.shape[0]):
                slice = x[z:z+1, :, :, :]
                #slice = slice / slice.max()
                #slice = (slice - 0.5) / 0.5
                out = seg(slice.cuda()).detach().cpu()
                out = torch.argmax(out, 1).squeeze()
                out = (out > 0).numpy().astype(np.uint8)
                bone.append(out)
            bone = np.stack(bone, axis=0)
            if filter_area:
                bone = filter_out_secondary_areas(bone).astype(np.uint8)
            tiff.imwrite(destination + f, bone)

    def torch_downsample_then_upsample(x):
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
        x = torch.nn.functional.interpolate(x, size=(23, 384, 384), mode='trilinear', align_corners=False)
        x = torch.nn.functional.interpolate(x, size=(184, 384, 384), mode='trilinear', align_corners=False)
        x = x.squeeze().numpy()
        return x

    def filter_out_secondary_areas(x):
        x = skimage.measure.label(bone)
        x0 = 0 * x
        areas = np.bincount(x.flatten())
        # top 3 areas:
        top2 = np.argsort(areas)[::-1][1:3]
        for t in top2:
            x0 += (x == t)
        return x0


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


def remove_last_after_underline(s):
    return s[:s.rfind('_')]


def IsoLesion_interpolate(source, destination, subjects, net, to_upsample=False, padding=False):

    for i in tqdm(range(len(subjects))):

        filename = subjects[i]
        x0 = tiff.imread(source + filename)

        x0 = x0 / x0.max()

        # crop Y axis
        #x0 = x0[:, 16:-16, :]

        #load 2d
        #filename = subjects[i]
        #tif_list = sorted(glob.glob(source + subjects[i] + '*.tif'))
        #x0 = np.stack([tiff.imread(x) for x in tif_list], 0)

        #x0[x0<=100] = 100

        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
        #x0 = upsample(x0)
        x0 = (x0 - 0.5) * 2

        #x0 = x0 * 0.7
        #x0[x0 >=1]  = 1
        #x0[x0 <=-1] = -1
        #x0[0, 0, 0, :32, :] = 1
        #x0[0, 0, 0, 32:, :] = -1

        print(x0.shape)
        #x0[x0<=-0.8] = -0.8

        x0 = x0[:, :, :, :, :]

        if padding:
            pad0 = 0 * torch.ones((x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], 2))
            pad = 1 * torch.ones((x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], 2))
            x0 = torch.cat((pad, pad0, x0, pad0, pad), 4)

        if to_upsample:
            upsample = torch.nn.Upsample(size=(x0.shape[2], x0.shape[3], x0.shape[4] * 8),mode='nearest')
            x0 = upsample(x0)

        #out_all = test_once(x0, net)
        #tiff.imwrite(destination + filename, out_all)

        if 0:
            o1 = []
            o0 = []
            o2 = []
            for mc in range(1):
                o1.append(test_once(x0[:, :, :, :, 64:-64], net))
                o0.append(test_once(x0[:, :, :, :, :128], net))
                o2.append(test_once(x0[:, :, :, :, -128:], net))
            o1 = np.stack(o1, 3)
            o0 = np.stack(o0, 3)
            o2 = np.stack(o2, 3)
            o1 = np.mean(o1, 3)
            o0 = np.mean(o0, 3)
            o2 = np.mean(o2, 3)

            mean01 = o1[:64, :, :].mean()
            mean12 = o1[64:, :, :].mean()

            o0m = o0 - o0[-64:, :, :].mean() + mean01
            o2m = o2 - o2[:64, :, :].mean() + mean12

            # combine o0 and o1 with the overlapped area weighted by a linear function of x
            w = np.linspace(0, 1, 64)
            w = torch.from_numpy(w).float()
            w = w.unsqueeze(1).unsqueeze(2).repeat(1, x0.shape[2], x0.shape[3]).numpy()

            combineA = o0m[:64, :, :]
            combineB = np.multiply(1 - w, o0m[64:, :, :]) + np.multiply(w, o1[:64, :, :])
            combineC = o1[64:-64, :, :]
            combineD = np.multiply(1 - w, o1[-64:, :, :]) + np.multiply(w, o2m[:64, :, :])
            combineE = o2m[64:, :, :]

            combine = np.concatenate((combineA, combineB, combineC, combineD, combineE), 0)
        else:
            combine = test_once(x0, net)

        tiff.imwrite(destination + filename, combine)


def test_once(x0, net):
    out_all = net(x0)['out0']
    out_all = out_all.detach().cpu()
    out_all = out_all[0, 0, :, :, :].numpy()
    out_all = np.transpose(out_all, (2, 0, 1))
    return out_all

def calculate_difference(x_list, y_list, destination, mask_list=None):
    #root = '/media/ExtHDD01/oai_diffusion_result/'
    #x_list = sorted(glob.glob(root + 'a2d/*'))
    #y_list = sorted(glob.glob(root + 'addpm2d/*'))

    for i in range(len(x_list)):
        x = tiff.imread(x_list[i])
        y = tiff.imread(y_list[i])

        #x = x / x.max()
        #y = y / y.max()

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        #for z in range(x.shape[0]):
        #    x[z, :, :] = (x[z, :, :] - x[z, :, :].min()) / (x[z, :, :].max() - x[z, :, :].min())
        #    y[z, :, :] = (y[z, :, :] - y[z, :, :].min()) / (y[z, :, :].max() - y[z, :, :].min())

        difference = x - y
        difference[difference < 0] = 0
        difference = difference / 1
        #difference = (difference + 1) / 2
        #difference = (difference - difference.min()) / (difference.max() - difference.min())
        difference = (difference * 255).astype(np.uint8)
        tiff.imwrite(root + destination + x_list[i].split('/')[-1], difference)


def save_a_to_2d(source, subjects, destination):
    for i in tqdm(range(len(subjects))):
        filename = subjects[i]
        tif_list = sorted(glob.glob(source + subjects[i] + '*.tif'))
        x0 = np.stack([tiff.imread(x) for x in tif_list], 0)
        tiff.imwrite(destination + filename + '.tif', x0)


def get_subjects_from_list_of_2d_tifs(source):
    l = sorted(glob.glob(source + '*'))
    subjects = sorted(list(set([remove_last_after_underline(x.split('/')[-1]) for x in l])))
    return subjects

def to_8bit(x):
    x = (x - x.min()) / (x.max() - x.min())
    x = (x * 255).astype(np.uint8)
    return x

def norm_11(x):
    x = (x - x.min()) / (x.max() - x.min())
    x = (x * 2) - 1
    return x

def expand_3d_to_2d_for_visualize(destination, suffix, png=False, upsample=None):
    #destination = root + 'expanded3d/'
    #suffix = 'difference2d/'
    source = root + suffix
    #os.makedirs(destination + 'xy' + suffix, exist_ok=True)
    os.makedirs(destination + 'zy' + suffix, exist_ok=True)
    os.makedirs(destination + 'zx' + suffix, exist_ok=True)
    os.makedirs(destination + 'xy' + suffix, exist_ok=True)

    l = sorted(glob.glob(source + '*'))
    for i in tqdm(range(len(l))):
        x0 = tiff.imread(l[i])  # (Z, X, Y)
        if upsample is not None:
            x0 = norm_11(x0)  # ?????
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
            x0 = torch.nn.functional.interpolate(x0, scale_factor=(upsample, 1, 1), mode='trilinear', align_corners=False)
            x0 = x0.squeeze().numpy()
            #x0 = x0.astype(np.uint8)

        filename = l[i].split('/')[-1].split('.')[0]
        # reslice
        for x in range(x0.shape[1]):
            if not png:
                tiff.imwrite(destination + 'zy' + suffix + filename + '_' + str(x).zfill(3) + '.tif', np.transpose(x0[:, x, :], (1, 0)))
            else:
                out = Image.fromarray(to_8bit(np.transpose(x0[:, x, :] , (1, 0))))
                out.save(destination + 'zy' + suffix + filename + '_' + str(x).zfill(3) + '.png')
        for y in range(x0.shape[2]):
            if not png:
                tiff.imwrite(destination + 'zx' + suffix + filename + '_' + str(y).zfill(3) + '.tif', np.transpose(x0[:, :, y], (1, 0)))
            else:
                out = Image.fromarray(to_8bit(np.transpose(x0[:, :, y], (1, 0))))
                out.save(destination + 'zx' + suffix + filename + '_' + str(y).zfill(3) + '.png')
        for z in range(x0.shape[0]):
            if not png:
                tiff.imwrite(destination + 'xy' + suffix + filename + '_' + str(z).zfill(3) + '.tif', x0[z, :, :])
            else:
                out = Image.fromarray(to_8bit(x0[z, :, :]))
                out.save(destination + 'xy' + suffix + filename + '_' + str(z).zfill(3) + '.png')


if __name__ == "__main__":
    #####################
    # 3D interpolation
    #####################
    # path
    # model
    if 0:
        prj = '/IsoLesion/DshareZngf48mc/'
        epoch = 200
        to_upsample = False
    elif 0:
        prj = '/IsoScopeXX/cyc0lb1skip4ndf32/'
        epoch = 320
        #prj = '/IsoScopeXX/cyc0lb1skip4ndf32randl1/'
        #epoch = 300
        to_upsample = True
    else:
        prj = '/IsoScopeXX/cyc0lb1skip4ndf32nomc/'
        epoch = 700
        to_upsample = True

    net = torch.load('/media/ExtHDD01/logs/womac4' + prj + 'checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                       map_location=torch.device('cpu'))#.eval() # newly ran

    # output root
    dppm_source = '/media/ExtHDD01/oai_diffusion_interpolated/diff0504/'
    root = '/media/ExtHDD01/oai_diffusion_interpolated/new/'
    ########################
    # Copy ddpm to 2d output
    ########################
    if 0:
        destination = root + 'addpm2d/'
        os.makedirs(destination, exist_ok=True)
        save_a_to_2d(source=dppm_source, subjects=get_subjects_from_list_of_2d_tifs(dppm_source), destination=destination)

    #####################
    # Copy a to 2d output
    #####################
    if 0:
        source = '/media/ExtHDD01/Dataset/paired_images/womac4/full/a/'
        destination = root + 'a2d/'
        os.makedirs(destination, exist_ok=True)
        save_a_to_2d(source=source, subjects=get_subjects_from_list_of_2d_tifs(source)[0:500:5], destination=destination)

    if 1:
        source = '/media/ExtHDD01/oai_diffusion_interpolated/original/' + 'a2d/'
        destination = root + 'a3d/'
        os.makedirs(destination, exist_ok=True)
        subjects = [x.split('/')[-1] for x in sorted(glob.glob(source + '*'))][:10]
        IsoLesion_interpolate(source, destination, subjects, net, to_upsample=to_upsample, padding=False)
        expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='a3d/')

    if 0:
        source = root + 'addpm2d/'
        destination = root + 'addpm3d/'
        os.makedirs(destination, exist_ok=True)
        subjects = [x.split('/')[-1] for x in sorted(glob.glob(source + '*'))][:]
        IsoLesion_interpolate(source, destination, subjects, net, to_upsample=to_upsample)


    #####################
    # Calculate difference
    #####################
    if 0:
        x_list = sorted(glob.glob(root + 'a2d/*'))
        y_list = sorted(glob.glob(root + 'addpm2d/*'))
        destination = 'difference2d/'
        os.makedirs(root + destination, exist_ok=True)
        calculate_difference(x_list, y_list, destination=destination)
    if 0:
        x_list = sorted(glob.glob(root + 'a3d/*'))
        y_list = sorted(glob.glob(root + 'addpm3d/*'))
        mask_list  = sorted(glob.glob(root + 'difference2d/*'))
        destination = 'difference3d/'
        os.makedirs(root + destination, exist_ok=True)
        calculate_difference(x_list, y_list, destination=destination)

    if 0:
        #expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='difference2d/', upsample=8)
        #expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='difference3d/')
        expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='a2d/', upsample=8)
        #expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='a3d/')

        #expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='addpm2d/')
        #expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='addpm3d/')


    if 0:
        #seg = torch.load('/home/ghc/Dropbox/TheSource/scripts/ContrastiveDiffusion/submodels/atten_0706.pth').eval()
        #seg = torch.load('/home/ghc/Dropbox/TheSource/scripts/ContrastiveDiffusion/submodels/80.pth').eval()
        seg = torch.load('/home/ghc/Dropbox/TheSource/scripts/ContrastiveDiffusion/submodels/dual_atten_0706.pth').eval()
        seg_bone(root + 'a2d/', root + 'a2d_bone/')
        #seg_bone(root + 'addpm2d/', root + 'addpm2d_bone/')
        seg_bone(root + 'a3d/', root + 'a3d_bone/')

    #y_list = sorted(glob.glob(root + 'a2d_bone/*'))
    #x_list = sorted(glob.glob(root + 'addpm2d_bone/*'))
    #os.makedirs(root + 'difference2d_bone/', exist_ok=True)
    #calculate_difference(x_list, y_list, destination='difference2d_bone/')

    if 0:
        root = '/media/ExtHDD01/oai_diffusion_interpolated/new/expanded3d/'
        xlist = sorted(glob.glob(root + 'zxdifference3d/*'))
        ylist = sorted(glob.glob(root + 'zxdifference2d/*'))

        for (x, y) in zip(xlist, ylist):
            x0 = tiff.imread(x)
            y0 = tiff.imread(y)
            print(x0.min(), x0.max(), y0.min(), y0.max())
            x0 = x0 / 255
            y0 = (y0 > 0) / 1
            mul = np.multiply(x0, y0)
            mul = (mul * 255).astype(np.uint8)
            tiff.imwrite(x.replace('zxdifference3d', 'mul'), mul)