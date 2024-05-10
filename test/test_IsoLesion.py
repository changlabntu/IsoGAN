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


def remove_last_after_underline(s):
    return s[:s.rfind('_')]


def IsoLesion_interpolate(source, destination, subjects, net, to_upsample=False):

    upsample = torch.nn.Upsample(size=(384, 384, 23 * 8))

    for i in tqdm(range(len(subjects))):
        filename = subjects[i]
        x0 = tiff.imread(source + filename)

        #load 2d
        #filename = subjects[i]
        #tif_list = sorted(glob.glob(source + subjects[i] + '*.tif'))
        #x0 = np.stack([tiff.imread(x) for x in tif_list], 0)

        x0 = x0 / x0.max()
        x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
        #x0 = upsample(x0)
        x0 = (x0 - 0.5) * 2

        if to_upsample:
            x0 = upsample(x0)
        out_all = net(x0)['out0']
        #out_all = nn.Sigmoid()(out_all)
        #out_all = nn.Tanh()(out_all)
        out_all = out_all.detach().cpu()
        out_all = out_all[0, 0, :, :, :].numpy()
        out_all = np.transpose(out_all, (2, 0, 1))

        tiff.imwrite(destination + filename, out_all)


def calculate_difference(x_list, y_list, destination):
    #root = '/media/ExtHDD01/oai_diffusion_result/'
    #x_list = sorted(glob.glob(root + 'a2d/*'))
    #y_list = sorted(glob.glob(root + 'addpm2d/*'))

    norm = True

    for i in range(len(x_list)):
        x = tiff.imread(x_list[i])
        if norm:
            x = x / x.max()
            #for z in range(x.shape[0]):
            #    x[z, :, :] = (x[z, :, :] - x[z, :, :].min()) / (x[z, :, :].max() - x[z, :, :].min())
        y = tiff.imread(y_list[i])
        difference = x - y
        difference[difference < 0] = 0
        difference = (difference / 2 * 255).astype(np.uint8)
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


def expand_3d_to_2d_for_visualize(destination, suffix):
    #destination = root + 'expanded3d/'
    #suffix = 'difference2d/'
    source = root + suffix
    #os.makedirs(destination + 'xy' + suffix, exist_ok=True)
    os.makedirs(destination + 'zy' + suffix, exist_ok=True)
    os.makedirs(destination + 'zx' + suffix, exist_ok=True)

    l = sorted(glob.glob(source + '*'))
    for i in range(len(l)):
        x0 = tiff.imread(l[i])  # (Z, X, Y)
        filename = l[i].split('/')[-1].split('.')[0]
        # reslice
        #for z in range(x0.shape[0]):
        #    tiff.imwrite(destination + 'xy' + suffix + filename + '_' + str(z).zfill(3) + '.tif', x0[z, :, :])
        for x in range(x0.shape[1]):
            tiff.imwrite(destination + 'zy' + suffix + filename + '_' + str(x).zfill(3) + '.tif', np.transpose(x0[:, x, :], (1, 0)))
        for y in range(x0.shape[2]):
            tiff.imwrite(destination + 'zx' + suffix + filename + '_' + str(y).zfill(3) + '.tif', np.transpose(x0[:, :, y], (1, 0)))


if __name__ == "__main__":
    root = '/media/ExtHDD01/oai_diffusion_interpolated/0506_400/'
    ########################
    # Copy ddpm to 2d output
    ########################
    if 1:
        source = root + 'Out/'
        destination = root + 'addpm2d/'
        os.makedirs(destination, exist_ok=True)
        save_a_to_2d(source=source, subjects=get_subjects_from_list_of_2d_tifs(source), destination=destination)

    #####################
    # Copy a to 2d output
    #####################
    if 1:
        source = '/media/ExtHDD01/Dataset/paired_images/womac4/full/a/'
        destination = root + 'a2d/'
        os.makedirs(destination, exist_ok=True)
        save_a_to_2d(source=source, subjects=get_subjects_from_list_of_2d_tifs(source)[0:500:5], destination=destination)

    #####################
    # 3D interpolation
    #####################
    # path
    # model
    #prj = '/IsoLesion/DshareZngf48mc/'
    #epoch = 400
    prj = '/IsoScopeXX/cyc0lb1/'
    epoch = 200
    net = torch.load('/media/ExtHDD01/logs/womac4' + prj + 'checkpoints/net_g_model_epoch_' + str(epoch) + '.pth',
                       map_location=torch.device('cpu'))#.eval() # newly ran
    if 1:
        source = root + 'a2d/'
        destination = root + 'a3d/'
        os.makedirs(destination, exist_ok=True)
        subjects = [x.split('/')[-1] for x in sorted(glob.glob(source + '*'))][:10]
        IsoLesion_interpolate(source, destination, subjects, net, to_upsample=True)

    if 1:
        source = root + 'addpm2d/'
        destination = root + 'addpm3d/'
        os.makedirs(destination, exist_ok=True)
        subjects = [x.split('/')[-1] for x in sorted(glob.glob(source + '*'))][:10]
        IsoLesion_interpolate(source, destination, subjects, net, to_upsample=True)

    #####################
    # Calculate difference
    #####################
    if 1:
        x_list = sorted(glob.glob(root + 'a2d/*'))
        y_list = sorted(glob.glob(root + 'addpm2d/*'))
        os.makedirs(root + 'difference2d/', exist_ok=True)
        calculate_difference(x_list, y_list, destination='difference2d/')

        x_list = sorted(glob.glob(root + 'a3d/*'))
        y_list = sorted(glob.glob(root + 'addpm3d/*'))
        os.makedirs(root + 'difference3d/', exist_ok=True)
        calculate_difference(x_list, y_list, destination='difference3d/')

    if 1:
        expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='difference2d/')
        expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='difference3d/')
        expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='a2d/')
        expand_3d_to_2d_for_visualize(destination=root + 'expanded3d/', suffix='a3d/')