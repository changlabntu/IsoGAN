import time, os, glob
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.make_config import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dotenv import load_dotenv
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import tifffile as tiff
from utils.metrics_classification import ClassificationLoss, GetAUC
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import glob
import tifffile as tiff
from utils.metrics_classification import ClassificationLoss, GetAUC
from dataloader.data_multi import MultiData as Dataset
from utils.data_utils import imagesc
import umap
import matplotlib.pyplot as plt
metrics = GetAUC()


def plotting_umap(prj_name, xAv, xBv, xAyv, xByv):
    reducer = umap.UMAP()
    f = [xAv, xBv, xAyv, xByv]
    f = [projection(x).detach() for x in f]
    e = reducer.fit_transform(torch.cat(f, 0).squeeze())

    L = [x.shape[0] for x in f]
    accumulated_L = np.cumsum(L)
    accumulated_L = np.insert(accumulated_L, 0, 0)

    ediv = [e[accumulated_L[i]:accumulated_L[i + 1], :] for i in range(len(L))]


    #ediv = np.load('idx10.npy')
    plot_umap_distance(title=prj_name + '    x > xy', i0=0, i1=2, ediv=ediv)
    plot_umap_distance(title=prj_name + '    y > yy', i0=1, i1=3, ediv=ediv)
    plot_umap_distance(title=prj_name + '    xy > y', i0=2, i1=1, ediv=ediv)
    plot_umap_distance(title=prj_name + '    xy > yy', i0=2, i1=3, ediv=ediv)


def plot_umap_distance(title, i0, i1, ediv):
    dist = np.linalg.norm(ediv[i0] - ediv[i1], axis=1)
    for i in range(ediv[i0].shape[0]):
        if dist[i] >= 0.1:
            plt.arrow(ediv[i0][i, 0], ediv[i0][i, 1], ediv[i1][i, 0] - ediv[i0][i, 0], ediv[i1][i, 1] - ediv[i0][i, 1],
                      head_width=0.05, color='r', alpha=0.2)
        if dist[i] < 0.1:
            plt.arrow(ediv[i0][i, 0], ediv[i0][i, 1], ediv[i1][i, 0] - ediv[i0][i, 0], ediv[i1][i, 1] - ediv[i0][i, 1],
                      head_width=0.05, color='g', alpha=0.2)
    plt.title(title.replace('/', '_') + '  dist<0.1=' + str(np.sum(dist < 0.1) / len(dist)))
    #plt.show()
    plt.savefig('outimg/' + title.replace('/', '_') + '  ' + auc_val + '.png')
    plt.close()


def args_train():
    parser = argparse.ArgumentParser()

    # projects
    parser.add_argument('--prj', type=str, default='', help='name of the project')

    # training modes
    parser.add_argument('--legacy', action='store_true', dest='legacy', default=False, help='legacy pytorch')
    parser.add_argument('--par', dest='parallel', action="store_true", help='run in multiple gpus')
    # training parameters
    parser.add_argument('-e', '--epochs', dest='epochs', default=101, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--bu', '--batch-update', dest='batch_update', default=1, type=int, help='batch to update')
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.0005, type=float, help='learning rate')

    parser.add_argument('-w', '--weight-decay', dest='weight_decay', default=0.005, type=float, help='weight decay')
    # optimizer
    parser.add_argument('--op', dest='op', default='sgd', type=str, help='type of optimizer')

    # models
    parser.add_argument('--fuse', dest='fuse', default='')
    parser.add_argument('--backbone', dest='backbone', default='vgg11')
    parser.add_argument('--pretrained', dest='pretrained', default=True)
    parser.add_argument('--freeze', action='store_true', dest='freeze', default=False)
    parser.add_argument('--classes', dest='n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--repeat', type=int, default=0, help='repeat the encoder N time')

    # misc
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')

    parser.add_argument('--dataset', type=str, default='womac4')
    parser.add_argument('--load3d', action='store_true', dest='load3d', default=True)
    parser.add_argument('--direction', type=str, default='a_b', help='a2b or b2a')
    parser.add_argument('--flip', action='store_true', dest='flip', default=False, help='image flip left right')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--cropsize', type=int, default=0)
    parser.add_argument('--n01', action='store_true', dest='n01', default=False)
    parser.add_argument('--trd', type=float, dest='trd', help='threshold of images', default=0)
    parser.add_argument('--preload', action='store_true', help='preload the data once to cache')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--fold', type=int, default=None)

    return parser


def get_xy(ax, alpha):
    mask = net_gan(ax.cuda(), alpha=alpha)['out0'].detach().cpu()
    mask = nn.Sigmoid()(mask)
    ax = torch.multiply(mask, ax.detach().cpu())
    return ax, mask

def single_encode_d(data, net, max=True):
    xAll = []
    for i in tqdm(range(0, len(data))):
        x = data[i]  # (Z, 1, H, W)
        dx = 64
        if dx > 0:
            x = x[:, :, dx:-dx, dx:-dx]
        x = x.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        x_ = test_method(x, net, max=max)
        xAll.append(x_.detach().cpu())
    if max:
        xAll = torch.cat(xAll, 0)
    return xAll


def monte_carlo_out():
    method = get_xy
    os.makedirs('/media/ExtHDD01/Dataset/paired_images/womac4/full/gan022724/mA/', exist_ok=True)
    os.makedirs('/media/ExtHDD01/Dataset/paired_images/womac4/full/gan022724/sA/', exist_ok=True)
    for ii in tqdm(range(2, 3)):
        batch = full_set.__getitem__(ii)
        a, b = batch['img']
        filenames = batch['filenames']
        (a, b) = (y.permute(3, 0, 1, 2).cuda() for y in (a, b))
        x = a
        xy_all = []
        for mc in range(100):
            xy, xymask = method(x, alpha=1)
            xy_all.append(x.cpu() - xy)

        xy_all = torch.stack(xy_all, 4)
        xy_mean = xy_all.mean(4)
        xy_var = xy_all.var(4)
        xy_sig = torch.divide(xy_mean, torch.sqrt(xy_var) + 0.01)

        destination = '/media/ExtHDD01/Dataset/paired_images/womac4/full/gan022724/'
        # remove and
        for i in range(xy_mean.shape[0]):
            tiff.imwrite(destination + 'mA/' + prj_name.replace('/', '_') + '.tif',
                         xy_mean[12, ::].detach().cpu().numpy().squeeze())
            tiff.imwrite(destination + 'sA/' + prj_name.replace('/', '_') + '.tif',
                         xy_sig[12, ::].detach().cpu().numpy().squeeze())

def get_models(prj_name, net_d=None):
    # ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/MOAKSID/adv1/00/checkpoints/net_d*.pth'))[7]#[7]
    ckpt = '/media/ExtHDD01/logs/womac4/' + prj_name + '/checkpoints/net_g_model_epoch_' + str(epoch) + '.pth'  # [7]
    # ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/MOAKS/contra/0/checkpoints/net_g*.pth'))[7]#[7]
    print(ckpt)
    net_gan = torch.load(ckpt, map_location='cpu')

    # THD OLD ONE
    #net_gan = torch.load('/media/ExtHDD01/logs/womac4old/3D/test4fixmcVgg10/checkpoints/net_g_model_epoch_40.pth',
    #                   map_location=torch.device('cpu')).cuda()

    net_gan = net_gan.cuda()

    try:
        net_ganY = torch.load(ckpt.replace('net_g', 'net_gY'), map_location='cpu')
        net_ganY = net_ganY.cuda()
        net_ganY = net_ganY.eval()
    except:
        net_ganY = None
        print('no net_gY')

    # classifier

    prj = '/media/ExtHDD01/logs/womac4/' + prj_name + '/'

    ckpt = prj + 'checkpoints/net_d_model_epoch_' + str(epoch) + '.pth'  # [7]
    # ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/MOAKS/contra/0/checkpoints/net_d*.pth'))[7]#[7]
    print(ckpt)
    net = torch.load(ckpt, map_location='cpu')
    net = net.eval()
    net = net.cuda()
    classifier = torch.load(prj + 'checkpoints/classifier_model_epoch_' + str(epoch) + '.pth', map_location='cpu')
    classifier = classifier.cuda()
    classifier = classifier.eval()
    projection = torch.load(prj + 'checkpoints/projection_model_epoch_' + str(epoch) + '.pth', map_location='cpu')
    projection = projection.eval()
    return net, classifier, net_gan, net_ganY, projection

from test_all_new import test_method, test_auc, test_by_dataset
def test_gan_by_dataset(data_set, method):
    out_all = []
    for i, x in enumerate(tqdm(data_set)):
        (x, ) = (y.permute(3, 0, 1, 2).cuda() for y in (x, ))
        xy, xymask = method(x, alpha=1)
        out_all.append(xy)
    return out_all


def GODPLZ0():
    print('classifier based on xAv - xAyv and xBv - xByv')
    x0, x1 = xAv - xAyv, xBv - xByv
    #[x0, x1] = [projection(x).detach() for x in (x0, x1)]
    xall = torch.cat([x0, x1], 0)
    out = []
    for i in range(xall.shape[0]):
        out.append(classifier(xall[i, :].unsqueeze(0).cuda()).detach().cpu())
    out = torch.cat(out)
    print('AUC0=  ' + str(metrics(np.concatenate([0 * np.ones(xAv.shape[0]), 1 * np.ones(xAv.shape[0])], 0), out)[0]))
    auc_val = str(metrics(np.concatenate([0 * np.ones(xAv.shape[0]), 1 * np.ones(xAv.shape[0])], 0), out)[0])[0:5]
    return auc_val


def GODPLZvgg():
    print('classifier based on xAv - xAyv and xBv - xByv')
    x0, x1 = xAv - xAyv, xBv - xByv
    xall = torch.cat([x0, x1], 0)
    out = []
    for i in range(xall.shape[0]):
        out.append(dvgg.classifier(xall[i, :].unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()).detach()[:, :, 0, 0].cpu())
    out = torch.cat(out)
    print('AUC0=  ' + str(metrics(np.concatenate([0 * np.ones(xAv.shape[0]), 1 * np.ones(xAv.shape[0])], 0), out)[0]))
    auc_val = str(metrics(np.concatenate([0 * np.ones(xAv.shape[0]), 1 * np.ones(xAv.shape[0])], 0), out)[0])[0:5]
    return auc_val


def test_vgg(x):
    x = dvgg.features(x.repeat(1, 3, 1, 1).cuda())
    x = torch.mean(x, dim=(2, 3))
    x, _ = torch.max(x, 0)
    return x.detach().cpu()



## Get Models

if __name__ == '__main__':
    ## Get Dataset
    load_dotenv('env/.t09b')
    parser = args_train()
    args = parser.parse_args()

    args.dataset_mode = "PairedSlices3D"
    args.rotate = False
    args.rgb = False
    args.nm = '01'
    eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
                       path='ap_bp', opt=args, labels=None, mode='test', index=None)
    # eval_set.shuffle_images()

    dvgg = torch.load('/media/ExtHDD01/logscls/moaksidvgg11max2/checkpoints/20.pth')
    #out = dv([torch.rand(1, 3, 256, 256, 23), torch.rand(1, 3, 256, 256, 23)])
    #x0 = torch.mean(x0, dim=(2, 3))  # (B, 512, 1, 1, 23)
    #x1 = torch.mean(x1, dim=(2, 3))
    #x0, _ = torch.max(x0, 2)
    #x1, _ = torch.max(x1, 2)


    #full_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/full/',
    #                   path='ap_bp', opt=args, labels=None, mode='test', index=None)
    eval_200 = Dataset(root='/media/ExtHDD01/Dataset/paired_images/' + args.dataset + '/val/',
                       path='a_b', opt=args, labels=None, mode='test', index=range(200))
    diff_200 = Dataset(root='/media/ExtHDD01/diff0/',
                       path='A2_B2', opt=args, labels=None, mode='test', index=range(200))

    for prj_name in ['MOAKSID/idx/0', 'MOAKSID/idx/0tc', 'MOAKSID/idx/0vgg10',
                     'MOAKSID/idx/0ngf64vgg10', 'MOAKSID/idx/gcls0', 'MOAKSID/idx1/0', 'MOAKSID/idx/gcls0l1',
                     'MOAKSID/idx0gcls0cut/0', 'MOAKSID/idx0gcls0cut/0cls0',
                     'MOAKSID/idx0xy/tcxy', 'MOAKSID/idx0xy/tcxyadv'][-2:]:
        epoch = 200
        net, classifier, net_gan, net_ganY, projection = get_models(prj_name=prj_name)

        if 1:  # encode and test original image by gan classifier
            xAv, xBv = test_by_dataset(eval_set, net)
            test_auc(xAv, xBv, classifier)
        if 1:  # get img xy by gan
            Ayv = test_gan_by_dataset((eval_set.__getitem__(i)['img'][0] for i in range(len(eval_set))), method=get_xy)
            Byv = test_gan_by_dataset((eval_set.__getitem__(i)['img'][1] for i in range(len(eval_set))), method=get_xy)
        if 1:  # encode gan img by gan discriminator
            (xAyv, xByv) = (single_encode_d(x, net) for x in (Ayv, Byv))
            del Ayv, Byv
            print('xBv vs xByv')
            test_auc(xBv, xByv, classifier)
        if 1:
            auc_val = GODPLZ0()
            plotting_umap(prj_name, xAv, xBv, xAyv, xByv)
        if 0:  # get monte carlo output
            monte_carlo_out()
        if 0: # trying diffusion results
            xAv, xBv = test_by_dataset(eval_200, net)
            test_auc(xAv, xBv, classifier)
            Ayv = []
            Byv = []
            for i, batch in enumerate(diff_200):
                (a, b) = batch['img']
                Ayv.append(a.permute(3, 0, 1, 2))
                Byv.append(b.permute(3, 0, 1, 2))
            (xAyv, xByv) = (single_encode_d(x, net) for x in (Ayv, Byv))
            auc_val = GODPLZ0()
            plotting_umap(prj_name, xAv, xBv, xAyv, xByv)
        if 0: # trying vgg classifier
            Av = []
            Bv = []
            for i, batch in enumerate(eval_200):
                (a, b) = batch['img']
                Av.append(a.permute(3, 0, 1, 2))
                Bv.append(b.permute(3, 0, 1, 2))
            xAv = [test_vgg(x) for x in Av]
            xBv = [test_vgg(x) for x in Bv]
            xAv = torch.stack(xAv, 0)
            xBv = torch.stack(xBv, 0)
            del Av, Bv

            Ayv = test_gan_by_dataset((eval_set.__getitem__(i)['img'][0] for i in range(len(eval_set))), method=get_xy)
            Byv = test_gan_by_dataset((eval_set.__getitem__(i)['img'][1] for i in range(len(eval_set))), method=get_xy)

            xAyv = [test_vgg(x) for x in Ayv]
            xByv = [test_vgg(x) for x in Byv]
            xAyv = torch.stack(xAyv, 0)
            xByv = torch.stack(xByv, 0)
            auc_val = GODPLZvgg()
            plotting_umap(prj_name, xAv, xBv, xAyv, xByv)

    # plt histogram
    if 0:
        distAv = torch.norm(xAv - xAyv, dim=1).detach().unsqueeze(1)
        distBv = torch.norm(xBv - xByv, dim=1).detach().unsqueeze(1)
        plt.hist(distAv.detach().cpu().numpy(), bins=100, alpha=0.5, label='A')
        plt.hist(distBv.detach().cpu().numpy(), bins=100, alpha=0.5, label='B')
       # plt.xlim([0, 0.5])
        plt.show()

    if 0:  # Do training
        train_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/train/',
                           path='ap_bp', opt=args, labels=None, mode='test', index=None)
        xAt, xBt = test_by_dataset(train_set, net)
        test_auc(xAt, xBt, classifier)
        Ayt = test_gan_by_dataset((train_set.__getitem__(i)['img'][0] for i in range(len(train_set))), method=get_xy)
        Byt = test_gan_by_dataset((train_set.__getitem__(i)['img'][1] for i in range(len(train_set))), method=get_xy)
        (xAyt, xByt) = (single_encode_d(x) for x in (Ayt, Byt))
        del Ayt, Byt


    def GODPIZ1():
        Av2 = torch.cat([xAv, xAyv], 1)
        Bv2 = torch.cat([xBv, xByv], 1)

        At2 = torch.cat([xAt, xAyt], 1)
        Bt2 = torch.cat([xBt, xByt], 1)

        X_train = torch.cat([At2, Bt2], 0).numpy()
        X_test = torch.cat([Av2, Bv2], 0).numpy()

        distAt = torch.norm(xAt - xAyt, dim=1).detach().unsqueeze(1)
        distBt = torch.norm(xBt - xByt, dim=1).detach().unsqueeze(1)

        X_train = torch.cat([distAt, distBt], 0).numpy()
        X_test = torch.cat([distAv, distBv], 0).numpy()

        Lt = At2.shape[0]
        Lv = Av2.shape[0]

        y_train = np.array([0] * Lt + [1] * Lt)
        y_test = np.array([0] * Lv + [1] * Lv)
        # Create a SVM Classifier
        clf = svm.SVC(kernel='linear', probability=True)  # Linear Kernel
        # Train the model using the training sets
        clf.fit(X_train, y_train)
        # Predict the response for test dataset
        y_pred = clf.predict(X_test)
        # Evaluate accuracy
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        # Compute AUC Score
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(['Acc contra: ' + "{0:0.3f}".format(accuracy_score(y_test, y_pred)) +
               ' AUC contra: ' + "{0:0.3f}".format(roc_auc_score(y_test, y_pred_proba))])
