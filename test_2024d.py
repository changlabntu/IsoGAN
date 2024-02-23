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
#from loaders.data_multi import MultiData as Dataset
from dotenv import load_dotenv
import argparse
#from loaders.data_multi import PairedData, PairedData3D
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

metrics = GetAUC()
load_dotenv('.env')


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


def list_to_tensor(ax, repeat=True, norm=None):
    ax = torch.cat([torch.from_numpy(x / 1).unsqueeze(2) for x in ax], 2).unsqueeze(0).unsqueeze(1)
    # normalize ax to -1 to 1
    ax = normalize(ax, norm=norm)
    if repeat:
        ax = ax.repeat(1, 3, 1, 1, 1)
    return ax


def normalize(x, norm=None):
    if norm == '01':
        x = (x - x.min()) / (x.max() - x.min())
    elif norm == '11':
        x = (x - x.min()) / (x.max() - x.min())
        x = x * 2 - 1
    else:
        x = (x - x.min()) / (x.max() - x.min())
        if len(x.shape) > 4:
            all = []
            for i in range(x.shape[4]):
                #all.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x[:, :, :, :, i]))
                all.append(transforms.Normalize(0.485, 0.229)(x[:, :, :, :, i]))
            x = torch.stack(all, 4)
        else:
            #x = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x)
            x = transforms.Normalize(0.485, 0.229)(x)
    return x


def flip_by_label(x, label):
    y = []
    for i in range(x.shape[0]):
        if label[i] == 1:
            y.append(torch.flip(x[i, :], [0]))
        else:
            y.append(x[i, :])
    return torch.stack(y, 0)


def manual_classifier_auc(xA, xB, labels, L):
    out = net.classifier((xA - xB).cuda()).detach().cpu()
    print('AUC0=  ' + str(metrics(labels[:L], flip_by_label((out[:L, :]), 1 - labels[:L]))[0]))
    out = net.classifier(-(xA - xB).cuda()).detach().cpu()
    print('AUC1=  ' + str(metrics(labels[:L], flip_by_label((out[:L, :]), labels[:L]))[0]))

def plot_umap(fA, fB):
    import umap
    import matplotlib.pyplot as plt
    reducer = umap.UMAP()
    e = reducer.fit_transform(torch.cat([fA, fB], 0).squeeze())
    L = fA.shape[0]
    plt.figure(figsize=(10, 8))
    plt.scatter(e[:L, 0], e[:L, 1], s=0.5 * np.ones(L))
    plt.scatter(e[L:L*2, 0], e[L:L*2, 1], s=0.5 * np.ones(L))
    #plt.scatter(e[667:2225, 0], e[667:2225, 1], s=0.5 * np.ones(1558))
    #plt.scatter(e[L+667:L + 2225, 0], e[L+667:L + 2225, 1], s=0.5 * np.ones(1558))
    plt.show()

def test_eval_set_manual():
    # Forward
    outAB = []
    xA = []
    xB = []

    #for data in tqdm(eval_loader):
    print('testing over the eval set...')
    #for i in tqdm(range(len(eval_set))):
    #    a, b = eval_set.__getitem__(i)['img']

    net = torch.load(ckpt, map_location='cpu')
    net = net.cuda()
    net.classifier = net.projection

    for i, x in tqdm(enumerate(eval_loader)):
        a, _ = x['img']
        batch = a.shape[0]
        Z = a.shape[4]
        a = a.permute(0, 4, 1, 2, 3)
        a = a.reshape(batch * a.shape[1], a.shape[2], a.shape[3], a.shape[4])
        a = net(a.cuda(), alpha=0, method='encode')[-1]
        a = a.reshape(batch, Z, a.shape[1], a.shape[2], a.shape[3])
        a = a.permute(0, 2, 3, 4, 1)
        a = pool(a)[:, :, 0, 0, 0]
        xA.append(a.detach().cpu())

    net = torch.load(ckpt, map_location='cpu')
    net = net.cuda()
    net.classifier = net.projection

    for i, x in tqdm(enumerate(eval_loader)):
        _, b = x['img']
        batch = b.shape[0]
        Z = b.shape[4]
        b = b.permute(0, 4, 1, 2, 3)
        b = b.reshape(batch * b.shape[1], b.shape[2], b.shape[3], b.shape[4])
        b = net(b.cuda(), alpha=0, method='encode')[-1]
        b = b.reshape(batch, Z, b.shape[1], b.shape[2], b.shape[3])
        b = b.permute(0, 2, 3, 4, 1)
        b = pool(b)[:, :, 0, 0, 0]
        xB.append(b.detach().cpu())

    (xA, xB) = (torch.cat(x) for x in (xA, xB))
    # AUC

    # flip = labels
    flip = ((torch.rand(667) > 0.5) / 1).long()
    manual_classifier_auc(xA, xB, flip, L=667)
    return outAB, xA, xB


def gan_run(a):
    batch = a.shape[0]
    Z = a.shape[4]
    a = a.permute(0, 4, 1, 2, 3)
    a = a.reshape(batch * a.shape[1], a.shape[2], a.shape[3], a.shape[4])
    a = net(a.cuda(), alpha=0, method='encode')[-1]
    a = a.reshape(batch, Z, a.shape[1], a.shape[2], a.shape[3])
    a = a.permute(0, 2, 3, 4, 1)
    a = pool(a)[:, :, 0, 0, 0]
    return a

def test_eval_set_manual2():
    # Forward
    outAB = []
    xA = []
    xB = []

    net = torch.load(ckpt, map_location='cpu')
    net = net.cuda()
    net.classifier = net.projection

    for i, x in tqdm(enumerate(eval_loader)):
        a, b = x['img']
        ab = torch.cat([a, b], 0)
        ab = gan_run(ab)
        xA.append(ab[:a.shape[0]].detach().cpu())
        xB.append(ab[a.shape[0]:].detach().cpu())
    (xA, xB) = (torch.cat(x) for x in (xA, xB))
    # AUC

    # flip = labels
    flip = ((torch.rand(667) > 0.5) / 1).long()
    manual_classifier_auc(xA, xB, flip, L=667)
    return outAB, xA, xB

# Extra Raw Data
# diffusion data
data_root = '/home/ghc/Dataset/paired_images/womac4/val/'
alist = sorted(glob.glob(data_root + 'ap/*'))
blist = sorted(glob.glob(data_root + 'bp/*'))
dlist = sorted(glob.glob(data_root + '003b/*'))


parser = args_train()
args = parser.parse_args()

load_dotenv('env/.t09b')
x = pd.read_csv('env/womac4_moaks.csv')
labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values


# Model
ckpt_path = '/media/ExtHDD01/logscls/'

for ckpt in sorted(glob.glob('/media/ExtHDD01/logs/womac4/siamese_gan/00/checkpoints/net_g*.pth'))[2:3]:
    print(ckpt)
    net = torch.load(ckpt, map_location='cpu')
    net = net.cuda()
    net.classifier = net.projection
    net = net.eval()
    #net.classifier = net.classifier.train()


    pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))

    # NORMALIZATION
    args.n01 = True
    norm_method = '01'

    # Data
    #eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
    #                   path='ap_bp', opt=args, labels=[(int(x),) for x in labels], mode='test', index=None)

    from dataloader.data_multi import MultiData as Dataset
    test_index = None
    args.dataset_mode = 'PairedSlices3D'
    args.rotate = False
    args.rgb = False
    args.crop = 256
    args.nm = '01'
    eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
                       path='ap_bp',
                       opt=args, mode='test', index=test_index, filenames=True)
    eval_loader = DataLoader(dataset=eval_set, num_workers=1, batch_size=2, shuffle=False,
                             pin_memory=True)


    # TESTING
    outAB, xA, xB = test_eval_set_manual()
    #plot_umap(xA, xB)

