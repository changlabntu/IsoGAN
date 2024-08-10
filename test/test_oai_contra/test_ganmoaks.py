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
from dataloader.data_multi import MultiData as Dataset
from dotenv import load_dotenv
import argparse
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
    out = classifier((xA - xB).cuda()).detach().cpu()
    print('AUC0=  ' + str(metrics(labels[:L], flip_by_label((out[:L, :]), 1 - labels[:L]))[0]))
    out = classifier(-(xA - xB).cuda()).detach().cpu()
    print('AUC1=  ' + str(metrics(labels[:L], flip_by_label((out[:L, :]), labels[:L]))[0]))


#def cls_methods(xA, xB, method='diff'):

def testing_manual(data_set, crop=False):
    # Forward
    outAB = []
    xA = []
    xB = []

    for i in tqdm(range(len(data_set))):
        batch = data_set.__getitem__(i)
        (a, b) = batch['img']

        if crop:
            dx = 64
            a = a[:, dx:-dx, dx:-dx, 4:20]
            b = b[:, dx:-dx, dx:-dx, 4:20]

        (a, b) = (x.unsqueeze(0).cuda() for x in (a, b))

        xA_ = test_method(a, net)
        xB_ = test_method(b, net)

        xA.append(xA_.detach().cpu())
        xB.append(xB_.detach().cpu())

    (xA, xB) = (torch.cat(x) for x in (xA, xB))

    xLabel = np.random.random(len(xA)) > 0.5
    x0 = flip_by_label(xA, xLabel)
    x1 = flip_by_label(xB, xLabel)

    out = classifier((x0 - x1).cuda()).detach().cpu()
    print('AUC0=  ' + str(metrics(1 - xLabel, out)[0]))

    out = classifier((x1 - x0).cuda()).detach().cpu()
    print('AUC0=  ' + str(metrics(xLabel, out)[0]))
    # AUC
    manual_classifier_auc(xA, xB, xLabel, L=xA.shape[0])
    return outAB, xA, xB


def test_method(x, net):
    (B, C, H, W, Z) = x.shape
    x = net(x.permute(0, 4, 1, 2, 3).reshape(B * Z, C, H, W)[:, :, :, :].cuda())[-1]
    x = x.permute(1, 2, 3, 0).unsqueeze(0)
    x = torch.mean(x, dim=(2, 3))
    x, _ = torch.max(x, 2)
    return x


def test_method_dual(x, net):
    (B, C, H, W, Z) = x.shape
    x = net(x.permute(0, 4, 1, 2, 3).reshape(B * Z, C, H, W)[:, :, :, :].cuda())[-1]
    x0 = x[:Z, ::]
    x1 = x[Z:, ::]
    x0 = x0.permute(1, 2, 3, 0).unsqueeze(0)
    x1 = x1.permute(1, 2, 3, 0).unsqueeze(0)
    x0 = torch.mean(x0, dim=(2, 3))
    x0, _ = torch.max(x0, 2)
    x1 = torch.mean(x1, dim=(2, 3))
    x1, _ = torch.max(x1, 2)
    return x0, x1


def testing_manual_dual(data_set, crop=False):
    # Forward
    outAB = []
    xA = []
    xB = []
    xLabel = []

    for i in tqdm(range(len(data_set))):
        batch = data_set.__getitem__(i)
        (a, b) = batch['img']
        xLabel.append(batch['labels'][0])
        if crop:
            a = a[:, 32:-32, 32:-32, :]
            b = b[:, 32:-32, 32:-32, :]
        # Raw Loading method
        #(a, b) = (list_to_tensor(x, repeat=True, norm=norm_method).float() for x in (ax, bx))
        (a, b) = (x.unsqueeze(0).cuda() for x in (a, b))
        if np.random.random() > 0.5:
            xA_, xB_ = test_method_dual(torch.cat([a, b], 0), net)
            #xLabel.append(1)
        else:
            xB_, xA_ = test_method_dual(torch.cat([b, a], 0), net)
            #xLabel.append(0)
        xA.append(xA_.detach().cpu())
        xB.append(xB_.detach().cpu())


    (xA, xB) = (torch.cat(x) for x in (xA, xB))
    # AUC
    xLabel = torch.tensor(xLabel)
    x0 = flip_by_label(xA, xLabel)
    x1 = flip_by_label(xB, xLabel)

    out = classifier((x0 - x1).cuda()).detach().cpu()
    print('AUC0=  ' + str(metrics(1 - xLabel, out)[0]))

    out = classifier((x1 - x0).cuda()).detach().cpu()
    print('AUC0=  ' + str(metrics(xLabel, out)[0]))

    manual_classifier_auc(xA, xB, xLabel, L=xA.shape[0])
    return outAB, xA, xB


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


def test_diffusion():
    print('testing over the diffusion set...')
    xAD = []
    xBD = []
    xD = []
    for i in tqdm(range(len(dlist) // 23)):
        (ax, bx, dx) = ([tiff.imread(x) for x in y[i * 23:(i + 1) * 23]] for y in (alist, blist, dlist))
        (ax, bx, dx) = (list_to_tensor(x, repeat=True, norm=norm_method).float() for x in (ax, bx, dx))

        xA_ = test_method(ax, net)
        xB_ = test_method(bx, net)
        xD_ = test_method(dx, net)
        xAD.append(xA_.detach().cpu())
        xBD.append(xB_.detach().cpu())
        xD.append(xD_.detach().cpu())

    (xAD, xBD, xD) = (torch.cat(x) for x in (xAD, xBD, xD))
    # AUC
    manual_classifier_auc(xAD, xBD, labels, L=200)
    # Probability of A vs B
    outADD = nn.Softmax(dim=1)(classifier((xAD - xD).cuda()).detach().cpu())
    return outADD


def test_gan():
    print('testing over the GAN set...')
    all_mask = []
    all_gx = []
    # test over the GAN
    for i in tqdm(range(len(dlist) // 23)):
        (ax, bx) = ([tiff.imread(x) for x in y[i * 23:(i + 1) * 23]] for y in (alist, blist))
        (ax, bx) = (list_to_tensor(x, repeat=True, norm=norm_method).float() for x in (ax, bx))
        gx, mask = get_xy(ax.permute(4, 1, 2, 3, 0)[:,:1,:,:,0])
        gx = gx.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        mask = mask.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
        all_gx.append(gx)
        all_mask.append(mask)

    xG = []
    for i in tqdm(range(len(dlist) // 23)):
        xG_ = test_method(gx, net)
        xG.append(xG_.detach().cpu())
        del xG_
    xG = torch.cat(xG)

    # Probability of A vs B
    manual_classifier_auc(xA[:xG.shape[0], :], xG, labels, L=200)
    outAGG = nn.Softmax(dim=1)(classifier((xA[:xG.shape[0], :] - xG).cuda()).detach().cpu())

    return outAGG


def test_gan_mc(mc_n=20):
    print('testing over the GAN set...')

    destination = '/media/ExtHDD01/Dataset/paired_images/womac4/full/gan022724/'

    # test over the GAN
    #for i in tqdm(range(len(dlist) // 23)):
    for i in tqdm(range(10)):
        (aname, bname) = (y[i * 23:(i + 1) * 23] for y in (alist, blist))
        (ax, bx) = (tiff.imread(x) for x in (aname, bname))
        (ax, bx) = (list_to_tensor(x, repeat=True, norm=norm_method).float() for x in (ax, bx))
        all_mask = []
        all_gx = []
        for mc in range(mc_n):
            gx, mask = get_xy(ax.permute(4, 1, 2, 3, 0)[:,:1,:,:,0])
            gx = gx.permute(1, 2, 3, 0).unsqueeze(0)
            mask = mask.permute(1, 2, 3, 0).unsqueeze(0)
            diff = ax - gx
            all_gx.append(gx)
            all_mask.append(diff)
        all_mask = torch.cat(all_mask)
        all_gx = torch.cat(all_gx)

        mask_mean = np.mean(all_mask.numpy(), 0)
        mask_var = np.var(all_mask.numpy(), 0)
        mask_sig = mask_mean / (mask_var + 1e-5)

        for z in range(len(aname)):
            slice_name = aname[z].split('/')[-1]
            tiff.imwrite(destination + 'meanA/' + slice_name, mask_mean[0, :, :, z])
            tiff.imwrite(destination + 'sigA/' + slice_name, mask_sig[0, :, :, z])

    return outAGG

parser = args_train()
args = parser.parse_args()

load_dotenv('env/.t09b')
x = pd.read_csv('../../env/womac4_moaks.csv')
labels = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values > (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values
labels = [int(x) for x in labels]
knee_painful = x.loc[(x['V$$WOMKP#'] > 0)].reset_index()
pmindex = knee_painful.loc[~knee_painful['READPRJ'].isna()].index.values
ID_has_eff = x.loc[~x['V$$MEFFWK'].isna()]['ID'].unique()
pmeffid = knee_painful.loc[knee_painful['ID'].isin(ID_has_eff)].index.values

train_index = knee_painful.loc[~knee_painful['ID'].isin(ID_has_eff)].index.values
test_index = pmeffid

train_label = [labels[i] for i in train_index]
test_label = [labels[i] for i in test_index]

# Extra Raw Data
# diffusion data
data_root = '/home/ghc/Dataset/paired_images/womac4/fullYYY/'
alist = sorted(glob.glob(data_root + 'a/*'))
blist = sorted(glob.glob(data_root + 'b/*'))
dlist = sorted(glob.glob(data_root + '003b/*'))

# Model
ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/MOAKSID/adv0/0/checkpoints/net_d*.pth'))[7]#[7]
net = torch.load(ckpt, map_location='cpu')
net = net.cuda()
net = net.eval()

classifier = torch.load(ckpt.replace('net_d', 'classifier'), map_location='cpu')
classifier = classifier.cuda()
classifier = classifier.eval()


try:
    projection = torch.load(ckpt.replace('net_d', 'projection'), map_location='cpu')
    projection = projection.eval()
except:
    print('no projection')


#ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/FIXING/gan0xy0con/conA/checkpoints/net_g*.pth'))[7]#[7]
#net_gan = torch.load(ckpt, map_location='cpu').cuda()
#alpha = 1


def get_xy(ax):
    mask = net_gan(ax.cuda(), alpha=alpha)['out0'].detach().cpu()
    mask = nn.Sigmoid()(mask)
    ax = torch.multiply(mask, ax.detach().cpu())
    return ax, mask


# NORMALIZATION

args.dataset_mode = "PairedSlices3D"
args.rotate = False
args.rgb = False
args.nm = '01'

# Data
train_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/train/',
                   path='ap_bp', opt=args, labels=[(int(x),) for x in train_label], mode='test', index=None)

eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
                   path='ap_bp', opt=args, labels=[(int(x),) for x in test_label], mode='test', index=None)

#from dataloader.data_multi import MultiData as Dataset
#eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
#                   path='a_b',
#                   opt=args, mode='test', index=test_index, filenames=True)

#eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# TESTING
if 0:
    outABt, xAt, xBt = testing_manual_dual(train_set)
    outABv, xAv, xBv = testing_manual_dual(eval_set)

    #outADD = test_diffusion()
    #outAGG = test_gan()
    #plt.scatter(outADD[:,1],outAGG[:,1]);plt.xlim(0,1);plt.ylim(0,1);plt.show()
    plot_umap(xAv, xBv)
    plot_umap(projection(xAv).detach(), projection(xBv).detach())

    # SVM
    if 0:
        X_train = torch.cat([xAt, xBt], 0).numpy()
        X_test = torch.cat([xAv, xBv], 0).numpy()
    else:
        #X_train = torch.cat([projection(xAt).detach(), projection(xBt).detach()], 0).numpy()
        #X_test = torch.cat([projection(xAv).detach(), projection(xBv).detach()], 0).numpy()
        X_train = projection(torch.cat([xAt, xBt], 0)).detach().numpy()
        X_test = projection(torch.cat([xAv, xBv], 0)).detach().numpy()

    y_train = np.array([0] * xAt.shape[0] + [1] * xBt.shape[0])
    y_test = np.array([0] * xAv.shape[0] + [1] * xBv.shape[0])
    # Create a SVM Classifier
    clf = svm.SVC(kernel='poly', probability=True)  # Linear Kernel
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