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
metrics = GetAUC()


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


load_dotenv('env/.t09b')
parser = args_train()
args = parser.parse_args()

args.dataset_mode = "PairedSlices3D"
args.rotate = False
args.rgb = False
args.nm = '01'
eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
                   path='ap%bp', opt=args, labels=None, mode='test', index=None)


def plot_umap(fA, fB, shift=0):
    import umap
    import matplotlib.pyplot as plt
    reducer = umap.UMAP()
    e = reducer.fit_transform(torch.cat([fA, fB], 0).squeeze())
    L = fA.shape[0]
    plt.figure(figsize=(10, 8))
    plt.scatter(e[:L, 0], e[:L, 1], s=2 * np.ones(L))
    plt.scatter(e[L:L*2, 0], e[L:L*2, 1], s=2 * np.ones(L))
    #plt.scatter(e[667:2225, 0], e[667:2225, 1], s=0.5 * np.ones(1558))
    #plt.scatter(e[L+667:L + 2225, 0], e[L+667:L + 2225, 1], s=0.5 * np.ones(1558))
    plt.show()


def test_method(x, net):
    x = net(x.permute(4, 1, 2, 3, 0).squeeze()[:, :1, :, :].cuda())[-1]
    x = x.permute(1, 2, 3, 0).unsqueeze(0)
    x = torch.mean(x, dim=(2, 3))
    x, _ = torch.max(x, 2)
    return x


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


def list_to_tensor(ax, repeat=True, norm=None):
    ax = torch.cat([torch.from_numpy(x / 1).unsqueeze(2) for x in ax], 2).unsqueeze(0).unsqueeze(1)
    # normalize ax to -1 to 1
    ax = normalize(ax, norm=norm)
    if repeat:
        ax = ax.repeat(1, 3, 1, 1, 1)
    return ax


def flip_features(fx, fy, labels):
    fA = []
    fB = []
    for i in range(len(labels)):
        l = labels[i]
        if l == 0:
            fA.append(fx[i, :])
            fB.append(fy[i, :])
        else:
            fA.append(fy[i, :])
            fB.append(fx[i, :])
    fA = torch.stack(fA, 0)
    fB = torch.stack(fB, 0)
    return fA, fB


def test_manual_load(data_root, net, classifier, norm_method='01'):
    alist = sorted(glob.glob(data_root + 'ap/*'))
    blist = sorted(glob.glob(data_root + 'bp/*'))
    outAB = []
    xA = []
    xB = []
    norm_method = '01'
    for i in tqdm(range(len(alist) // 23)):

        (ax, bx) = ([tiff.imread(x) for x in y[i * 23:(i + 1) * 23]] for y in (alist, blist))
        (a, b) = (list_to_tensor(x, repeat=True, norm=norm_method).float() for x in (ax, bx))

        dx = 64
        a = a[:, :, dx:-dx, dx:-dx, 4:20]
        b = b[:, :, dx:-dx, dx:-dx, 4:20]

        xA_ = test_method(a, net)
        xB_ = test_method(b, net)

        xA.append(xA_.detach().cpu())
        xB.append(xB_.detach().cpu())

    (xA, xB) = (torch.cat(x) for x in (xA, xB))
    return xA, xB


def test_auc(xA, xB, classifier):
    xLabel = np.random.randint(0, 2, xA.shape[0])
    x0, x1 = flip_features(xA, xB, xLabel)

    # TEST AUC
    out = []
    for i in range(x0.shape[0]):
        out.append(classifier((x1[i, :] - x0[i, :]).unsqueeze(0).cuda()).detach().cpu())
    out = torch.cat(out)
    print('AUC0=  ' + str(metrics(1 - xLabel, out)[0]))

    out = []
    for i in range(x0.shape[0]):
        out.append(classifier((x0[i, :] - x1[i, :]).unsqueeze(0).cuda()).detach().cpu())
    out = torch.cat(out)
    print('AUC1=  ' + str(metrics(xLabel, out)[0]))


def svm_single_knee_classification(xAt, xBt, xAv, xBv, use_projection):
    if not use_projection:
        X_train = torch.cat([xAt, xBt], 0).numpy()
        X_test = torch.cat([xAv, xBv], 0).numpy()
    else:
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

# Model
ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/MOAKSID/contra00/00Bavgy_011/checkpoints/net_d*.pth'))[4]#[7]
net = torch.load(ckpt, map_location='cpu')
net = net.cuda()
net = net.eval()

classifier = torch.load(ckpt.replace('net_d', 'classifier'), map_location='cpu')
classifier = classifier.cuda()
classifier = classifier.eval()

projection = torch.load(ckpt.replace('net_d', 'projection'), map_location='cpu')
projection = projection.eval()

train_root = '/home/ghc/Dataset/paired_images/womac4/train/'
xAt, xBt = test_manual_load(train_root, net, classifier, norm_method='01')
val_root = '/home/ghc/Dataset/paired_images/womac4/val/'
xAv, xBv = test_manual_load(val_root, net, classifier, norm_method='01')
test_auc(xAv, xBv, classifier)

# outADD = test_diffusion()
# outAGG = test_gan()
# plt.scatter(outADD[:,1],outAGG[:,1]);plt.xlim(0,1);plt.ylim(0,1);plt.show()
plot_umap(xAv, xBv)
plot_umap(projection(xAv).detach(), projection(xBv).detach())
svm_single_knee_classification(xAt, xBt, xAv, xBv, use_projection=True)


mm = xBt.mean(0).unsqueeze(0)

X_train = projection(torch.cat([xAt-mm, xBt-mm], 0)).detach().numpy()
X_test = projection(torch.cat([xAv-mm, xBv-mm], 0)).detach().numpy()

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