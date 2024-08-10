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


# pain valuesa
def testing_pain_values():  ## this is so not going to work
    train_index, test_index, train_label, test_label, pain_val = moaks_index()

    Av_mean = xAv.detach().mean(0).unsqueeze(0)
    Bv_mean = xBv.detach().mean(0).unsqueeze(0)
    Av = xAv  # projection(xAv).detach()
    Bv = xBv  # projection(xBv).detach()
    # Av_Bt_distance = torch.norm(Av - Bv, dim=1).detach().cpu().numpy()
    # Av_Bt_distance = torch.norm(classifier(Av - Bv), dim=1).detach().cpu().numpy()
    probA = nn.Softmax(dim=1)(classifier(Av - Bv))[:, 1].detach().cpu().numpy()
    probB = nn.Softmax(dim=1)(classifier(Av - Bv))[:, 0].detach().cpu().numpy()

    to_draw = probA - probB

    plt.scatter(pain_val[test_index] + np.random.normal(0, 0.1, len(test_index)), to_draw);
    plt.show()
    # plt.scatter(pain_val[test_index] + np.random.normal(0, 0.1, len(test_index)), nn.Softmax(dim=1)(out)[:,1]); plt.show()


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


def test_method(x, net, max=True):
    x = net(x.permute(4, 1, 2, 3, 0).squeeze()[:, :1, :, :].cuda())[-1]
    x = x.permute(1, 2, 3, 0).unsqueeze(0)
    x = torch.mean(x, dim=(2, 3))
    if max == True:
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


def test_manual_load(data_root, net, classifier, norm_method='01', max=True):
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
        if dx > 0:
            a = a[:, :, dx:-dx, dx:-dx, 4:20]
            b = b[:, :, dx:-dx, dx:-dx, 4:20]

        xA_ = test_method(a, net, max=max)
        xB_ = test_method(b, net, max=max)
        xA.append(xA_.detach().cpu())
        xB.append(xB_.detach().cpu())

    if max == True:
        (xA, xB) = (torch.cat(x) for x in (xA, xB))
    return xA, xB


def test_by_dataset(data_set, net, max=True):
    xA = []
    xB = []
    for i in tqdm(range(0, len(data_set))):
        batch = data_set.__getitem__(i)
        a, b = batch['img']  # (1, H, W, Z)
        (a, b) = (x.permute(3, 0, 1, 2).cuda() for x in (a, b))  # (Z, 1, H, W)
        dx = 64
        if dx > 0:
            a = a[:, :, dx:-dx, dx:-dx]
            b = b[:, :, dx:-dx, dx:-dx]

        (a, b) = (x.permute(1, 2, 3, 0).unsqueeze(0).repeat(1, 3, 1, 1, 1) for x in (a, b))  # (1, 3, H, W, Z)

        xA_ = test_method(a, net, max=max)
        xB_ = test_method(b, net, max=max)
        xA.append(xA_.detach().cpu())
        xB.append(xB_.detach().cpu())
    if max:
        (xA, xB) = (torch.cat(x) for x in (xA, xB))
    return xA, xB


def test_auc(xA, xB, classifier):
    xLabel = np.random.randint(0, 2, xA.shape[0])
    #xLabel = np.ones(xA.shape[0])
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


def shuffle_tensor(x):
    idx = torch.randperm(x.shape[0])
    return x[idx, ...]


def classifier_unpaired_by_compare_to_mean():
    # SINGLE SIDE TEST AUC
    xBmean = xBt.mean(0).unsqueeze(0) * 1
    xA = xAv
    xB = xBv
    xLabel = np.random.randint(0, 2, xA.shape[0])
    x0, x1 = flip_features(xA - xBmean, xB - xBmean, xLabel)

    out = []
    for i in range(x0.shape[0]):
        out.append(classifier((x1[i, :]).unsqueeze(0).cuda()).detach().cpu())
    out = torch.cat(out)
    print('AUC0=  ' + str(metrics(1 - xLabel, out)[0]))

    out = []
    for i in range(x0.shape[0]):
        out.append(classifier((x0[i, :]).unsqueeze(0).cuda()).detach().cpu())
    out = torch.cat(out)
    print('AUC1=  ' + str(metrics(xLabel, out)[0]))


def classifier_unpaired_by_compare_to_all_the_cases():
    # SINGLE SIDE TEST AUC
    if 0:
        out0 = []
        for i in range(xAv.shape[0]):
            a_out = []
            b_out = []
            for j in range(xAt.shape[0]):
                a_out.append(classifier((xBv[i, :] - xBt[j, :]).unsqueeze(0).cuda()).detach().cpu())
                b_out.append(classifier((xBv[i, :] - xAt[j, :]).unsqueeze(0).cuda()).detach().cpu())
            a_out = torch.cat(a_out)
            a_out = nn.Softmax(dim=1)(a_out).detach().cpu()
            #a_out = nn.Softmax(dim=1)(a_out).detach().cpu()
            a_out = a_out.mean(0).unsqueeze(0)
            b_out = torch.cat(b_out)
            b_out = nn.Softmax(dim=1)(b_out).detach().cpu()
            #b_out = nn.Softmax(dim=1)(b_out).detach().cpu()
            b_out = b_out.mean(0).unsqueeze(0)
            if a_out.max() > b_out.max():
                out0.append(a_out)
            else:
                out0.append(torch.flip(b_out, [0]))
        out0 = torch.cat(out0, 0)

        # SINGLE SIDE TEST AUC
        out00 = []
        for i in range(xAv.shape[0]):
            a_out = []
            b_out = []
            for j in range(xAt.shape[0]):
                a_out.append(classifier((xAv[i, :] - xBt[j, :]).unsqueeze(0).cuda()).detach().cpu())
                b_out.append(classifier((xAv[i, :] - xAt[j, :]).unsqueeze(0).cuda()).detach().cpu())
            a_out = torch.cat(a_out)
            a_out = nn.Softmax(dim=1)(a_out).detach().cpu()
            #a_out = nn.Softmax(dim=1)(a_out).detach().cpu()
            a_out = a_out.mean(0).unsqueeze(0)
            b_out = torch.cat(b_out)
            b_out = nn.Softmax(dim=1)(b_out).detach().cpu()
            #b_out = nn.Softmax(dim=1)(b_out).detach().cpu()
            b_out = b_out.mean(0).unsqueeze(0)
            if a_out.max() > b_out.max():
                out00.append(a_out)
            else:
                out00.append(torch.flip(b_out, [0]))
        out00 = torch.cat(out00, 0)

        metrics(torch.cat([torch.zeros(xAv.shape[0]), torch.ones(xAv.shape[0])]), torch.cat([out00, out0]))

    out2 = []
    for i in range(xAv.shape[0]):
        a_out = []
        b_out = []
        for j in range(xAt.shape[0]):
            a_out.append(classifier((xBv[i, :] - xAt[j, :]).unsqueeze(0).cuda()).detach().cpu())
            #b_out.append(classifier((xB[i, :] - xA[j, :]).unsqueeze(0).cuda()).detach().cpu())
        a_out = torch.cat(a_out)
        #a_out = nn.Softmax(dim=1)(a_out)[:, 1].detach().cpu()
        a_out = nn.Softmax(dim=1)(a_out).detach().cpu()
        a_out = a_out.mean(0).unsqueeze(0)
        out2.append(a_out)
    out2 = torch.cat(out2, 0)

    out1 = []
    for i in range(xAv.shape[0]):
        a_out = []
        b_out = []
        for j in range(xAt.shape[0]):
            a_out.append(classifier((xAv[i, :] - xAt[j, :]).unsqueeze(0).cuda()).detach().cpu())
            #b_out.append(classifier((xB[i, :] - xA[j, :]).unsqueeze(0).cuda()).detach().cpu())
        a_out = torch.cat(a_out)
        #a_out = nn.Softmax(dim=1)(a_out)[:, 1].detach().cpu()
        a_out = nn.Softmax(dim=1)(a_out).detach().cpu()
        a_out = a_out.mean(0).unsqueeze(0)
        out1.append(a_out)
    out1 = torch.cat(out1, 0)

    print(metrics(torch.cat([torch.zeros(xAv.shape[0]), torch.ones(xAv.shape[0])]), torch.cat([out1, out2])))


def svm_by_difference_to_shuffled_not_useful():
    mt = xBt.mean(0).unsqueeze(0) * 1
    mv = xBv.mean(0).unsqueeze(0) * 1

    #X_train = (torch.cat([xAt - mv, xBt - mv], 0)).detach().numpy()
    #X_test = (torch.cat([xAv - mv, xBv - mv], 0)).detach().numpy()

    # X_train = projection(torch.cat([xAt-mm, xBt-mm], 0)).detach().numpy()
    # X_test = projection(torch.cat([xAv-mm, xBv-mm], 0)).detach().numpy()
    # X_test = projection(torch.cat([xAv-mm, xBv-mm], 0)).detach().numpy()

    X_train = (torch.cat([xAt-shuffle_tensor(xBt), xBt-shuffle_tensor(xAt)], 0)).detach().numpy()
    X_test = (torch.cat([xAv-shuffle_tensor(xBv), xBv-shuffle_tensor(xAv)], 0)).detach().numpy()

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


def moaks_index():
    load_dotenv('env/.t09b')
    x = pd.read_csv('../../env/womac4_moaks.csv')
    pain_right = (x.loc[x['SIDE'] == 'RIGHT']['V$$WOMKP#']).values
    pain_left = (x.loc[x['SIDE'] == 'LEFT']['V$$WOMKP#']).values
    pain_val = np.maximum(pain_right, pain_left)
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
    return train_index, test_index, train_label, test_label, pain_val


if __name__ == '__main__':
    # Model
    #ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/MOAKSID/adv1/00/checkpoints/net_d*.pth'))[7]#[7]
    ckpt = sorted(glob.glob('/media/ExtHDD01/logs/womac4/MOAKSID/adv1/01b/checkpoints/net_d*.pth'))[7]
    print(ckpt)
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
        print('No projection')

    train_root = '/home/ghc/Dataset/paired_images/womac4/train/'
    xAt, xBt = test_manual_load(train_root, net, classifier, norm_method='01')
    val_root = '/home/ghc/Dataset/paired_images/womac4/val/'
    xAv, xBv = test_manual_load(val_root, net, classifier, norm_method='01')
    test_auc(xAv, xBv, classifier)

    # manual classifier

    # outADD = test_diffusion()
    # outAGG = test_gan()
    # plt.scatter(outADD[:,1],outAGG[:,1]);plt.xlim(0,1);plt.ylim(0,1);plt.show()
    plot_umap(xAv, xBv)
    plot_umap(projection(xAv).detach(), projection(xBv).detach())
    try:
        svm_single_knee_classification(xAt, xBt, xAv, xBv, use_projection=True)
    except:
        svm_single_knee_classification(xAt, xBt, xAv, xBv, use_projection=False)


    #classifier_unpaired_by_compare_to_all_the_cases()
    svm_by_difference_to_shuffled_not_useful()


    xA = 1 * xAv
    xB = 1 * xBv
    xA = shuffle_tensor(xA)
    xB = shuffle_tensor(xB)
    xLabel = np.random.randint(0, 2, xA.shape[0])
    #xLabel = np.ones(xA.shape[0])
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

# /media/ExtHDD01/Dataset/paired_images/womac4/full/gan_results

if 0:
    import tifffile as tiff
    import glob

    l = sorted(glob.glob("/media/ExtHDD01/Dataset/paired_images/womac4/full/gan_results/meanA/*.tif"))

    for i in range(len(l)):
        x = tiff.imread(l[i])
        x = (x * 255).astype('uint8')
        tiff.imwrite(l[i].replace('meanA', 'meanA8bit'), x)
