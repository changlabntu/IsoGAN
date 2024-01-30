import torch, os, glob
import torch.nn as nn
import numpy as np
import tifffile as tiff
import umap
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, metrics
import pandas as pd
## torch
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from utils.data_utils import imagesc


def classify(f):
    # this is for the knee (Z, C, X, Y), you may need to change it
    (B, C, X, Y) = f.shape
    batch = B // 23
    f = f.view(B // batch, batch, C, X, Y)
    f = f.permute(1, 2, 3, 4, 0)
    f = pool(f)[:, :, 0, 0, 0]
    return f


def get_features(alist, N):
    all = []
    for i in range(N):
        if (i % 100) == 0:
            print(i)
        ax = alist[i * 23: (i + 1) * 23]
        ax = [tiff.imread(x) for x in ax]
        ax = np.stack(ax, 0)
        ax = ax / ax.max()
        ax = torch.from_numpy(ax).float()
        ax = ax.unsqueeze(1)
        if gpu:
            ax = ax.cuda()
        ax = net(ax, alpha=1, method='encode')[-1].detach().cpu()
        ax = ax.permute(1, 2, 3, 0).unsqueeze(0)
        ax = pool(ax)[:, :, 0, 0, 0]
        #if (projection is not None) and not force_no_projection:
        #    ax = projection(ax).detach().cpu()
        all.append(ax)
        del ax
    all = torch.cat(all, 0)
    all = all.numpy()
    return all


# SVM Model
class LinearSVM(nn.Module):
    def __init__(self):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(x)


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.
    Parameters
    ----------
    x : 2D array (n,d)
      Samples from distribution P, which typically represents the true
      distribution.
    y : 2D array (m,d)
      Samples from distribution Q, which typically represents the approximate
      distribution.
    Returns
    -------
    out : float
      The estimated Kullback-Leibler divergence D(P||Q).
    References
    ----------
    Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
    """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert (d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.))


reducer = umap.UMAP()
gpu = True

# data_root = '/workspace/Data/womac4/full/'
# Change to your data root
data_root = '/home/ghc/Dataset/paired_images/womac4/full/'
# Change to your log root
# log_root = '/run/user/1000/gvfs/smb-share:server=changlab-nas.local,share=data/Data_GHC/OAI/contrastive_checkpoints/'
log_root = '/media/ExtHDD01/logs/womac4/'
# epoch
n_epoch = 200

# %%

df = pd.read_csv('env/csv/womac4_moaks.csv')
df = df.loc[df['V$$WOMKP#'] > 0, :]
df.reset_index(inplace=True)

## Models
## OPTION 1
# prj_name = '/global/1_project128/' # 0.90
# prj_name = '/global/1_project256_cosine/' #0.84
# prj_name = 'global1_cut1/nce4_down2_0011_ngf24_proj128/'
# prj_name = 'global1_cut1/nce4_down2_0011_ngf32_proj128_zcrop16/' # 0.88
# prj_name = 'global1_cut1/nce4_down2_0011_ngf24_proj128/' # 0.81
#prj_name = 'global1_cut1/nce4_down2_0011_ngf32_proj128_zcrop16_meanpool/' # bad
#prj_name = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16/' # 0.92
#prj_name = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16_moaks/' # 0.936
# prj_name = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16_unpaired/' # 0.857
# prj_name = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16_unpaired_nce0/' # 0.857
# prj_name = 'global1_cut1/nce4_down4_0011_ngf32_proj32_zcrop16_unpaired_moaks/'  # 0.914
#prj_name = 'global1_cut1/nce0_down4_0011_ngf32_proj32_zcrop16_unpaired/'  # 0.846

#prj_name = 'global1_cutx3/nce0_down4_0011_ngf32_proj32_zcrop16_unpaired_moaks/'

#prj_name = 'global1_cut4/b_alpha0/'  #
prj_name = 'global1_cut1x/2/'  #

force_no_projection = False  # force to not use projection, or it will be used if .projection is in a model
use_eval = False
pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))

net = torch.load(log_root + prj_name + 'checkpoints/net_g_model_epoch_' + str(n_epoch) + '.pth', map_location='cpu')

if gpu:
    net = net.cuda()
pool = nn.AdaptiveMaxPool3d(output_size=(1, 1, 1))
#pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

try:
    projection = net.projection.cpu()
    print('With projection')
except:
    projection = None
    print('No projection')

try:
    classifier = net.classifier.cpu()
    print('With classifier')
except:
    classifier = None
    print('No classifier')

if use_eval:
    net = net.eval()
    #net.projection = net.projection.eval()
else:
    net = net.train()
    #net.projection = net.projection.train()

# %%
selected = 2225*23
alist = sorted(glob.glob(data_root + 'a/*'))[:selected]
blist = sorted(glob.glob(data_root + 'b/*'))[:selected]
clist = sorted(glob.glob(data_root + 'cf3/*'))

# get features
af = get_features(alist, N=len(alist) // 23)
bf = get_features(blist, N=len(blist) // 23)
all = np.concatenate([af, bf], 0)

# indicies and labels
L = (all.shape[0] // 2)
train_index = list((df.loc[:(L-1), :]).loc[df['READPRJ'].isnull(), :].index)[:]
test_index = list((df.loc[:(L-1), :]).loc[df['READPRJ'].notnull(), :].index)[:]
#train_index = list(range(667, 2225))
#test_index = list(range(667))
train_index = train_index + [L + x for x in train_index]#[2 * x for x in train_index] + [2 * x + 1 for x in train_index]
test_index = test_index + [L + x for x in test_index]#[2 * x for x in test_index] + [2 * x + 1 for x in test_index]
labels = np.array([0] * L + [1] * L)

# Split dataset into training and testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

if projection is not None and not force_no_projection:
    contra = projection(torch.from_numpy(all).float()).detach().numpy()
else:
    contra = all
X_train = contra[train_index, :]
y_train = labels[train_index]
X_test = contra[test_index, :]
y_test = labels[test_index]

# Create a SVM Classifier
clf = svm.SVC(kernel='poly', probability=True)  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Evaluate accuracy
print("Acc Contra:", accuracy_score(y_test, y_pred))
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Compute AUC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC Contra:", auc_score)

cf3 = get_features(clist, N=50)
cf3proj = projection(torch.from_numpy(cf3).float()).detach().numpy()
# C test
c_pred = clf.predict_proba(cf3proj)[:,1]
a_pred = clf.predict_proba(contra[:contra.shape[0]//2, :])[:,1]
b_pred = clf.predict_proba(contra[contra.shape[0]//2:, :])[:,1]

print('b_pred - a_pred')
print((b_pred-a_pred).mean())
print('(c_pred-a_pred[:50])')
print((c_pred-a_pred[:50]).mean())
print('(b_pred-a_pred[:50])')
print((b_pred[:50]-a_pred[:50]).mean())
print('flip rate :50')
print((c_pred > 0.5).mean())

#  classfication
try:
    print('classfication')
    test_cls = nn.Softmax(dim=1)(classifier((torch.from_numpy(all[test_index, :])))).detach().numpy()
    auc_score = roc_auc_score(y_test, test_cls[:, 1])
    print("Test AUC Score:", auc_score)
    c_cls = nn.Softmax(dim=1)(classifier((torch.from_numpy(cf3)))).detach().numpy()
except:
    print('no classfication')

# umap
e0 = reducer.fit_transform(contra)
e = e0#[:23*5*2, :]
plt.scatter(e[:L, 0], e[:L, 1], s=0.5*np.ones(e.shape[0] // 2))
plt.scatter(e[L:, 0], e[L:, 1], s=0.5*np.ones(e.shape[0] // 2))
plt.show()


if 0:
    def get_features2(alist, N):
        all = []
        for i in range(N):
            ax = alist
            ax = [tiff.imread(x) for x in ax]
            ax = np.stack(ax, 0)
            ax = ax / ax.max()
            ax = torch.from_numpy(ax).float()
            ax = ax.unsqueeze(1)
            if gpu:
                ax = ax.cuda()
            ax = net(ax, alpha=1, method='encode')[-1].detach().cpu()
            ax = ax.permute(1, 2, 3, 0).unsqueeze(0)
            ax = pool(ax)[:, :, 0, 0, 0]
            if (projection is not None) and (not force_no_projection):
                ax = projection(ax).detach().cpu()
            all.append(ax)
            del ax
        all = torch.cat(all, 0)
        all = all.numpy()
        return all


    def get_features3(ax, N):
        all = []
        for i in range(N):
            if gpu:
                ax = ax.cuda()
            ax = net(ax, alpha=1, method='encode')[-1].detach().cpu()
            ax = ax.permute(1, 2, 3, 0).unsqueeze(0)
            ax = pool(ax)[:, :, 0, 0, 0]
            if (projection is not None) and (not force_no_projection):
                ax = projection(ax).detach().cpu()
            all.append(ax)
            del ax
        all = torch.cat(all, 0)
        all = all.numpy()
        return all


    def get_image(ax):
        ax = [tiff.imread(x) for x in ax]
        ax = np.stack(ax, 0)
        ax = ax / ax.max()
        ax = torch.from_numpy(ax).float()
        ax = ax.unsqueeze(1)
        if gpu:
            ax = ax.cuda()
        mask = net(ax, alpha=1)['out0'].detach().cpu()
        ax = torch.multiply(nn.Sigmoid()(mask), ax.detach().cpu())
        return ax

    # misc
    cfall = [get_features2([x],1) for x in blist[23:46]]
    cp2 = [clf.predict_proba(x)[:,1] for x in cfall]
    print(np.array(cp2).mean())

    cfall = [get_features2(clist[23:46],1)]
    cp2 = [clf.predict_proba(x)[:,1] for x in cfall]
    print(cp2)

# get image
img_all = []
for i in range(50):
    print(i)
    imga = get_image(alist[i * 23:(i + 1) * 23])
    imgb = get_image(blist[i * 23:(i + 1) * 23])
    img = torch.cat([imga, imgb], 0)
    imgz = net(img.cuda(), alpha=1, method='encode')[-1].detach().cpu()
    imgz = classify(imgz)
    imgzproj = projection(imgz)
    img_all.append(imgzproj)

img_all = torch.cat(img_all, 0)

zzz = img_all.detach().numpy()

# umap
e0 = reducer.fit_transform(np.concatenate([contra, zzz], 0))
e = e0#[:23*5*2, :]
#plt.scatter(e[:L, 0], e[:L, 1], s=0.5*np.ones(L))
plt.scatter(e[L:2*L, 0], e[L:2*L, 1], s=0.5*np.ones(L))
plt.scatter(e[2*L:, 0], e[2*L:, 1], s=2*np.ones(zzz.shape[0]))
plt.show()


