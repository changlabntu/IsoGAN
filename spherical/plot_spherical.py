import tifffile as tiff
import os, glob
from utils.data_utils import imagesc
import numpy as np
from tqdm import tqdm


def image_to_list_of_values_by_interval(x):
    intervals = range(0, 360, 5)
    values = []
    for i in intervals:
        values.append(((x > i) & (x <= i + 5)).sum())
    return values


root = '/media/ExtHDD01/Dataset/paired_images/womac4/full/seg/'
slices = sorted(glob.glob(root + 'beffphi0/*.tif'))

#subjects = [x.strip(x.split('_')[-1]) for x in slices]
#subjects = sorted(list(set([x.split('/')[-1] for x in subjects])))
#subjects = [x[:-1] for x in subjects]

subjects = sorted(list(set([x.strip(x.split('_')[-1]) for x in slices])))

#total = np.zeros((23, 36))
#for s in tqdm(range(len(slices))):
total = []
for s in tqdm(range(len(subjects[:]))):
    stacks = sorted(glob.glob(subjects[s] + '*.tif'))
    one_subject = []
    for i in range(len(stacks)):
        name = stacks[i]
        x = tiff.imread(name)
        z = int(name.split('_')[-1].split('.')[0])
        v = image_to_list_of_values_by_interval(x)
        one_subject.append(v)
    one_subject = np.array(one_subject)
    #imagesc(one_subject, show=False, save='spherical/outimg/' + subjects[s].split('/')[-1][:-1] +'.png')
    tiff.imwrite('spherical/outimg2/' + subjects[s].split('/')[-1][:-1] +'.tif', ((one_subject)/1000).astype(np.float32))

    total.append(one_subject)

total = np.array(total)
imagesc(total.mean(0))

