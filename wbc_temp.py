import glob, os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

matplotlib.colors.rgb_to_hsv(arr)
# np.stack([matplotlib.colors.rgb_to_hsv(oriall[i, :, :, :]) for i in range(oriall.shape[0])], 0)

def read_all(path):
    xall = []
    for p in path:
        x = tiff.imread(p)
        xall.append(x)
    xall = np.stack(xall, 0)
    return xall

def save_as_hsv(path, destination):
    for p in path:
        x = tiff.imread(p)
        x = matplotlib.colors.rgb_to_hsv(x)
        tiff.imsave(destination + os.path.basename(p), x)

root = '/media/ExtHDD01/wbc_diffusion/'

ori = sorted(glob.glob(root + 'myelo/*'))
seg = sorted(glob.glob(root + 'cyc_seg/*'))

oriall = read_all(ori)
segall = read_all(seg)


for i in range(len(ori)):
    o = tiff.imread(ori[i])
    s = tiff.imread(seg[i])

    o0 = o[:, :, 0]
    o1 = o[:, :, 1]
    o2 = o[:, :, 2]

    s0 = s[:, :, 0]
    s1 = s[:, :, 1]
    s2 = s[:, :, 2]

    s0 = s0 - s0.mean()
    s1 = s1 - s1.mean()
    s2 = s2 - s2.mean()
    s0 = s0 / s0.std()
    s1 = s1 / s1.std()
    s2 = s2 / s2.std()

    s0 = s0 * o0.std()
    s1 = s1 * o1.std()
    s2 = s2 * o2.std()

    s0 = s0 + o0.mean()
    s1 = s1 + o1.mean()
    s2 = s2 + o2.mean()

    s_new = np.stack([s0, s1, s2], 2)
    s_new[s_new < 0] = 0
    s_new[s_new > 255] = 255
    tiff.imsave(root + 'cyc_seg_new/' + os.path.basename(seg[i]), s_new.astype(np.uint8))
