import nibabel as nib
import numpy as np
import glob, os
import tifffile as tiff

root = '/run/user/1000/gvfs/smb-share:server=changlab-nas.local,share=data/evachen/Brain_Dataset/*/*/'
destination = '/media/ExtHDD01/Dataset/paired_images/BraTSReg/train/t1norm2/'

list_nii = sorted(glob.glob(root + '*/*t1.nii.gz'))

for x in list_nii:
    print(x)
    nii_img = nib.load(x)
    nii_data = nii_img.get_fdata()
    nii_data = np.transpose(nii_data, (2, 0, 1))

    nii_data = nii_data / 1

    # normalize by mean and std
    #nii_data = (nii_data - nii_data.mean()) / nii_data.std()

    # normalize to -1 to 1
    nii_data = (nii_data - nii_data.min()) / (nii_data.max() - nii_data.min())
    nii_data = (nii_data - 0.5) * 2

    #cropping

    filename = x.split('/')[-1].split('.')[0].replace('_', '')

    nii_data = nii_data.astype(np.float32)

    for z in range(nii_data.shape[0]):
        tiff.imwrite(destination + filename + '_' + str(z).zfill(3) + '.tif', nii_data[z, :, :])

    #dcm = tiff.imwrite(destination + filename + '.tif', nii_data)

