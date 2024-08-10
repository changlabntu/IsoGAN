import torch
import torchmetrics
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from PIL import Image
import os
import tifffile as tiff


def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path))[:1024]:
        if filename.endswith((".tif")):
            img_path = os.path.join(folder_path, filename)
            img = tiff.imread(img_path)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
            images.append(img_tensor)
    images = torch.stack(images, 0)
    images = (images + 1) / 2
    images = (images * 255).type(torch.uint8)
    images = images.repeat(1, 3, 1, 1)
    return images.cuda()


def calculate_metrics(option, folder1, folder2, subset_size=50):
    # Set seed for reproducibility
    torch.manual_seed(123)

    # Initialize metrics
    if option == 'kid':
        metrics = KernelInceptionDistance(subset_size=subset_size).cuda()
    elif option == 'fid':
        metrics = FrechetInceptionDistance().cuda()
    elif option == 'lpips':
        metrics = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').cuda()

    # Load images
    imgs_dist1 = load_images_from_folder(folder1)[:, :, :, :]
    imgs_dist2 = load_images_from_folder(folder2)[:, :, :, :]

    if option in ['kid', 'fid']:
        # Update metrics with images from both folders
        metrics.update(imgs_dist1, real=True)
        metrics.update(imgs_dist2, real=False)
        # Compute metrics
        try:
            metric_mean, metric_std = metrics.compute()
            print(f": {metric_mean:.4f} ± {metric_std:.4f}")
        except:
            metric_mean = metrics.compute()
            print(f": {metric_mean:.4f}")
    else:
        print(metrics(imgs_dist1, imgs_dist2))


# All: 353: 216
# Usage
folder1 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/xya2d"
#folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/original/expanded3d/zxa2d"
folder2 = "/media/ExtHDD01/oai_diffusion_interpolated/redounet/expanded3d/zya3d"
calculate_metrics('fid', folder1, folder2)
#print(f"metrics: {metrics_mean:.4f} ± {metrics_std:.4f}")