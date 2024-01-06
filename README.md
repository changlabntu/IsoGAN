### Repository of our paper "Isotropic Multi-Scale Neuronal Reconstruction from High-Ratio Expansion Microscopy with Contrastive Unsupervised Deep Generative Models", currently under review

Documentations and testting data are under construction.

### USAGE
Training of IsoGAN:
```bash
python train.py --jsn wnwp3d --prj prj_name --models IsoGAN -b 16 --direction zyft0_zyori%xyft0_xyori --trd 2000 --nm 11 --netG edescarnoumc --split all --env t09b --dataset_mode PairedSlices --use_mlp
```
Training of CycleGAN:
```bash
python train.py --jsn wnwp3d --prj prj_name --models cyc4 -b 16 --direction zyft0_zyori%xyft0_xyori --trd 2000 --nm 11 --netG edescarnoumc --split all --env t09b --dataset_mode PairedSlices --use_mlp
```