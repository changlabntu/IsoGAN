#python train.py --jsn womac4 --prj TESTING/clsfix0 --models siamese_cls_0 --netG edalphand2 --dataset womac4 --lbvgg 0 --lbNCE 0 --nm 01 --fDown 4 -b 4 --ngf 32 --projection 2 --n_epochs 200 --lr_policy cosine --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj TESTING/fix0 --models siamese_gan_0 --netG edalphand2  --dataset womac4 --lbvgg 0 --lbNCE 0 --nm 01 --fDown 4  -b 4 --ngf 32 --projection 2  --lr_policy cosine --direction ap_bp --alpha 1 --env runpod
#python train.py --jsn womac4 --prj TESTING/00 --models lesion_siamese_gan --netG edalphand2 --dataset womac4 --lbvgg 0 --lbNCE 0 --nm 01 --fDown 4 -b 4 --ngf 32 --projection 2 --lr_policy cosine --direction ap_bp --alpha 1 --env runpod


#python train.py --jsn womac4 --prj FIXING/siamese0/b1bt1 --models siamese0 -b 1 --bt 1 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj FIXING/siamese0mean/b4bt1 --models siamese0mean -b 4 --bt 1 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj FIXING/siamese0mean/b1bt1 --models siamese0mean -b 1 --bt 1 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj FIXING/siamese0mean/b4bt4 --models siamese0mean -b 4 --bt 4 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod

# CUDA_VISIBLE_DEVICES=0

python train.py --jsn womac4 --prj MOAKS/contra/0 --models contra0 -b 2 --bt 2 --netG edalphand2 --dataset womac4 --nm 01 --ngf 32 --projection 32 --n_epochs 200  --direction ap_bp --save_d --alpha 1 --env runpod --lbcls 1
python train.py --jsn womac4 --prj MOAKS/contra/A --models contraA -b 2 --bt 2 --netG edalphand2 --dataset womac4 --nm 01 --ngf 32 --projection 32 --n_epochs 200  --direction ap_bp --save_d --alpha 1 --env runpod --lbcls 1
python train.py --jsn womac4 --prj MOAKS/contra/B --models contraB -b 2 --bt 2 --netG edalphand2 --dataset womac4 --nm 01 --ngf 32 --projection 32 --n_epochs 200  --direction ap_bp --save_d --alpha 1 --env runpod --lbcls 1