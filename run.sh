#python train.py --jsn womac4 --prj TESTING/clsfix0 --models siamese_cls_0 --netG edalphand2 --dataset womac4 --lbvgg 0 --lbNCE 0 --nm 01 --fDown 4 -b 4 --ngf 32 --projection 2 --n_epochs 200 --lr_policy cosine --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj TESTING/fix0 --models siamese_gan_0 --netG edalphand2  --dataset womac4 --lbvgg 0 --lbNCE 0 --nm 01 --fDown 4  -b 4 --ngf 32 --projection 2  --lr_policy cosine --direction ap_bp --alpha 1 --env runpod
#python train.py --jsn womac4 --prj TESTING/00 --models lesion_siamese_gan --netG edalphand2 --dataset womac4 --lbvgg 0 --lbNCE 0 --nm 01 --fDown 4 -b 4 --ngf 32 --projection 2 --lr_policy cosine --direction ap_bp --alpha 1 --env runpod


#python train.py --jsn womac4 --prj FIXING/siamese0/b1bt1 --models siamese0 -b 1 --bt 1 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj FIXING/siamese0mean/b4bt1 --models siamese0mean -b 4 --bt 1 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj FIXING/siamese0mean/b1bt1 --models siamese0mean -b 1 --bt 1 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod
#python train.py --jsn womac4 --prj FIXING/siamese0mean/b4bt4 --models siamese0mean -b 4 --bt 4 --netG edalphand2 --dataset womac4 --nm 01   --ngf 32 --projection 2 --n_epochs 100  --direction ap_bp --save_d --alpha 1 --env runpod

# CUDA_VISIBLE_DEVICES=0

#python train.py --jsn womac4 --prj MOAKSID/adv0/0catA --models contra0catA -b 2 --bt 2 --netG edalphand2 --dataset womac4 --nm 01 --ngf 32 --projection 32 --n_epochs 200  --direction ap_bp --save_d --alpha 1 --env runpod --lbcls 1 --adv 0

#python train.py --jsn womac4 --prj MOAKSID/contra00/00Bavgy_011 --models contra00Bavgy -b 4 --bt 4 --netG edalphand2 --dataset womac4 --nm 01 --ngf 32 --projection 32 --n_epochs 200 --direction ap_bp --save_d --alpha 1 --env runpod --lbcls 1 --adv 0 --lbt 0 --lbc 1 --lbtc 1a


#python train.py --jsn cyc_imorphics --prj IsoScopeXXcyc0B/ngf32lb10cut0 --models IsoScopeXXcyc0Bcut --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction xyori --nm 00 --netG ed023d --dataset DPM4X --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 10

#python train.py --jsn cyc_imorphics --prj IsoScopeXXcut/ngf32lb10 --models IsoScopeXXcut --cropz 20 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction xyori --nm 00 --netG ed023d --dataset Fly0B --n_epochs 2000 --lr_policy cosine --mc --lamb 10

#python train.py --jsn cyc_imorphics --prj IsoScopeXX/cyc0lb1skip4ndf32 --models IsoScopeXXcut --cropz 128 --cropsize 128 --env t09 --adv 1 --rotate --ngf 32 --direction t1norm2 --nm 11 --netG ed023d --dataset BraTSReg --n_epochs 2000 --lr_policy cosine --mc --lamb 1 --nocut  --ndf 32

#python train.py --jsn cyc_imorphics --prj IsoScopeXX/cyc0lb1skip4ndf32 --models IsoScopeXXcyc0cut --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG ed023d --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32

python train.py --jsn cyc_imorphics --prj IsoScopeXX/cyc0lb1skip4ndf32FIXBACK --models IsoScopeXXcyc0cut --cropz 128 --cropsize 128 --env t09 --adv 1 --rotate --ngf 32 --direction t1norm2 --nm 11 --netG ed023d --dataset BraTSReg --n_epochs 2000 --lr_policy cosine --mc --uprate 1 --lamb 1 --nocut  --skipl1 4 --ndf 32