#python train.py --jsn cyc_imorphics --prj IsoScopeXX/cyc0lb1skip4ndf32randl1 --models IsoScopeXXcyc0cut --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG ed023d --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32 --randl1

#python train.py --jsn cyc_imorphics --prj IsoScopeXX/cyc0lb1skip4ndf32nomc --models IsoScopeXXcyc0cut --cropz 16 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG ed023d --dataset womac4 --n_epochs 2000 --lr_policy cosine --uprate 8 --lamb 1 --nocut --skipl1 4 --ndf 32

#python train.py --jsn cyc_imorphics --prj IsoScopeXXcutcyc/ngf32lb10FIXBACK --models IsoScopeXXcutcyc --cropz 20 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction xyori --nm 00 --netG ed023d --dataset Fly0B --n_epochs 6000 --lr_policy cosine --mc --lamb 10

#python train.py --jsn cyc_imorphics --prj IsoScopeXXcut/ngf32lb10 --models IsoScopeXXcut --cropz 20 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction xyori --nm 00 --netG ed023d --dataset Fly0B --n_epochs 6000 --lr_policy cosine --mc --lamb 10


# IsoScopeXY

# womac4
#python train.py --jsn cyc_imorphics --prj IsoScopeXY/ngf32ndf32lb10skip4nocut --models IsoScopeXY --cropz 16 --cropsize 128 --env a6k --adv 1 --rotate --ngf 32 --direction ap --nm 11 --netG ed023d --dataset womac4 --n_epochs 2000 --lr_policy cosine --mc --lamb 1 --skipl1 4 --nocut

# Fly0B
#python train.py --jsn cyc_imorphics --prj IsoScopeXY/ngf32lb10skip4 --models IsoScopeXY --cropz 20 --cropsize 128 --env runpod --adv 1 --rotate --ngf 32 --direction xyori --nm 00 --netG ed023d --dataset Fly0B --n_epochs 6000 --lr_policy cosine --mc --lamb 10 --skipl1 4
