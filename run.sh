CUDA_VISIBLE_DEVICES=0 python train.py --jsn womac4 --prj cls2/ --models lesion_siamese --netG edalphand2  --dataset womac4 --lbvgg 0 --lbNCE 0 --nm 01 --fDown 4  -b 4 --ngf 32 --projection 2 --n_epochs 200 --lr_policy cosine --direction ap_bp --save_d --alpha 1 --env t09b --split a