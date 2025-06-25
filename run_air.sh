#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --dataset aircraft --root /data/yhx/data/FGVC-Aircraft --max-epoch 30 --batch-size 16 --max-iter 40 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 25,35 --num-samples 2000 --info 'Aircraft' --momen 0.91