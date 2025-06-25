# MDH

The official implementation of the paper "MDH: Mask-induced Decoupled Hashing for Fine-grained Image Retrieval".

## Dependencies

This is the list of the package versions required for our experiments.

```txt
python==3.10.14
pytorch==2.4.0
loguru == 0.7.3
thop==0.1.1
```

## training
```txt
python main.py --dataset cub-2011 --root /data/CUB_200_2011 --max-epoch 30 --batch-size 16 --max-iter 40 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 25,35 --num-samples 2000 --info 'CUB' --momen 0.91
```
