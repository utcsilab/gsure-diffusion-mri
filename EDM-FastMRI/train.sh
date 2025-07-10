CUDA_VISIBLE_DEVICES=4
NPROC=1
LOADER=Noisy
ANATOMY=knee
DATA=noisy
SNR=14dB
ROOT=/csiNAS/asad/GSURE-FastMRI
ROOT_DATA=/csiNAS/asad/DATA-FastMRI
BATCH_SIZE=2
NORMALIZE=0
PRECOND=gsure
AUGMENT=0

torchrun --standalone --nproc_per_node=$NPROC train.py \
 --outdir=$ROOT/models/edm/$ANATOMY/$SNR \
 --data=$ROOT_DATA/$ANATOMY/train/$SNR/$DATA \
 --cond=0 --arch=ddpmpp --duration=10 \
 --batch=$BATCH_SIZE --cbase=128 --cres=1,1,2,2,2,2,2 \
 --lr=1e-4 --ema=0.1 --dropout=0.0 \
 --desc=container_test --tick=1 --snap=10 \
 --dump=200 --seed=2023 --precond=$PRECOND --augment=$AUGMENT \
 --normalize=$NORMALIZE --loader=$LOADER --gpu=$CUDA_VISIBLE_DEVICES