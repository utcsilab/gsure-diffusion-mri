CUDA_VISIBLE_DEVICES=0
NPROC=1
EPOCHS=10
ANATOMY=brain
DATA=noisy
ROOT=/path/to/root/
METHOD=modl

SNR=32dB

KSP_PATH=/path/to/data/$ANATOMY/train/$SNR/ksp/
DATA_PATH=/path/to/data/$ANATOMY/train/$SNR/$DATA.pt
R=4

torchrun --standalone --nproc_per_node=$NPROC train.py \
    --gpu=$CUDA_VISIBLE_DEVICES --data_R=$R --epochs=$EPOCHS \
    --snr=$SNR --anatomy=$ANATOMY --root=$ROOT \
    --ksp_path=$KSP_PATH --data_path=$DATA_PATH \
    --data_type=$DATA --method=$METHOD