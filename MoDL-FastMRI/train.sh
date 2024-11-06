GPU=0
EPOCHS=10
ANATOMY=brain
DATA=noisy
ROOT=/csiNAS/asad/MoDL-FastMRI

for SNR in 32dB 22dB 12dB;
do
    KSP_PATH=$ROOT/data/$ANATOMY/ksp/$SNR/
    DATA_PATH=$ROOT/data/$ANATOMY/samples/$SNR.pt
   
    for R in 4 8;
    do
        torchrun --standalone train.py \
            --gpu=$GPU --data_R=$R --epochs=$EPOCHS \
            --snr=$SNR --anatomy=$ANATOMY --root=$ROOT \
            --ksp_path=$KSP_PATH --data_path=$DATA_PATH \
            --data_type=$DATA
    done
done