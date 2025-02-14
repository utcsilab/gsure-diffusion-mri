GPU=2
EPOCHS=10
ANATOMY=brain
DATA=noisy
ROOT=/csiNAS/asad/MoDL-FastMRI
METHOD=ensure

for SNR in 12dB;
do
    KSP_PATH=/csiNAS/asad/DATA-FastMRI/$ANATOMY/train/$SNR/ksp/
    DATA_PATH=/csiNAS/asad/DATA-FastMRI/$ANATOMY/train/$SNR/$DATA.pt
   
    for R in 4 8;
    do
        python -m torch.distributed.run --standalone train.py \
            --gpu=$GPU --data_R=$R --epochs=$EPOCHS \
            --snr=$SNR --anatomy=$ANATOMY --root=$ROOT \
            --ksp_path=$KSP_PATH --data_path=$DATA_PATH \
            --data_type=$DATA --method=$METHOD
    done
done