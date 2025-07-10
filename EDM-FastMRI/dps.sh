CUDA_VISIBLE_DEVICES=0
NPROC=1
ANATOMY=brain
NATIVE_SNR=32dB
ROOT=/path/to/root/
MODEL_PATH=network-snapshot.pkl
MEAS_PATH=/path/to/meas/$NATIVE_SNR
STEPS=500
SAMPLE_START=0
SAMPLE_END=100

DATA=noisy32dB
INFERENCE_SNR=32dB
R=4
SEED=15
KSP_PATH=/path/to/ksp/$INFERENCE_SNR

torchrun --standalone --nproc_per_node=$NPROC dps.py \
    --seed $SEED --latent_seeds $SEED --gpu=$CUDA_VISIBLE_DEVICES \
    --sample_start $SAMPLE_START --sample_end $SAMPLE_END \
    --inference_R $R --inference_snr $INFERENCE_SNR \
    --num_steps $STEPS --S_churn 0 \
    --measurements_path $MEAS_PATH \
    --ksp_path $KSP_PATH \
    --network=$ROOT/models/edm/$ANATOMY/$DATA/$MODEL_PATH \
    --outdir=$ROOT/results/posterior/$ANATOMY/$DATA