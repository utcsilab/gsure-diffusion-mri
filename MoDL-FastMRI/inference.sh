CUDA_VISIBLE_DEVICES=0
NPROC=1
ANATOMY=brain
NATIVE_SNR=32dB
ROOT=/path/to/root/
MODEL_PATH=ckpt_9.pt
MEAS_PATH=/path/to/data/brain/val/$NATIVE_SNR
SAMPLE_START=0
SAMPLE_END=100
METHOD=modl

DATA=noisy32dB
INFERENCE_SNR=32dB
R=4
KSP_PATH=/path/to/data/brain/val/$INFERENCE_SNR
            
torchrun --standalone --nproc_per_node=$NPROC inference.py \
    --gpu=$CUDA_VISIBLE_DEVICES --anatomy=$ANATOMY\
    --sample_start $SAMPLE_START --sample_end $SAMPLE_END \
    --inference_R $R --inference_snr $INFERENCE_SNR \
    --measurements_path $MEAS_PATH \
    --ksp_path $KSP_PATH \
    --network=$ROOT/models_$METHOD/$ANATOMY/$DATA/R=$R/$MODEL_PATH \
    --outdir=$ROOT/results_$METHOD/$ANATOMY/$DATA/R=$R/snr$INFERENCE_SNR \
    --method=$METHOD