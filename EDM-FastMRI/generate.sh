CUDA_VISIBLE_DEVICES=0
NPROC=1
ROOT=/path/to/root/
MODEL_PATH=network-snapshot.pkl
SAMPLE_DIM=384,320
NUM_SAMPLES=100
BATCH_SIZE=40

ANATOMY=brain
DATA=noisy32dB

torchrun --standalone --nproc_per_node=$NPROC generate.py \
        --outdir=$ROOT/results/priors/$ANATOMY/$DATA --seeds=1-$NUM_SAMPLES \
        --batch=$BATCH_SIZE --network=$ROOT/models/edm/$ANATOMY/$DATA/$MODEL_PATH \
        --sample_dim=$SAMPLE_DIM --gpu=$CUDA_VISIBLE_DEVICES