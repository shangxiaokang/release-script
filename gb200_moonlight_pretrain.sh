# Version Info
# NGC Image: nvcr.io/nvidia/pytorch:26.04-py3
# Driver Version: 580.159.04
# CUDA Version: 13.2
# TE: v2.15 42b84005
# MCore: 2ee3bfb2c
# install transformers by pip install transformers in the container
pip install flask flask-restful uvloop
pip install omegaconf
pip install tiktoken 

# wandb login
if [ "${SLURM_LOCALID:-0}" -eq 0 ]; then
   wandb login "your wandb token"
fi

# 1.Select the megatron-lm code base and set the log name
EXP="Moonlight16B-nvfp4"
NAME=Moonlight_16B

CODE_DIR=/home/xshang/Megatron-LM
CODE_DIR=/home/xshang/Quark/Megatron-LM

OUT_DIR=/lustre/fsw/general_sa/xshang/Moonlight-16B
DATA_CACHE_PATH=$OUT_DIR/cache/$NAME
CHECKPOINT_PATH=$OUT_DIR/checkpoint/$NAME
TENSORBOARD_PATH=$OUT_DIR/tensorboard/$NAME

mkdir -p $OUT_DIR
mkdir -p $DATA_CACHE_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $TENSORBOARD_PATH

echo "Current megatron code base is: $CODE_DIR"
echo "Current output path is: $OUT_DIR"
echo "Current checkpoint path is: $CHECKPOINT_PATH"
echo "Current tensorboard path is: $TENSORBOARD_PATH"
echo "Current data cache path is: $DATA_CACHE_PATH"
echo "Current wandb save path is: $OUT_DIR/wandb"

# 2.Set quantization config
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/caell/TE_build/main/2712bb/build/lib.linux-x86_64-cpython-312/transformer_engine

# 3.Set the training env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_DEBUG=0

# 4.Set the dataset, tokenizer and checkpoint path and config
DATA_PATH=/lustre/fsw/general_sa/xshang/dataset/imdb/imdb_megatron_text_document

TOKENIZER_MODEL="moonshotai/Moonlight-16B-A3B-Instruct"

DATA_ARGS="
    --data-path ${DATA_PATH} 
    --data-cache-path $DATA_CACHE_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
    --split 99,1,0
    --num-workers 8 
"

# 5.Set the distributed arguments env and config
GPUS_PER_NODE=4
NNODES=${SLURM_NNODES:-2}
NODE_RANK=${SLURM_NODEID:-0}
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
"

# 6.Set the parallel startegy env and config
GBS=768
MBS=1
TP_SIZE=1
PP_SIZE=1
CP_SIZE=1
EP_SIZE=4
SEED=1234

MODEL_PARALLEL_ARGS="
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --overlap-grad-reduce
    --overlap-param-gather
"

# 7.Set the model env and config
	# --no-load-optim	
MODEL_ARGS="

	--untie-embeddings-and-output-weights 
	--no-bias-swiglu-fusion 
    --swiglu 
    --use-mcore-models
    --transformer-impl transformer_engine
    --disable-bias-linear
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-percent 1.0
    --rotary-base 50000
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --multi-latent-attention 
    --num-attention-heads 16 
    --attention-backend auto 
    --hidden-dropout 0.0 
    --kv-lora-rank 512 
    --qk-head-dim 128 
    --qk-pos-emb-head-dim 64 
    --v-head-dim 128 
    --seq-length 8192
    --num-layers 27
    --hidden-size 2048
    --ffn-hidden-size 11264
    --qk-layernorm
    --max-position-embeddings 8192
    --attention-dropout 0.0
"

MOE_ARGS=" 
    --num-experts 64 
    --expert-model-parallel-size ${EP_SIZE} 
    --moe-layer-freq 1 
    --moe-ffn-hidden-size 1408 
    --moe-shared-expert-intermediate-size 2816 
    --moe-router-load-balancing-type seq_aux_loss 
    --moe-aux-loss-coeff 1e-3 
    --moe-router-topk 6 
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --moe-router-dtype fp32 

    --moe-router-topk-scaling-factor 2.446 
    --moe-router-score-function sigmoid 
    --moe-router-enable-expert-bias 
    --moe-router-bias-update-rate 1e-3 
    --moe-token-dispatcher-type alltoall 
"

# 8. Set the training time and log info
WANDB_EXP_NAME="${EXP}_gbs${GBS}_mbs${MBS}_tp${TP_SIZE}_ep${EP_SIZE}"

EVAL_AND_LOGGING_ARGS="
    --log-interval 20
    --save-interval 200
    --eval-interval 2000
    --eval-iters 14
    --log-throughput
    --log-num-zeros-in-grad
    --log-timers-to-tensorboard
    --log-params-norm
    --log-straggler 	
    --disable-straggler-on-startup 
    --straggler-minmax-count 16 
    --timing-log-option minmax
    --tensorboard-dir $TENSORBOARD_PATH
    --wandb-project "Moonlight-16B-A3B-Instruct"
    --wandb-exp-name $WANDB_EXP_NAME
    --wandb-save-dir $OUT_DIR
"

# 9. Set Training

SAMPLES=135235850
TRAINING_ARGS="
    --check-weight-hash-across-dp-replicas-interval 20000
    --train-samples ${SAMPLES} 
    --lr-decay-samples 134211850 
    --lr-wsd-decay-style linear 
    --lr-wsd-decay-samples 20285377 
    --lr-warmup-samples 1024000 
    --manual-gc 
    --ckpt-assume-constant-structure 
    --ckpt-format torch_dist 
    --ckpt-fully-parallel-save 
    --ckpt-fully-parallel-load 
    --use-distributed-optimizer
    --lr 1.2e-3
    --weight-decay 0.1
    --lr-decay-style WSD
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --min-lr 1.2e-5
    --init-method-std 0.0198
    --attention-backend auto
    --seed $SEED
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --no-create-attention-mask-in-dataloader 
"


# 10. Set the training precision
PRECISION_ARGS="
	--bf16
"

OTHER_ARGS="
    --trust-remote-code 
    --tiktoken-pattern v2 
    --no-mmap-bin-files 
    --log-progress
    --fp4-recipe nvfp4
    --fp4-format e2m1
"
    # --moe-shared-expert-overlap 

RANK=$SLURM_PROCID
echo "RANK: $RANK"
echo "WORLD SIZE: $WORLD_SIZE"

#    $OTHER_ARGS \
# 11. launch the training task with torchrun
torchrun $DISTRIBUTED_ARGS ${CODE_DIR}/pretrain_gpt.py \
    $DATA_ARGS \
    $MODEL_PARALLEL_ARGS \
    $MODEL_ARGS \
    $EVAL_AND_LOGGING_ARGS \
    $TRAINING_ARGS \
    $PRECISION_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    1>$OUT_DIR/rank_${RANK}.log 2>$OUT_DIR/rank_${RANK}.err 

echo "Training completed or terminated. Check logs at $OUT_DIR"

