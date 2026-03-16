#!/bin/bash
# Megatron-LM evaluation script with Tensor Parallelism

export HF_HOME=/lustre/raplab/client/xshang/workspace/huggingface
export MEGATRON_PATH=/lustre/raplab/client/xshang/workspace/Megatron-LM
export MASTER_ADDR=localhost
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_ALLOW_CODE_EVAL=1
export HF_HUB_TRUST_REMOTE_CODE=1

# 根据 DTYPE 选择 checkpoint 路径
DTYPE=BF16
if [ -z "${DTYPE}" ]; then
    echo "Error: DTYPE environment variable is not set. Please set DTYPE=FP4 or DTYPE=BF16"
    exit 1
elif [ "${DTYPE}" = "NVFP4" ]; then
    LOAD=/lustre/raplab/client/xshang/workspace/huggingface/Ling-Mini/NVFP4/
elif [ "${DTYPE}" = "BF16" ]; then
    LOAD=/lustre/raplab/client/xshang/workspace/huggingface/Ling-Mini/BF16/ITER
else
    echo "Error: Invalid DTYPE value '${DTYPE}'. Must be NVFP4 or BF16"
    exit 1
fi

TOKENIZER_TYPE=HuggingFaceTokenizer
TOKENIZER_MODEL=moonshotai/Moonlight-16B-A3B-Instruct

EXTRA_ARGS="
    --use-checkpoint-args
    --no-use-tokenizer-model-from-checkpoint-args
    --trust-remote-code
    --untie-embeddings-and-output-weights
    --swiglu
    --use-mcore-models
    --transformer-impl transformer_engine
    --disable-bias-linear
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-base 10000
    --rotary-percent 0.5
    --rotary-scaling-factor 40
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --attention-backend auto
    --hidden-dropout 0
    --num-layers 20
    --hidden-size 2048
    --ffn-hidden-size 5120
    --qk-layernorm
    --max-position-embeddings 4096
    --attention-dropout 0
    --num-experts 256
    --moe-layer-freq [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-router-load-balancing-type aux_loss
    --moe-z-loss-coeff 0.0000035
    --moe-router-topk 8
    --moe-router-topk-scaling-factor 2.5
    --moe-grouped-gemm
    --moe-router-dtype fp32
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type alltoall
    --moe-shared-expert-overlap
    --moe-permute-fusion
    --moe-aux-loss-coeff 0.001
"

# Tensor Parallel size (number of GPUs)
TP=1
EP=8
DEVICES=${DEVICES:-8}

CKPT_STEP=${STEP:-196500}
BATCH_SIZE=${BATCH:-16}
TASK=${TASK:-humaneval}

# Convert EXTRA_ARGS to single line (remove newlines)
EXTRA_ARGS_ONELINE=$(echo $EXTRA_ARGS | tr '\n' ' ' | tr -s ' ')

# Build model_args as JSON to avoid comma-delimiter conflicts
# (extra_args contains values like --moe-layer-freq [0,1,...] with internal commas)
read -r -d '' MODEL_ARGS_JSON <<EOJSON
{
  "devices": ${DEVICES},
  "tensor_model_parallel_size": ${TP},
  "expert_model_parallel_size": ${EP},
  "micro_batch_size": ${BATCH_SIZE},
  "load": "${LOAD}",
  "ckpt_step": "${CKPT_STEP}",
  "tokenizer_type": "${TOKENIZER_TYPE}",
  "tokenizer_model": "${TOKENIZER_MODEL}",
  "extra_args": "${EXTRA_ARGS_ONELINE}"
}
EOJSON

# Use torchrun for multi-GPU with Tensor Parallelism
torchrun --nproc_per_node=${DEVICES} --master_port ${MASTER_PORT} \
    -m lm_eval --model megatron_lm \
    --model_args "${MODEL_ARGS_JSON}" \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    --log_samples \
    --confirm_run_unsafe_code \
    --output_path ./results
