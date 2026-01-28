#!/bin/bash
set -euxo pipefail

################################################################################
# Qwen3-235B-A22B GB300 Benchmarking Script (Torchrun Version)
# 
# This script uses torchrun for distributed training instead of SLURM.
# Supports both single-node and multi-node configurations.
#
# Usage:
#   Single Node:  ./gb300_launch_235b_v3.sh
#   Multi-Node:   Run on each node with appropriate NODE_RANK:
#     Master:     MASTER_ADDR=<master_ip> NODE_RANK=0 ./gb300_launch_235b_v3.sh
#     Worker:     MASTER_ADDR=<master_ip> NODE_RANK=1 ./gb300_launch_235b_v3.sh
#                 MASTER_ADDR=<master_ip> NODE_RANK=2 ./gb300_launch_235b_v3.sh
#                 ...
################################################################################


#===============================================================================
# Image and Repo version
#===============================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IMAGE: nvcr.io/nvidia/pytorch:25.12-py3
# MCore(2490e0c)
# TE(2f8ae81)
# DeepEP(3d4a884)


#===============================================================================
# Distributed Configuration (torchrun)
#===============================================================================
# Master node address (IP or hostname)
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# Master port for distributed communication
export MASTER_PORT=${MASTER_PORT:-29500}
# Current node rank (0 for master, 1,2,3... for workers)
export NODE_RANK=${NODE_RANK:-0}
# Number of GPUs per node
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}

#===============================================================================
# Path Configuration
#===============================================================================
export MEGATRON_PATH=${MEGATRON_PATH:-"/Your/Megatron-LM"}
export WORKSPACE=$(dirname "$(readlink -f "$0")")/../..
export BINDPCIE_PATH=${BINDPCIE_PATH:-""}

#===============================================================================
# GB300 Environment Variables
#===============================================================================
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1

#===============================================================================
# Model Configuration
#===============================================================================
export MODEL=Qwen3-235B-A22B
export WANDB_API_KEY=${WANDB_API_KEY:-"Your-wandb-api-key"}
export WANDB_PROJECT=Qwen3-235B-Benchmark-GB300
export OUTPUT_PATH=${OUTPUT_PATH:-"/Your/output/$WANDB_PROJECT"}

#===============================================================================
# Training Parameters (Overridable via environment)
#===============================================================================
# Parallelism configuration
export TP=${TP:-2}
export PP=${PP:-3}
export EP=${EP:-8}
export CP=${CP:-1}
export VPP=${VPP:-16}

# Batch size configuration
export MBS=${MBS:-3}
export GBS=${GBS:-1296}

# Model architecture
export NUM_LAYERS=${NUM_LAYERS:-94}
export SEQ_LEN=${SEQ_LEN:-4096}

# Node configuration
export NNODES=${NNODES:-18}

# HybridEP configuration, it must equal the EP size.
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-8}

# Precision configuration (bf16, fp8, mxfp8, nvfp4)
export PR=${PR:-"mxfp8"}

# Other settings
export PROFILE=${PROFILE:-0}
export PRETRAIN=${PRETRAIN:-1}
export DRY_RUN=${DRY_RUN:-0}
export A2A_OVERLAP=${A2A_OVERLAP:-0}
export DISPATCHER=${DISPATCHER:-"hybridep"}
export MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-true}

# Your Megatron-LM dataformat paths
export DATA_PATH=${DATA_PATH:-"/Your/datapath/OpenWebText/openwebtext_text_document"}
export LOAD_PATH="false"  # PRETRAIN=1 means training from scratch

# VPP calculation
if [[ ${VPP} -gt 1 ]]; then
    export LAYERS_PER_VP=$((NUM_LAYERS / PP / VPP))
else
    export LAYERS_PER_VP=false
fi

#===============================================================================
# Environment Variables for Training
#===============================================================================
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export NCCL_P2P_LEVEL=PXB
export NCCL_CUMEM_ENABLE=1
export NCCL_NET_GDR_C2C=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_DEBUG=VERSION
export NCCL_GRAPH_REGISTER=0
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache"}

# A2A Overlap settings
if [[ ${A2A_OVERLAP} == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NVTE_FWD_LAYERNORM_SM_MARGIN=20
    export NVTE_BWD_LAYERNORM_SM_MARGIN=20
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
fi

#===============================================================================
# Wandb Configuration
#===============================================================================
export COMMENT=${PR}
export WANDB_EXP_NAME="Qwen3-235B-A22B-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}"

#===============================================================================
# Output Directories
#===============================================================================
export OUTPUT_PATH="${OUTPUT_PATH}/TP${TP}PP${PP}EP${EP}VPP${VPP}-MBS${MBS}GBS${GBS}"
mkdir -p ${OUTPUT_PATH}/logs
mkdir -p ${OUTPUT_PATH}/tensorboard
mkdir -p ${OUTPUT_PATH}/checkpoints

#===============================================================================
# Training Script Path
#===============================================================================
export TRAINING_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"

#===============================================================================
# Build Training Parameters
#===============================================================================
TRAINING_PARAMS=""

# Distributed args
TRAINING_PARAMS+=" --distributed-timeout-minutes 220"
TRAINING_PARAMS+=" --tensor-model-parallel-size ${TP}"
TRAINING_PARAMS+=" --pipeline-model-parallel-size ${PP}"
if [[ ${LAYERS_PER_VP} != "false" ]]; then
    TRAINING_PARAMS+=" --num-layers-per-virtual-pipeline-stage ${LAYERS_PER_VP}"
fi
TRAINING_PARAMS+=" --expert-model-parallel-size ${EP}"
TRAINING_PARAMS+=" --context-parallel-size ${CP}"
TRAINING_PARAMS+=" --expert-tensor-parallel-size 1"
TRAINING_PARAMS+=" --use-distributed-optimizer"
TRAINING_PARAMS+=" --no-create-attention-mask-in-dataloader"
TRAINING_PARAMS+=" --account-for-embedding-in-pipeline-split"
TRAINING_PARAMS+=" --account-for-loss-in-pipeline-split"
TRAINING_PARAMS+=" --cross-entropy-loss-fusion"
TRAINING_PARAMS+=" --cross-entropy-fusion-impl native"
TRAINING_PARAMS+=" --enable-experimental"

# Training args
TRAINING_PARAMS+=" --use-mcore-models"
TRAINING_PARAMS+=" --sequence-parallel"
TRAINING_PARAMS+=" --use-flash-attn"
TRAINING_PARAMS+=" --disable-bias-linear"
TRAINING_PARAMS+=" --micro-batch-size ${MBS}"
TRAINING_PARAMS+=" --global-batch-size ${GBS}"
TRAINING_PARAMS+=" --train-samples 268554688"
TRAINING_PARAMS+=" --exit-duration-in-mins 230"
TRAINING_PARAMS+=" --manual-gc"
TRAINING_PARAMS+=" --manual-gc-interval 5"

# Transformer Engine args
TRAINING_PARAMS+=" --transformer-impl transformer_engine"

# Data args
TRAINING_PARAMS+=" --data-cache-path ${WORKSPACE}/data_cache"
TRAINING_PARAMS+=" --tokenizer-type HuggingFaceTokenizer"
TRAINING_PARAMS+=" --tokenizer-model Qwen/Qwen3-235B-A22B"
TRAINING_PARAMS+=" --data-path ${DATA_PATH}"
TRAINING_PARAMS+=" --split 99,1,0"
TRAINING_PARAMS+=" --no-mmap-bin-files"
TRAINING_PARAMS+=" --num-workers 6"

# Model architecture args
TRAINING_PARAMS+=" --untie-embeddings-and-output-weights"
TRAINING_PARAMS+=" --position-embedding-type rope"
TRAINING_PARAMS+=" --rotary-percent 1.0"
TRAINING_PARAMS+=" --rotary-base 1000000"
TRAINING_PARAMS+=" --rotary-seq-len-interpolation-factor 1"
TRAINING_PARAMS+=" --normalization RMSNorm"
TRAINING_PARAMS+=" --swiglu"
TRAINING_PARAMS+=" --norm-epsilon 1e-06"
TRAINING_PARAMS+=" --num-layers ${NUM_LAYERS}"
TRAINING_PARAMS+=" --hidden-size 4096"
TRAINING_PARAMS+=" --ffn-hidden-size 12288"
TRAINING_PARAMS+=" --num-attention-heads 64"
TRAINING_PARAMS+=" --kv-channels 128"
TRAINING_PARAMS+=" --group-query-attention"
TRAINING_PARAMS+=" --num-query-groups 4"
TRAINING_PARAMS+=" --qk-layernorm"
TRAINING_PARAMS+=" --seq-length ${SEQ_LEN}"
TRAINING_PARAMS+=" --max-position-embeddings 4096"
TRAINING_PARAMS+=" --make-vocab-size-divisible-by 1187"

# Regularization args
TRAINING_PARAMS+=" --attention-dropout 0.0"
TRAINING_PARAMS+=" --hidden-dropout 0.0"
TRAINING_PARAMS+=" --clip-grad 1.0"
TRAINING_PARAMS+=" --weight-decay 0.1"

# Learning rate args
TRAINING_PARAMS+=" --lr-decay-samples 584765624"
TRAINING_PARAMS+=" --lr-warmup-samples 1536000"
TRAINING_PARAMS+=" --lr-warmup-init 3.9e-7"
TRAINING_PARAMS+=" --lr 3.9e-6"
TRAINING_PARAMS+=" --min-lr 3.9e-7"
TRAINING_PARAMS+=" --lr-decay-style cosine"
TRAINING_PARAMS+=" --adam-beta1 0.9"
TRAINING_PARAMS+=" --adam-beta2 0.95"

# MoE args
TRAINING_PARAMS+=" --num-experts 128"
TRAINING_PARAMS+=" --moe-ffn-hidden-size 1536"
TRAINING_PARAMS+=" --moe-router-load-balancing-type aux_loss"
TRAINING_PARAMS+=" --moe-router-topk 8"
TRAINING_PARAMS+=" --moe-aux-loss-coeff 1e-3"
TRAINING_PARAMS+=" --moe-permute-fusion"
TRAINING_PARAMS+=" --moe-router-dtype fp32"
TRAINING_PARAMS+=" --moe-router-fusion"
if [[ ${MOE_GROUPED_GEMM} == "true" ]]; then
    TRAINING_PARAMS+=" --moe-grouped-gemm"
fi

# Validation args
TRAINING_PARAMS+=" --eval-iters 32"
TRAINING_PARAMS+=" --eval-interval 500"

# Checkpointing args
TRAINING_PARAMS+=" --finetune"
TRAINING_PARAMS+=" --auto-detect-ckpt-format"
TRAINING_PARAMS+=" --load ${LOAD_PATH}"
TRAINING_PARAMS+=" --no-load-rng"
TRAINING_PARAMS+=" --no-load-optim"
TRAINING_PARAMS+=" --save ${OUTPUT_PATH}/checkpoints"
TRAINING_PARAMS+=" --save-interval 1000"
TRAINING_PARAMS+=" --dist-ckpt-strictness log_all"

# Initialization args
TRAINING_PARAMS+=" --init-method-std 0.02"

# Logging args
TRAINING_PARAMS+=" --log-timers-to-tensorboard"
TRAINING_PARAMS+=" --log-memory-to-tensorboard"
TRAINING_PARAMS+=" --log-validation-ppl-to-tensorboard"
TRAINING_PARAMS+=" --log-throughput"
TRAINING_PARAMS+=" --log-interval 1"
TRAINING_PARAMS+=" --logging-level 40"
TRAINING_PARAMS+=" --tensorboard-dir ${OUTPUT_PATH}/tensorboard"
TRAINING_PARAMS+=" --wandb-project ${WANDB_PROJECT}"
TRAINING_PARAMS+=" --wandb-exp-name ${WANDB_EXP_NAME}"

# Mixed precision args
TRAINING_PARAMS+=" --bf16"

#===============================================================================
# Token Dispatcher Configuration
#===============================================================================
if [[ ${DISPATCHER} == "alltoall" ]]; then
    TRAINING_PARAMS+=" --moe-token-dispatcher-type alltoall"
elif [[ ${DISPATCHER} == "deepep" ]]; then
    TRAINING_PARAMS+=" --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep"
elif [[ ${DISPATCHER} == "hybridep" ]]; then
    TRAINING_PARAMS+=" --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --moe-hybridep-num-sms 32"
fi

#===============================================================================
# Precision-Specific Configuration
#===============================================================================
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS+=" --fp8-recipe blockwise --fp8-format e4m3"
    TRAINING_PARAMS+=" --fp8-param-gather"
    TRAINING_PARAMS+=" --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS+=" --moe-router-padding-for-fp8"
fi

if [[ ${PR} == "mxfp8" ]]; then
    TRAINING_PARAMS+=" --fp8-recipe mxfp8 --fp8-format e4m3"
    TRAINING_PARAMS+=" --fp8-param-gather --reuse-grad-buf-for-mxfp8-param-ag"
    TRAINING_PARAMS+=" --overlap-grad-reduce --overlap-param-gather"
    TRAINING_PARAMS+=" --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS+=" --moe-router-padding-for-quantization"
fi

if [[ ${PR} == "nvfp4" ]]; then
    TRAINING_PARAMS+=" --fp4-recipe nvfp4 --fp4-format e2m1"
    TRAINING_PARAMS+=" --overlap-grad-reduce --overlap-param-gather"
    TRAINING_PARAMS+=" --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS+=" --moe-router-padding-for-quantization"
fi

#===============================================================================
# A2A Overlap Configuration
#===============================================================================
if [[ ${A2A_OVERLAP} == 1 ]]; then
    TRAINING_PARAMS+=" --delay-wgrad-compute --overlap-moe-expert-parallel-comm"
fi

#===============================================================================
# VPP Configuration
#===============================================================================
if [[ ${VPP} -gt 1 ]]; then
    if [[ ! "${TRAINING_PARAMS}" =~ "--num-virtual-stages-per-pipeline-rank" ]] && \
       [[ ! "${TRAINING_PARAMS}" =~ "--num-layers-per-virtual-pipeline-stage" ]]; then
        TRAINING_PARAMS+=" --num-virtual-stages-per-pipeline-rank ${VPP}"
    fi
fi

#===============================================================================
# Additional Command Line Arguments (CUDA Graph, Force Load Balancing, etc.)
#===============================================================================
TRAINING_PARAMS+=" --moe-router-force-load-balancing"
TRAINING_PARAMS+=" --cuda-graph-impl transformer_engine"
TRAINING_PARAMS+=" --cuda-graph-scope attn moe_router moe_preprocess"

#===============================================================================
# Profile Configuration
#===============================================================================
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        -f true -x true \
        -o ${NSYS_PATH}/${WANDB_EXP_NAME}"
    TRAINING_PARAMS+=" --profile --profile-step-start 20 --profile-step-end 22 --profile-ranks 0"
else
    PROFILE_CMD=""
fi

#===============================================================================
# Calculate World Size
#===============================================================================
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
echo "World Size: ${WORLD_SIZE} (${NNODES} nodes x ${NPROC_PER_NODE} GPUs/node)"

#===============================================================================
# Build Torchrun Command
#===============================================================================
TORCHRUN_CMD="torchrun"
TORCHRUN_CMD+=" --nnodes=${NNODES}"
TORCHRUN_CMD+=" --nproc_per_node=${NPROC_PER_NODE}"
TORCHRUN_CMD+=" --node_rank=${NODE_RANK}"
TORCHRUN_CMD+=" --master_addr=${MASTER_ADDR}"
TORCHRUN_CMD+=" --master_port=${MASTER_PORT}"

# Optional: Add rdzv backend for elastic training
# TORCHRUN_CMD+=" --rdzv_backend=c10d"
# TORCHRUN_CMD+=" --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

#===============================================================================
# Full Training Command
#===============================================================================
if [[ -n "${BINDPCIE_PATH}" ]]; then
    FULL_CMD="${PROFILE_CMD} ${BINDPCIE_PATH} ${TORCHRUN_CMD} ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"
else
    FULL_CMD="${PROFILE_CMD} ${TORCHRUN_CMD} ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"
fi

#===============================================================================
# Execution
#===============================================================================
TIMESTAMP=$(date +'%y%m%d_%H%M%S')
LOG_FILE="${OUTPUT_PATH}/logs/train_${TIMESTAMP}_node${NODE_RANK}.log"

echo "============================================================"
echo "=== Configuration Summary ==="
echo "============================================================"
echo "MODEL:           ${MODEL}"
echo "MEGATRON_PATH:   ${MEGATRON_PATH}"
echo "MASTER_ADDR:     ${MASTER_ADDR}"
echo "MASTER_PORT:     ${MASTER_PORT}"
echo "NODE_RANK:       ${NODE_RANK}"
echo "NNODES:          ${NNODES}"
echo "NPROC_PER_NODE:  ${NPROC_PER_NODE}"
echo "WORLD_SIZE:      ${WORLD_SIZE}"
echo "TP:              ${TP}"
echo "PP:              ${PP}"
echo "EP:              ${EP}"
echo "VPP:             ${VPP}"
echo "MBS:             ${MBS}"
echo "GBS:             ${GBS}"
echo "PRECISION:       ${PR}"
echo "DISPATCHER:      ${DISPATCHER}"
echo "A2A_OVERLAP:     ${A2A_OVERLAP}"
echo "OUTPUT_PATH:     ${OUTPUT_PATH}"
echo "LOG_FILE:        ${LOG_FILE}"
echo "============================================================"

if [[ ${DRY_RUN:-0} -eq 1 ]]; then
    echo "=== DRY RUN - Full Training Command ==="
    echo "============================================================"
    echo "${FULL_CMD}"
    echo "============================================================"
    echo ""
    echo "=== Multi-Node Launch Instructions ==="
    echo "============================================================"
    echo "To launch on multiple nodes, run on each node:"
    echo ""
    echo "Node 0 (Master):"
    echo "  MASTER_ADDR=${MASTER_ADDR} NODE_RANK=0 ./gb300_launch_235b_v3.sh"
    echo ""
    for ((i=1; i<NNODES; i++)); do
        echo "Node ${i} (Worker):"
        echo "  MASTER_ADDR=${MASTER_ADDR} NODE_RANK=${i} ./gb300_launch_235b_v3.sh"
        echo ""
    done
    echo "============================================================"
else
    echo "=== Starting Training ==="
    echo "Logging to: ${LOG_FILE}"
    echo "============================================================"
    
    # Change to Megatron directory
    cd ${MEGATRON_PATH}
    
    # Run training
    ${FULL_CMD} 2>&1 | tee ${LOG_FILE}
    
    echo "============================================================"
    echo "=== Training Completed ==="
    echo "============================================================"
fi

