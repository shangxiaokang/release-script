#!/bin/bash
set -euxo pipefail

###############################################################################
# Self-contained DeepSeek-V3 benchmarking script for GB300 (torchrun launcher)
# Converted from: run_deepseek_v3_gb300_slurm.sh
#
# Usage (single-node):
#   bash run_deepseek_v3_gb300_torchrun.sh
#
# Usage (multi-node): run on EACH node with appropriate NODE_RANK
#   NNODES=18 NODE_RANK=0 MASTER_ADDR=<master_ip> bash run_deepseek_v3_gb300_torchrun.sh  # on node 0
#   NNODES=18 NODE_RANK=1 MASTER_ADDR=<master_ip> bash run_deepseek_v3_gb300_torchrun.sh  # on node 1
#   ...
###############################################################################

#===============================================================================
# Image and Repo version
#===============================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IMAGE: nvcr.io/nvidia/pytorch:25.12-py3
# MCore(9374a4d)
# TE(2f8ae81)
# DeepEP(3d4a884)


# ========================== Cluster / path variables ==========================
export MEGATRON_PATH=/Your/Megatron-LM
export MCORE_RELEASE_VERSION='Main'
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false

# ========================== Model selection ===================================
export MODEL=DeepSeek-V3
export RUN_NAME="${MODEL}-benchmarking"

# ========================== Parallelism & training ============================
export TP=${TP:-2}
export PP=${PP:-3}
export EP=${EP:-8}
export CP=${CP:-1}
export VPP=${VPP:-1}
export MBS=${MBS:-1}
export GBS=${GBS:-1536}
export SEQ_LEN=${SEQ_LEN:-4096}
export NNODES=${NNODES:-18}
export NUM_LAYERS=31
export PRETRAIN=${PRETRAIN:-1}
export PR=${PR:-mxfp8}
export DISPATCHER=${DISPATCHER:-"hybridep"}
export PROFILE=${PROFILE:-0}

# ========================== Torchrun variables ================================
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}

# ========================== NCCL / HybridEP env vars =========================
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export USE_MNNVL=1
export NUM_OF_IN_FLIGHT_S2G_DISPATCH_API=8
export NUM_OF_STAGES_DISPATCH_API=10
export NVLINK_DOMAIN_SIZE=72

# ========================== Derived paths =====================================
export TRAINING_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"
export COMMENT=${COMMENT:-"v${MCORE_RELEASE_VERSION}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"${USER}-moe-benchmarking-v${MCORE_RELEASE_VERSION}"}
export WANDB_EXP_NAME="DeepSeek-V3-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}"

WORKSPACE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
export OUTPUT_PATH=${OUTPUT_PATH:-"${WORKSPACE}/output/mcore-benchmarking-v${MCORE_RELEASE_VERSION}/${MODEL}-TP${TP}PP${PP}EP${EP}VPP${VPP}-MBS${MBS}GBS${GBS}"}

# ========================== Data & checkpoint paths ===========================
export DATA_PATH=${DATA_PATH:-"/Your/dataset"}

if [[ ${PRETRAIN} == 0 ]]; then
    export LOAD_PATH=${LOAD_PATH:-"/Your/ckpt"}
else
    export LOAD_PATH="false"
fi

# ========================== Environment variables (from YAML ENV_VARS) ========
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=VERSION
export NCCL_GRAPH_REGISTER=0
export NCCL_P2P_LEVEL=PXB
export NCCL_CUMEM_ENABLE=1
export NCCL_NET_GDR_C2C=1
export NCCL_NET_GDR_LEVEL=SYS
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache"}

# ========================== Build TRAINING_PARAMS =============================
TRAINING_PARAMS=""

# --- Distributed args ---
TRAINING_PARAMS+=" --distributed-timeout-minutes 60"
TRAINING_PARAMS+=" --tensor-model-parallel-size ${TP}"
TRAINING_PARAMS+=" --pipeline-model-parallel-size ${PP}"
TRAINING_PARAMS+=" --expert-model-parallel-size ${EP}"
TRAINING_PARAMS+=" --context-parallel-size ${CP}"
TRAINING_PARAMS+=" --expert-tensor-parallel-size 1"
TRAINING_PARAMS+=" --use-distributed-optimizer"
TRAINING_PARAMS+=" --overlap-grad-reduce"
TRAINING_PARAMS+=" --overlap-param-gather"

# --- Training args ---
TRAINING_PARAMS+=" --use-mcore-models"
TRAINING_PARAMS+=" --sequence-parallel"
TRAINING_PARAMS+=" --use-flash-attn"
TRAINING_PARAMS+=" --disable-bias-linear"
TRAINING_PARAMS+=" --micro-batch-size ${MBS}"
TRAINING_PARAMS+=" --global-batch-size ${GBS}"
TRAINING_PARAMS+=" --train-samples 585937500"
TRAINING_PARAMS+=" --exit-duration-in-mins 220"
TRAINING_PARAMS+=" --no-save-optim"
TRAINING_PARAMS+=" --no-check-for-nan-in-loss-and-grad"
TRAINING_PARAMS+=" --cross-entropy-loss-fusion"
TRAINING_PARAMS+=" --cross-entropy-fusion-impl te"
TRAINING_PARAMS+=" --manual-gc"
TRAINING_PARAMS+=" --manual-gc-interval 10"
TRAINING_PARAMS+=" --enable-experimental"

# --- Transformer Engine args ---
TRAINING_PARAMS+=" --transformer-impl transformer_engine"

# --- Data args ---
TRAINING_PARAMS+=" --seq-length ${SEQ_LEN}"
TRAINING_PARAMS+=" --data-cache-path ${WORKSPACE}/data_cache"
TRAINING_PARAMS+=" --tokenizer-type HuggingFaceTokenizer"
TRAINING_PARAMS+=" --tokenizer-model unsloth/DeepSeek-V3"
TRAINING_PARAMS+=" --data-path ${DATA_PATH}"
TRAINING_PARAMS+=" --split 99,1,0"
TRAINING_PARAMS+=" --no-mmap-bin-files"
TRAINING_PARAMS+=" --no-create-attention-mask-in-dataloader"
TRAINING_PARAMS+=" --num-workers 6"

# --- Network size args ---
TRAINING_PARAMS+=" --num-layers ${NUM_LAYERS}"
TRAINING_PARAMS+=" --hidden-size 7168"
TRAINING_PARAMS+=" --ffn-hidden-size 18432"
TRAINING_PARAMS+=" --num-attention-heads 128"
TRAINING_PARAMS+=" --kv-channels 128"
TRAINING_PARAMS+=" --max-position-embeddings 4096"
TRAINING_PARAMS+=" --position-embedding-type rope"
TRAINING_PARAMS+=" --rotary-base 10000"
TRAINING_PARAMS+=" --make-vocab-size-divisible-by 3232"
TRAINING_PARAMS+=" --normalization RMSNorm"
TRAINING_PARAMS+=" --norm-epsilon 1e-6"
TRAINING_PARAMS+=" --swiglu"
TRAINING_PARAMS+=" --untie-embeddings-and-output-weights"
TRAINING_PARAMS+=" --multi-latent-attention"

# --- Regularization args ---
TRAINING_PARAMS+=" --attention-dropout 0.0"
TRAINING_PARAMS+=" --hidden-dropout 0.0"
TRAINING_PARAMS+=" --clip-grad 1.0"
TRAINING_PARAMS+=" --weight-decay 0.1"
TRAINING_PARAMS+=" --qk-layernorm"

# --- Learning rate args ---
TRAINING_PARAMS+=" --lr-decay-samples 584765624"
TRAINING_PARAMS+=" --lr-warmup-samples 1536000"
TRAINING_PARAMS+=" --lr-warmup-init 3.9e-7"
TRAINING_PARAMS+=" --lr 3.9e-6"
TRAINING_PARAMS+=" --min-lr 3.9e-7"
TRAINING_PARAMS+=" --lr-decay-style cosine"
TRAINING_PARAMS+=" --adam-beta1 0.9"
TRAINING_PARAMS+=" --adam-beta2 0.95"

# --- MoE args ---
TRAINING_PARAMS+=" --num-experts 256"
TRAINING_PARAMS+=" --moe-layer-freq ([0]*3+[1]*28)"
TRAINING_PARAMS+=" --moe-ffn-hidden-size 2048"
TRAINING_PARAMS+=" --moe-shared-expert-intermediate-size 2048"
TRAINING_PARAMS+=" --moe-router-load-balancing-type seq_aux_loss"
TRAINING_PARAMS+=" --moe-router-topk 8"
TRAINING_PARAMS+=" --moe-grouped-gemm"
TRAINING_PARAMS+=" --moe-aux-loss-coeff 1e-4"
TRAINING_PARAMS+=" --moe-router-group-topk 4"
TRAINING_PARAMS+=" --moe-router-num-groups 8"
TRAINING_PARAMS+=" --moe-router-topk-scaling-factor 2.5"
TRAINING_PARAMS+=" --moe-router-score-function sigmoid"
TRAINING_PARAMS+=" --moe-router-enable-expert-bias"
TRAINING_PARAMS+=" --moe-router-bias-update-rate 1e-3"
TRAINING_PARAMS+=" --moe-router-dtype fp32"
TRAINING_PARAMS+=" --moe-permute-fusion"
TRAINING_PARAMS+=" --moe-router-fusion"

# --- MLA args ---
TRAINING_PARAMS+=" --q-lora-rank 1536"
TRAINING_PARAMS+=" --kv-lora-rank 512"
TRAINING_PARAMS+=" --qk-head-dim 128"
TRAINING_PARAMS+=" --qk-pos-emb-head-dim 64"
TRAINING_PARAMS+=" --v-head-dim 128"
TRAINING_PARAMS+=" --rotary-scaling-factor 40"
TRAINING_PARAMS+=" --mscale 1.0"
TRAINING_PARAMS+=" --mscale-all-dim 1.0"

# --- Validation args ---
TRAINING_PARAMS+=" --eval-iters 32"
TRAINING_PARAMS+=" --eval-interval 200"

# --- Checkpointing args ---
TRAINING_PARAMS+=" --no-load-optim"
TRAINING_PARAMS+=" --no-load-rng"
TRAINING_PARAMS+=" --auto-detect-ckpt-format"
TRAINING_PARAMS+=" --load ${LOAD_PATH}"
TRAINING_PARAMS+=" --save ${OUTPUT_PATH}/checkpoints"
TRAINING_PARAMS+=" --save-interval 500"
TRAINING_PARAMS+=" --dist-ckpt-strictness log_all"

# --- Initialization args ---
TRAINING_PARAMS+=" --init-method-std 0.02"

# --- Logging args ---
TRAINING_PARAMS+=" --log-timers-to-tensorboard"
TRAINING_PARAMS+=" --log-memory-to-tensorboard"
TRAINING_PARAMS+=" --log-validation-ppl-to-tensorboard"
TRAINING_PARAMS+=" --log-throughput"
TRAINING_PARAMS+=" --log-interval 1"
TRAINING_PARAMS+=" --logging-level 40"
TRAINING_PARAMS+=" --tensorboard-dir ${OUTPUT_PATH}/tensorboard"
TRAINING_PARAMS+=" --wandb-project ${WANDB_PROJECT}"
TRAINING_PARAMS+=" --wandb-exp-name ${WANDB_EXP_NAME}"

# --- Mixed precision args ---
TRAINING_PARAMS+=" --bf16"
TRAINING_PARAMS+=" --enable-experimental"

# ========================== VPP handling ======================================
if [[ ${VPP} -gt 1 ]]; then
    TRAINING_PARAMS+=" --num-virtual-stages-per-pipeline-rank ${VPP}"
fi

# ========================== Dispatcher handling ===============================
if [[ ${DISPATCHER} == "alltoall" ]]; then
    TRAINING_PARAMS+=" --moe-token-dispatcher-type alltoall"
elif [[ ${DISPATCHER} == "deepep" ]]; then
    TRAINING_PARAMS+=" --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep"
elif [[ ${DISPATCHER} == "hybridep" ]]; then
    TRAINING_PARAMS+=" --moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --moe-hybridep-num-sms 32"
fi

# ========================== Precision handling ================================
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS+=" --fp8-recipe blockwise --fp8-format e4m3"
    TRAINING_PARAMS+=" --fp8-param-gather"
    TRAINING_PARAMS+=" --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS+=" --moe-router-padding-for-fp8"
elif [[ ${PR} == "mxfp8" ]]; then
    TRAINING_PARAMS+=" --fp8-recipe mxfp8 --fp8-format e4m3"
    TRAINING_PARAMS+=" --fp8-param-gather --reuse-grad-buf-for-mxfp8-param-ag"
    TRAINING_PARAMS+=" --overlap-grad-reduce --overlap-param-gather"
    TRAINING_PARAMS+=" --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS+=" --moe-router-padding-for-quantization"
elif [[ ${PR} == "nvfp4" ]]; then
    TRAINING_PARAMS+=" --fp4-recipe nvfp4 --fp4-format e2m1"
    TRAINING_PARAMS+=" --overlap-grad-reduce --overlap-param-gather"
    TRAINING_PARAMS+=" --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS+=" --moe-router-padding-for-quantization"
fi

# ========================== 1F1B overlapping ==================================
A2A_OVERLAP=${A2A_OVERLAP:-0}
if [[ ${A2A_OVERLAP} == 1 ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=32
    export NVTE_FWD_LAYERNORM_SM_MARGIN=20
    export NVTE_BWD_LAYERNORM_SM_MARGIN=20
    TRAINING_PARAMS+=" --delay-wgrad-compute --overlap-moe-expert-parallel-comm"
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVTE_FWD_LAYERNORM_SM_MARGIN=0
    export NVTE_BWD_LAYERNORM_SM_MARGIN=0
fi

# ========================== Extra CLI args ====================================
TRAINING_PARAMS+=" --recompute-granularity selective --recompute-modules moe_act mlp"
TRAINING_PARAMS+=" --cuda-graph-impl transformer_engine --cuda-graph-scope attn moe_router moe_preprocess --te-rng-tracker --cuda-graph-warmup-steps 0"
TRAINING_PARAMS+=" --moe-router-force-load-balancing"
TRAINING_PARAMS+=" --offload-optimizer-states"
TRAINING_PARAMS+=' --pipeline-model-parallel-layout Et*10|t*10|t*11L '

if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS+=" $@"
fi

# ========================== Profile (optional) ================================
PROFILE_CMD=""
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
fi

# ========================== Create output dirs ================================
LOGS_DIR="${OUTPUT_PATH}/logs"
mkdir -p "${LOGS_DIR}"

TIMESTAMP=$(date +'%y%m%d_%H%M%S')

# ========================== Launch with torchrun ==============================
set +e
if [[ ${DRY_RUN:-0} -eq 1 ]]; then
    echo "=== DRY RUN - torchrun command ==="
    echo "cd ${MEGATRON_PATH} && \\"
    echo "torchrun \\"
    echo "    --nproc_per_node=${NPROC_PER_NODE} \\"
    echo "    --nnodes=${NNODES} \\"
    echo "    --node_rank=${NODE_RANK} \\"
    echo "    --master_addr=${MASTER_ADDR} \\"
    echo "    --master_port=${MASTER_PORT} \\"
    echo "    ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"
    echo "=== End of DRY RUN ==="
else
    cd "${MEGATRON_PATH}"

    ${PROFILE_CMD} \
    torchrun \
        --nproc_per_node=${NPROC_PER_NODE} \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        ${TRAINING_SCRIPT_PATH} \
        ${TRAINING_PARAMS} \
        2>&1 | tee "${LOGS_DIR}/run_${TIMESTAMP}.log"
fi
set -e
