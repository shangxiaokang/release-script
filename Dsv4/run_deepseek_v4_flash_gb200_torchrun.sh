#!/bin/bash
# DeepSeek-V4-Flash GB200 torchrun launcher.
#
# Reference environment:
#   NGC Image: nvcr.io/nvidia/pytorch:26.04-py3
#   MCore: 3e8ce1f9df81e9b477927a7ed80aa0d5bb2034bb
#
# This script is the torchrun counterpart of run_deepseek_v4_flash_gb200_slurm.sh.
# It keeps the same DeepSeek-V4-Flash Megatron arguments, but starts training with
# torchrun instead of sbatch/srun. Run it inside the target container/allocation.
#
# Usage examples:
#   bash run_deepseek_v4_flash_gb200_torchrun.sh [extra Megatron args...]
#   DRY_RUN=1 bash run_deepseek_v4_flash_gb200_torchrun.sh
#   NNODES=32 NODE_RANK=0 MASTER_ADDR=<rank0-host> bash run_deepseek_v4_flash_gb200_torchrun.sh

set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
WORKSPACE=$(cd "$(dirname "${SCRIPT_PATH}")/../.." && pwd)
export WORKSPACE

# ---------------------------------------------------------------------------
# Runtime configuration.
# ---------------------------------------------------------------------------
export CLUSTER=${CLUSTER:-"lyris"}
export NGC_IMAGE=${NGC_IMAGE:-"nvcr.io/nvidia/pytorch:26.04-py3"}
export MEGATRON_PATH=${MEGATRON_PATH:-"/lustre/fsw/general_sa/xshang/DSv4/Megatron-LM"}
export TRAINING_SCRIPT_PATH=${TRAINING_SCRIPT_PATH:-"${MEGATRON_PATH}/pretrain_gpt.py"}
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-"0.13"}
export RUN_TIME=${RUN_TIME:-"00:15:00"}

# ---------------------------------------------------------------------------
# DeepSeek-V4-Flash GB200 launch overrides from run_deepseek_v4_flash_gb200.sh.
# ---------------------------------------------------------------------------
export MODEL=${MODEL:-"DeepSeek-V4-Flash"}
export RUN_NAME=${RUN_NAME:-"${MODEL}-benchmarking"}
export WANDB_PROJECT=${WANDB_PROJECT:-"DeepSeek-V4-Flash-GB200"}
export WANDB_TOGGLE=${WANDB_TOGGLE:-1}
export DATASET=${DATASET:-"Slimpajama"}

export SEQ_LEN=${SEQ_LEN:-4096}
export GBS=${GBS:-2048}
export MBS=${MBS:-1}
export TP=${TP:-1}
export EP=${EP:-64}
export CP=${CP:-1}
export PP=${PP:-2}
export VPP=${VPP:-4}
export NNODES=${NNODES:-32}
export NUM_LAYERS=${NUM_LAYERS:-43}
export MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-true}
export PRETRAIN=${PRETRAIN:-1}
export PR=${PR:-"mxfp8"}
export DISPATCHER=${DISPATCHER:-"hybridep"}
export SEGMENT=${SEGMENT:-$((EP * TP / 4))}
export BINDPCIE_PATH=${BINDPCIE_PATH:-""}

# HybridEP auto-detects topology via memory accessibility probing. Do not set
# NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN or USE_MNNVL here.
export NUM_OF_STAGES_DISPATCH_API=${NUM_OF_STAGES_DISPATCH_API:-10}
export NUM_OF_IN_FLIGHT_S2G_DISPATCH_API=${NUM_OF_IN_FLIGHT_S2G_DISPATCH_API:-8}

# ---------------------------------------------------------------------------
# Inlined lyris.conf values for this MODEL/DATASET.
# ---------------------------------------------------------------------------
export OUTPUT_PATH=${OUTPUT_PATH:-"${WORKSPACE}/output/mcore-benchmarking-v${MCORE_RELEASE_VERSION}/:${MODEL}-TP${TP}PP${PP}EP${EP}VPP${VPP}-MBS${MBS}GBS${GBS}"}
export DATA_PATH=${DATA_PATH:-"/lustre/fsw/general_sa/xshang/dataset/OpenWebText/openwebtext_dsv3_text_document"}
export LOAD_PATH=${LOAD_PATH:-"false"}
export N_TASKS_PER_NODE=${N_TASKS_PER_NODE:-4}
export NVLINK_DOMAIN_SIZE=${NVLINK_DOMAIN_SIZE:-72}

# ---------------------------------------------------------------------------
# Inlined ENV_VARS from DeepSeek-V4-Flash.yaml plus runtime env adjustments.
# ---------------------------------------------------------------------------
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-0}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True,graph_capture_record_stream_reuse:True"}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
export NVTE_NORM_FWD_USE_CUDNN=${NVTE_NORM_FWD_USE_CUDNN:-1}
export NVTE_NORM_BWD_USE_CUDNN=${NVTE_NORM_BWD_USE_CUDNN:-1}
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore"}
export NCCL_DEBUG=${NCCL_DEBUG:-"VERSION"}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=${NVTE_CUTEDSL_FUSED_GROUPED_MLP:-1}
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-128}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-0}
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-0}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache_${SLURM_NODEID:-0}"}

COMMENT=${COMMENT:-"v${MCORE_RELEASE_VERSION}"}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-"${MODEL}-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}PR${PR}-${COMMENT}"}
LOGS_PATH="${OUTPUT_PATH}/torchrun_logs"
mkdir -p "${LOGS_PATH}"

if [[ "${PR}" != "bf16" && "${PR}" != "fp8" && "${PR}" != "mxfp8" ]]; then
    echo "Error: PR must be one of bf16, fp8, or mxfp8. Current value: ${PR}" >&2
    exit 1
fi

TRAINING_PARAMS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size "${TP}"
    --pipeline-model-parallel-size "${PP}"
    --expert-model-parallel-size "${EP}"
    --context-parallel-size "${CP}"
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather

    --use-mcore-models
    --sequence-parallel
    --use-flash-attn
    --disable-bias-linear
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --train-samples 585937500
    --exit-duration-in-mins 220
    --no-save-optim
    --no-check-for-nan-in-loss-and-grad
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --manual-gc
    --manual-gc-interval 10

    --transformer-impl transformer_engine

    --seq-length "${SEQ_LEN}"
    --data-cache-path "${WORKSPACE}/data_cache"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model unsloth/DeepSeek-V3
    --data-path "${DATA_PATH}"
    --split 99,1,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 6

    --num-layers 43
    --hidden-size 4096
    --num-attention-heads 64
    --kv-channels 512
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 10000
    --make-vocab-size-divisible-by 3232
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention

    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --qk-layernorm

    --lr-decay-samples 584765624
    --lr-warmup-samples 1536000
    --lr-warmup-init 3.9e-7
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95
    --optimizer adam
    --muon-num-ns-steps 10

    --num-experts 256
    --moe-n-hash-layers 0
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 6
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-topk-scaling-factor 1.5
    --moe-router-score-function sqrtsoftplus
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-router-fusion

    --q-lora-rank 1024
    --qk-pos-emb-head-dim 64
    --v-head-dim 512
    --rotary-scaling-factor 4
    --mscale 1.0
    --mscale-all-dim 1.0
    --o-groups 8
    --o-lora-rank 1024

    --experimental-attention-variant dsv4_hybrid
    --csa-window-size 128
    --csa-compress-ratios "([0,0,4]+[128,4]*20)"
    --csa-compress-rotary-base 40000
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512
    --dsa-indexer-loss-coeff 1e-2
    --dsa-indexer-use-sparse-loss

    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --use-fused-mhc

    --eval-iters 32
    --eval-interval 200

    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --save "${OUTPUT_PATH}/checkpoints"
    --save-interval 500
    --dist-ckpt-strictness log_all

    --init-method-std 0.02

    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --logging-level 40
    --tensorboard-dir "${OUTPUT_PATH}/tensorboard"

    --bf16
    --enable-experimental
)

if [[ "${DISPATCHER}" == "alltoall" ]]; then
    TRAINING_PARAMS+=(--moe-token-dispatcher-type alltoall)
elif [[ "${DISPATCHER}" == "deepep" ]]; then
    TRAINING_PARAMS+=(--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep)
elif [[ "${DISPATCHER}" == "hybridep" ]]; then
    TRAINING_PARAMS+=(--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --moe-hybridep-num-sms 32)
else
    echo "Error: unsupported DISPATCHER=${DISPATCHER}" >&2
    exit 1
fi

if [[ "${PR}" == "fp8" ]]; then
    TRAINING_PARAMS+=(
        --fp8-recipe blockwise
        --fp8-format e4m3
        --fp8-param-gather
        --use-precision-aware-optimizer
        --main-grads-dtype fp32
        --main-params-dtype fp32
        --exp-avg-dtype bf16
        --exp-avg-sq-dtype bf16
        --moe-router-padding-for-fp8
    )
elif [[ "${PR}" == "mxfp8" ]]; then
    TRAINING_PARAMS+=(
        --fp8-recipe mxfp8
        --fp8-format e4m3
        --fp8-param-gather
        --reuse-grad-buf-for-mxfp8-param-ag
        --use-precision-aware-optimizer
        --main-grads-dtype fp32
        --main-params-dtype fp32
        --exp-avg-dtype bf16
        --exp-avg-sq-dtype bf16
        --moe-router-padding-for-quantization
    )
fi

if [[ "${SEQ_LEN}" -gt 4096 ]]; then
    TRAINING_PARAMS+=(--max-position-embeddings "${SEQ_LEN}")
fi

if [[ "${WANDB_TOGGLE}" == 1 ]]; then
    export WANDB_API_KEY=${WANDB_API_KEY:-0}
    TRAINING_PARAMS+=(--wandb-project "${WANDB_PROJECT}" --wandb-exp-name "${WANDB_EXP_NAME}")
fi

# Original entry script appends these flags after YAML-derived args.
TRAINING_PARAMS+=(
    --moe-router-force-load-balancing
    --pipeline-model-parallel-layout "Et*3|(tttttt|)*6ttttL"
    --logging-level 20
    --use-transformer-engine-op-fuser
    --moe-mlp-glu-interleave-size 32
    --moe-expert-rank-capacity-factor 1.5
    --moe-paged-stash
    --moe-paged-stash-page-size 64
    --moe-paged-stash-buffer-size-factor-cuda 1.2
    --moe-paged-stash-buffer-size-factor-cpu 0.0
    --moe-pad-experts-for-cuda-graph-inference
    --cuda-graph-impl transformer_engine
    --cuda-graph-scope attn moe_router moe_preprocess
)

if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS+=("$@")
fi

BINDPCIE_ARGS=()
if [[ -n "${BINDPCIE_PATH}" ]]; then
    read -r -a BINDPCIE_ARGS <<< "${BINDPCIE_PATH}"
fi

# torchrun topology. Values are inferred from Slurm when available, and can be
# overridden explicitly for non-Slurm launches.
export NNODES=${TORCHRUN_NNODES:-${NNODES}}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_NTASKS_PER_NODE:-${N_TASKS_PER_NODE}}}
export NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
export MASTER_PORT=${MASTER_PORT:-29500}

if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
        MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')
    else
        MASTER_ADDR=127.0.0.1
    fi
fi
export MASTER_ADDR

TORCHRUN_CMD=(
    torchrun
    --nnodes "${NNODES}"
    --nproc-per-node "${NPROC_PER_NODE}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    "${TRAINING_SCRIPT_PATH}"
    "${TRAINING_PARAMS[@]}"
)

if [[ ${#BINDPCIE_ARGS[@]} -gt 0 ]]; then
    TORCHRUN_CMD=("${BINDPCIE_ARGS[@]}" "${TORCHRUN_CMD[@]}")
fi

printf -v TORCHRUN_CMD_QUOTED '%q ' "${TORCHRUN_CMD[@]}"

print_dry_run() {
    echo "=== Reference environment ==="
    echo "NGC Image: ${NGC_IMAGE}"
    echo "MCore: 3e8ce1f9df81e9b477927a7ed80aa0d5bb2034bb"
    echo
    echo "=== torchrun topology ==="
    echo "NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE} NODE_RANK=${NODE_RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
    echo
    echo "=== torchrun command ==="
    echo "${TORCHRUN_CMD_QUOTED}"
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    print_dry_run
    exit 0
fi

"${TORCHRUN_CMD[@]}" 2>&1 | tee "${LOGS_PATH}/${SLURM_JOB_ID:-torchrun_${NODE_RANK}}.log"
