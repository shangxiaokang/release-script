#!/bin/bash
#
# NCCL Send/Recv: cross-node GPU→CPU (HBM vs Pinned vs EGM)
# 4 nodes × 1 rank (GPU0) per node = 4 ranks total
#
# Usage:
#   sbatch run_send_recv.sh
#

#SBATCH -A general_sa
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -p gb300-backfill
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH -J nccl_sr
#SBATCH --output=/home/xshang/workspace/my-script/test_egm/nccl_send_recv_%j.out

IMAGE=/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB300-HybridEP-V2.sqsh
WORKDIR=/home/xshang/workspace/my-script/test_egm
SRC=test_nccl_send_recv.cu
BIN=test_nccl_send_recv

# ---------- Build ----------
echo "=== Building ${BIN} ==="
srun -N1 -n1 \
    --container-image=${IMAGE} \
    --container-writable \
    --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/general_sa/:/lustre/fsw/general_sa/ \
    bash -c "
        cd ${WORKDIR} && \
        MPI_HOME=\$(dirname \$(dirname \$(which mpicc))) && \
        nvcc -o ${BIN} ${SRC} \
            -lnccl -lmpi -lcuda -lnuma -std=c++17 -arch=native \
            -I\${MPI_HOME}/include -L\${MPI_HOME}/lib \
            2>&1 && echo '=== BUILD OK ==='
    "

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# ---------- Run ----------
echo ""
echo "=== Running NCCL Send/Recv: cross-node GPU→CPU ==="
echo "=== 4 nodes × 1 GPU(GPU0) = 4 ranks ==="
echo ""

export NCCL_DEBUG=WARN

srun -N4 --ntasks-per-node=1 --gpus-per-node=1 \
    --container-image=${IMAGE} \
    --container-writable \
    --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/general_sa/:/lustre/fsw/general_sa/ \
    ${WORKDIR}/${BIN}

echo ""
echo "=== Done ==="
