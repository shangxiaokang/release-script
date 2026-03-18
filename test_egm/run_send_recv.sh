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
#SBATCH -p gb300
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --segment=4
#SBATCH --time=0:30:00
#SBATCH -J nccl_sr
#SBATCH --output=/home/xshang/my-script/test_egm/nccl_send_recv_%j.out

IMAGE=/lustre/fsw/general_sa/xshang/sqsh/Pytorch-2512-GB300-HybridEP-V2.sqsh
WORKDIR=/home/xshang/my-script/test_egm
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

# Run the actual test with IB monitor inside the same container
echo "=== Running NCCL test + IB monitor (same container) ==="
srun -N4 --ntasks-per-node=1 --mpi=pmix \
    --container-image=${IMAGE} \
    --container-writable \
    --container-mounts=/home/xshang:/home/xshang,/lustre/fsw/general_sa/:/lustre/fsw/general_sa/ \
    bash -c "
        LOGFILE=${WORKDIR}/IBStream_\$(hostname)_\${SLURM_JOB_ID}.log

        # Start IB monitor in background
        (
            echo \"=== IB Monitor on \$(hostname) started at \$(date) ===\" > \${LOGFILE}
            echo \"Timestamp, Port, RxBytes, TxBytes, RxPkts, TxPkts\" >> \${LOGFILE}
            while true; do
                for dev in /sys/class/infiniband/*/ports/*/counters; do
                    PORT=\$(echo \${dev} | grep -oP 'infiniband/\K[^/]+')
                    PORT_NUM=\$(echo \${dev} | grep -oP 'ports/\K[0-9]+')
                    RX=\$(cat \${dev}/port_rcv_data 2>/dev/null || echo 0)
                    TX=\$(cat \${dev}/port_xmit_data 2>/dev/null || echo 0)
                    RX_PKT=\$(cat \${dev}/port_rcv_packets 2>/dev/null || echo 0)
                    TX_PKT=\$(cat \${dev}/port_xmit_packets 2>/dev/null || echo 0)
                    echo \"\$(date +%H:%M:%S.%3N), \${PORT}/\${PORT_NUM}, \${RX}, \${TX}, \${RX_PKT}, \${TX_PKT}\" >> \${LOGFILE}
                done
                sleep 0.1
            done
        ) &
        IB_PID=\$!

        # Run NCCL test
        ${WORKDIR}/${BIN}
        TEST_RC=\$?

        # Stop IB monitor
        kill \${IB_PID} 2>/dev/null
        wait \${IB_PID} 2>/dev/null
        echo \"=== IB log: \${LOGFILE} ===\"

        exit \${TEST_RC}
    "

echo ""
echo "=== IB logs: ${WORKDIR}/IBStream_*_\${SLURM_JOB_ID}.log ==="
echo "=== Done ==="
