rm -rf /opt/megatron-lm
apt-get update
apt-get install -y --no-install-recommends \
        sudo gdb bash-builtins git zsh autojump tmux curl gettext libfabric-dev
wget https://github.com/mikefarah/yq/releases/download/v4.27.5/yq_linux_arm64 -O /usr/bin/yq
chmod +x /usr/bin/yq
apt-get clean
rm -rf /var/lib/apt/lists/*

unset PIP_CONSTRAINT && pip install --no-cache-dir \
    debugpy dm-tree torch_tb_profiler einops wandb \
    sentencepiece tokenizers transformers==4.57.1 torchvision ftfy modelcards datasets tqdm pydantic \
    nvidia-pytriton py-spy yapf darker \
    tiktoken flask-restful \
    nltk wrapt pytest pytest_asyncio pytest-cov pytest_mock pytest-random-order \
    black==24.4.2 isort==5.13.2 flake8==7.1.0 pylint==3.2.6 coverage mypy \
    one-logger --index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple \
    setuptools==69.5.1 nvidia-cutlass-dsl==4.4.2

export TE_COMMIT="01aef4fc721bd12fd09cd56d53a314aee1b953d6"
pip install --no-cache-dir flash-attn-4==4.0.0b4 nvidia-mathdx==25.1.1 && \
    unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch \
    pip install --no-build-isolation --no-cache-dir \
    "git+https://github.com/NVIDIA/TransformerEngine.git@${TE_COMMIT}"

git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    pushd DeepEP && git checkout 1b8f467965bb818bf2f6511e06993f5607e1721f && \
    TORCH_CUDA_ARCH_LIST="10.0" pip install --no-build-isolation . && \
    popd

# Fast Hadamard Transform (used by DSA indexer)
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git && \
    pushd fast-hadamard-transform && \
    pip install --no-build-isolation . && popd

# Emerging-Optimizers (Muon)
git clone https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git && \
    pushd Emerging-Optimizers && \
    pip install --no-build-isolation . && popd

# FlashMLA (DSA kernels)
git clone --branch nv_dev https://github.com/deepseek-ai/FlashMLA.git && \
    pushd FlashMLA && \
    FLASH_MLA_DISABLE_SM90=1 \
    NVCC_THREADS=16 \
    CFLAGS="-I/usr/local/cuda/include/cccl" \
    CXXFLAGS="-I/usr/local/cuda/include/cccl" \
    pip install --no-build-isolation . && popd
pushd /opt
cp -r /lustre/fsw/general_sa/xshang/DSv4/cudnn_frontend .
# git clone ssh://git@gitlab-master.nvidia.com:12051/cudnn/cudnn_frontend.git
pushd cudnn_frontend && git checkout devtech/dsa
pip install --no-build-isolation .
pip install --force-reinstall 'nvidia-cutlass-dsl[cu13]==4.4.2'
popd
popd
