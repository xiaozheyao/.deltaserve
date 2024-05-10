#!/bin/bash
#SBATCH --partition=llm_s
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=./logs/%A_%x.out
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8GB

set -e

export HOME=/mnt/petrelfs/huqinghao
export GPUS_PER_NODE=4

# <<< init conda >>>
# Set the path according to your conda environment
export PATH=/mnt/hwfile/huqinghao/miniconda3/bin:$PATH
source /mnt/hwfile/huqinghao/miniconda3/etc/profile.d/conda.sh
conda activate base
conda activate vllm
which python

# <<< init conda end >>>

cd /mnt/petrelfs/huqinghao/xzyao/code/vllm
# spin up vllm
ray stop
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
export NUMEXPR_MAX_THREADS=128
apptainer run --nv \
    --home /mnt/petrelfs/huqinghao/xzyao/:/home/xiayao \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/vllm:/vllm \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/triteia:/triteia \
    --env PYTHONPATH=/vllm:/triteia \
    --env CUDA_LAUNCH_BLOCKING=1 \
    --env TVM_TARGET=nvidia/nvidia-a100 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env NUMEXPR_MAX_THREADS=128 \
    --env USE_BITBLAS=1 \
    --env BITBLAS_TARGET=nvidia/nvidia-a100 \
    --env RAY_DEDUP_LOGS=0 \
    --workdir /vllm \
    /mnt/petrelfs/huqinghao/xzyao/images/deltaserve.sif \
    python3 /vllm/vllm/entrypoints/openai/api_server.py --model /vllm/.idea/full_models/llama-70b --enable-delta --disable-log-requests --gpu-memory-utilization 0.95 --delta-modules delta-1=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-1 delta-2=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-2 delta-3=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-3 delta-4=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-4 delta-5=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-5 delta-6=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-6 delta-7=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-7 delta-8=/vllm/.idea/models/70b/llama2-chat-70b.4b75s128g-bitblas-tp_4-8 --max-deltas 4 --tensor-parallel-size 4 --enforce-eager --enable-prefetch --max-model-len 2048 & python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/distribution=uniform,ar=0.5,duration=60.0.jsonl --output .artifact/benchmarks/results & wait

# srun --jobid 3666621 --interactive /bin/bash

# python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/distribution=uniform,ar=0.5,duration=60.0.jsonl --output .artifact/benchmarks/results