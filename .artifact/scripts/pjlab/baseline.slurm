#!/bin/bash
#SBATCH --partition=llm_s
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --output=./logs/%A_%x.out
#SBATCH --mem-per-cpu=8GB

set -e

export HOME=/mnt/petrelfs/huqinghao
export GPUS_PER_NODE=8

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

apptainer run --nv \
    --home /mnt/petrelfs/huqinghao/xzyao/:/home/xiayao \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/vllm:/vllm \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/triteia:/triteia \
    --env PYTHONPATH=/vllm:/triteia \
    --env CUDA_LAUNCH_BLOCKING=1 \
    --env TVM_TARGET=nvidia/nvidia-a100 \
    --env BITBLAS_TARGET=nvidia/nvidia-a100 \
    --env RAY_DEDUP_LOGS=0 \
    --workdir /vllm \
    /mnt/petrelfs/huqinghao/xzyao/images/deltaserve.sif \
    python3 /vllm/vllm/entrypoints/openai/api_server.py --model /vllm/.idea/full_models/llama-70b --disable-log-requests --gpu-memory-utilization 0.85 --swap-modules delta-1=/vllm/.idea/full_models/llama-70b-1 delta-2=/vllm/.idea/full_models/llama-70b-2 delta-3=/vllm/.idea/full_models/llama-70b-3 delta-4=/vllm/.idea/full_models/llama-70b-4 delta-5=/vllm/.idea/full_models/llama-70b-5 delta-6=/vllm/.idea/full_models/llama-70b-6 delta-7=/vllm/.idea/full_models/llama-70b-7 delta-8=/vllm/.idea/full_models/llama-70b-8 --tensor-parallel-size 8 --enforce-eager --max-deltas 0

# srun --jobid 3666603 --interactive /bin/bash
# unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

# python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/distribution=uniform,ar=1.0,duration=30.0.jsonl --output .artifact/benchmarks/results