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
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
# spin up vllm
ray stop
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
    python3 /vllm/vllm/entrypoints/openai/api_server.py --model /vllm/.idea/full_models/Llama-2-13b-hf --disable-log-requests --gpu-memory-utilization 0.85 --swap-modules /vllm/.idea/full_models/Llama-2-13b-hf=/vllm/.idea/full_models/Llama-2-13b-hf delta-1=/vllm/.idea/full_models/Llama-2-13b-hf-1 delta-2=/vllm/.idea/full_models/Llama-2-13b-hf-2 delta-3=/vllm/.idea/full_models/Llama-2-13b-hf-3 delta-4=/vllm/.idea/full_models/Llama-2-13b-hf-4 delta-5=/vllm/.idea/full_models/Llama-2-13b-hf-5 delta-6=/vllm/.idea/full_models/Llama-2-13b-hf-6 delta-7=/vllm/.idea/full_models/Llama-2-13b-hf-7 delta-8=/vllm/.idea/full_models/Llama-2-13b-hf-8 delta-9=/vllm/.idea/full_models/Llama-2-13b-hf-9 delta-10=/vllm/.idea/full_models/Llama-2-13b-hf-10 delta-11=/vllm/.idea/full_models/Llama-2-13b-hf-11 delta-12=/vllm/.idea/full_models/Llama-2-13b-hf-12 delta-13=/vllm/.idea/full_models/Llama-2-13b-hf-13 delta-14=/vllm/.idea/full_models/Llama-2-13b-hf-14 delta-15=/vllm/.idea/full_models/Llama-2-13b-hf-15 delta-16=/vllm/.idea/full_models/Llama-2-13b-hf-16 delta-17=/vllm/.idea/full_models/Llama-2-13b-hf-17 delta-18=/vllm/.idea/full_models/Llama-2-13b-hf-18 delta-19=/vllm/.idea/full_models/Llama-2-13b-hf-19 delta-20=/vllm/.idea/full_models/Llama-2-13b-hf-20 delta-21=/vllm/.idea/full_models/Llama-2-13b-hf-21 delta-22=/vllm/.idea/full_models/Llama-2-13b-hf-22 delta-23=/vllm/.idea/full_models/Llama-2-13b-hf-23 delta-24=/vllm/.idea/full_models/Llama-2-13b-hf-24 delta-25=/vllm/.idea/full_models/Llama-2-13b-hf-25 delta-26=/vllm/.idea/full_models/Llama-2-13b-hf-26 delta-27=/vllm/.idea/full_models/Llama-2-13b-hf-27 delta-28=/vllm/.idea/full_models/Llama-2-13b-hf-28 delta-29=/vllm/.idea/full_models/Llama-2-13b-hf-29 delta-30=/vllm/.idea/full_models/Llama-2-13b-hf-30 delta-31=/vllm/.idea/full_models/Llama-2-13b-hf-31 delta-32=/vllm/.idea/full_models/Llama-2-13b-hf-32 --tensor-parallel-size 4 --enforce-eager --max-deltas 0 --enable-swap --max-swap-slots 8 --max-cpu-models 18 & python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/models=32,distribution=uniform,ar=6.0,duration=30.0.jsonl --output .artifact/benchmarks/results & wait
