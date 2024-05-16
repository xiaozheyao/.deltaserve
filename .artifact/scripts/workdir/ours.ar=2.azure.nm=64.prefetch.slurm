#!/bin/bash
#SBATCH --partition=llm_s
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=./logs/%A_%x.out
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=2:00:00

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
    /mnt/petrelfs/huqinghao/xzyao/images/vllm-0.4.1.sif \
    python3 /vllm/vllm/entrypoints/openai/api_server.py --model /vllm/.idea/full_models/Llama-2-13b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 --delta-modules delta-1=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-1 delta-2=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-2 delta-3=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-3 delta-4=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-4 delta-5=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-5 delta-6=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-6 delta-7=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-7 delta-8=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-8 delta-9=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-9 delta-10=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-10 delta-11=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-11 delta-12=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-12 delta-13=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-13 delta-14=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-14 delta-15=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-15 delta-16=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-16 delta-17=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-17 delta-18=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-18 delta-19=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-19 delta-20=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-20 delta-21=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-21 delta-22=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-22 delta-23=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-23 delta-24=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-24 delta-25=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-25 delta-26=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-26 delta-27=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-27 delta-28=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-28 delta-29=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-29 delta-30=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-30 delta-31=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-31 delta-32=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-32 delta-33=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-33 delta-34=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-34 delta-35=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-35 delta-36=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-36 delta-37=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-37 delta-38=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-38 delta-39=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-39 delta-40=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-40 delta-41=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-41 delta-42=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-42 delta-43=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-43 delta-44=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-44 delta-45=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-45 delta-46=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-46 delta-47=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-47 delta-48=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-48 delta-49=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-49 delta-50=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-50 delta-51=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-51 delta-52=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-52 delta-53=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-53 delta-54=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-54 delta-55=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-55 delta-56=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-56 delta-57=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-57 delta-58=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-58 delta-59=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-59 delta-60=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-60 delta-61=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-61 delta-62=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-62 delta-63=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-63 delta-64=/vllm/.idea/models/13b/lmsys.vicuna-13b-v1.5.4b75s128g-64 --max-deltas 24 --tensor-parallel-size 4 --enforce-eager --enable-prefetch & python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/models=64,distribution=azure,ar=2.0,duration=30.0.jsonl --output .artifact/benchmarks/results & wait

# srun --jobid 3666621 --interactive /bin/bash

# python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/distribution=uniform,ar=0.5,duration=60.0.jsonl --output .artifact/benchmarks/results