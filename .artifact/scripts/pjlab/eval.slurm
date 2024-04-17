#!/bin/bash
#SBATCH --partition=llm_s
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --output=./logs/%A_%x.out
#SBATCH --mem-per-cpu=2GB

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

UNOPTIMIZED_DELTA=1 python -m vllm.entrypoints.openai.api_server --model .idea/full_models/llama-70b --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 --delta-modules delta-1=.idea/models/llama2-chat-70b.4b75s128g-unopt-1 delta-2=.idea/models/llama2-chat-70b.4b75s128g-unopt-2 delta-3=.idea/models/llama2-chat-70b.4b75s128g-unopt-3 delta-4=.idea/models/llama2-chat-70b.4b75s128g-unopt-4 delta-5=.idea/models/llama2-chat-70b.4b75s128g-unopt-5 delta-6=.idea/models/llama2-chat-70b.4b75s128g-unopt-6 delta-7=.idea/models/llama2-chat-70b.4b75s128g-unopt-7 delta-8=.idea/models/llama2-chat-70b.4b75s128g-unopt-8 --max-deltas 1 --tensor-parallel-size 8 --enforce-eager

# srun --jobid 3653359 --interactive /bin/bash
# unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

# python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/distribution=uniform,ar=1.0,duration=30.0.jsonl --output .artifact/benchmarks/results