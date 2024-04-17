#!/bin/bash
#shellcheck disable=SC2206
#SBATCH --partition=llm_s
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --output=./logs/%A_%x.out

set -e

export HOME=/mnt/petrelfs/huqinghao/xzyao
export GPUS_PER_NODE=8

# <<< init conda >>>
# Set the path according to your conda environment
export PATH=/mnt/hwfile/huqinghao/miniconda3/bin:$PATH
source /mnt/hwfile/huqinghao/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# <<< init conda end >>>

cd /mnt/petrelfs/huqinghao/xzyao/code/vllm
# spin up vllm

python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-70b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 --delta-modules delta-1=.idea/models/vicuna-7b-4b0.75s-tp_2-1 delta-2=.idea/models/vicuna-7b-4b0.75s-tp_2-2 delta-3=.idea/models/vicuna-7b-4b0.75s-tp_2-3 delta-4=.idea/models/vicuna-7b-4b0.75s-tp_2-4 delta-5=.idea/models/vicuna-7b-4b0.75s-tp_2-5 delta-6=.idea/models/vicuna-7b-4b0.75s-tp_2-6 delta-7=.idea/models/vicuna-7b-4b0.75s-tp_2-7 delta-8=.idea/models/vicuna-7b-4b0.75s-tp_2-8 --max-deltas 1 --tensor-parallel-size 8 --enforce-eager & \
python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/distribution=uniform,ar=1.0,duration=30.0.jsonl --output .artifact/benchmarks/results