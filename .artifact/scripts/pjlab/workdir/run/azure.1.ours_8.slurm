#!/bin/bash
#SBATCH --partition=llm_s
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=./logs/%A_%x.out
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=8GB
#SBATCH --exclude=HOST-10-140-60-14
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
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
# spin up vllm
ray stop
apptainer run --nv \
--home /mnt/petrelfs/huqinghao/xzyao/:/home/xiayao \
--bind /mnt/petrelfs/huqinghao/xzyao/code/vllm:/vllm \
--bind /mnt/petrelfs/huqinghao/xzyao/code/triteia:/triteia \
--env PYTHONPATH=/vllm:/triteia \
--env RAY_DEDUP_LOGS=1 \
--env LOG_LEVEL=ERROR \
--env USE_MARLIN=1 \
--workdir /vllm \
/mnt/petrelfs/huqinghao/xzyao/images/deltaserve.sif \
python3 /vllm/vllm/entrypoints/openai/api_server.py --model /vllm/.idea/full_models/Llama-2-13b-hf --disable-log-requests --gpu-memory-utilization 0.85 --delta-modules delta-1=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-1 delta-2=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-2 delta-3=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-3 delta-4=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-4 delta-5=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-5 delta-6=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-6 delta-7=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-7 delta-8=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-8 delta-9=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-9 delta-10=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-10 delta-11=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-11 delta-12=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-12 delta-13=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-13 delta-14=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-14 delta-15=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-15 delta-16=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-16 delta-17=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-17 delta-18=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-18 delta-19=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-19 delta-20=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-20 delta-21=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-21 delta-22=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-22 delta-23=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-23 delta-24=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-24 delta-25=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-25 delta-26=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-26 delta-27=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-27 delta-28=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-28 delta-29=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-29 delta-30=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-30 delta-31=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-31 delta-32=/vllm/.idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g-32 --tensor-parallel-size 4 --enforce-eager --max-deltas 8 --enable-delta & python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/models=32,distribution=azure,ar=1.0,duration=300.0.jsonl  --output .artifact/benchmarks/results/pjlab & wait

# srun --jobid 3666603 --interactive /bin/bash
# unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

# python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_auto/distribution=uniform,ar=1.0,duration=30.0.jsonl --output .artifact/benchmarks/results