#!/bin/bash
#SBATCH --partition=llm_s
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=./logs/%A_%x.out
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodelist=HOST-10-140-60-5

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

cd /mnt/petrelfs/huqinghao/xzyao/images

apptainer run --nv \
    --home /home/xzyao:/home/xiayao \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/vllm:/vllm \
    deltaserve.sif \
    python3 /workspace/triteia/triteia/tools/convert_gptq_to_bitblas.py --ckpt /vllm/.idea/models/llama2-chat-70b.4b75s128g-bitblas-1/deltazip-compressed.safetensors --output /vllm/.idea/models/llama2-chat-70b.4b75s128g-bitblas-1/bitblas.safetensors --bitwidth 4
    
apptainer run --nv \
    --home /home/xzyao:/home/xiayao \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/vllm:/vllm \
    --workdir /vllm \
    /mnt/petrelfs/huqinghao/xzyao/images/deltaserve.sif \
    python3 /vllm/vllm/tools/optimize_io_bitblas.py --input /vllm/.idea/models/llama2-chat-70b.4b75s128g-bitblas-tp_8-1 --tp-size 8

apptainer run --nv \
    --home /home/xzyao:/home/xiayao \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/vllm:/vllm \
    --workdir /vllm \
    /mnt/petrelfs/huqinghao/xzyao/images/deltaserve.sif \
    ls /vllm
    
apptainer shell --nv \
    --home /mnt/petrelfs/huqinghao/xzyao/:/home/xiayao \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/vllm:/vllm \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/deltazip:/deltazip \
    --bind /mnt/petrelfs/huqinghao/xzyao/code/triteia:/triteia \
    --env PYTHONPATH=/vllm:/triteia \
    --env CUDA_LAUNCH_BLOCKING=1 \
    --env TVM_TARGET=nvidia/nvidia-a100 \
    --env BITBLAS_TARGET=nvidia/nvidia-a100 \
    --env RAY_DEDUP_LOGS=0 \
    --workdir /vllm \
    /mnt/petrelfs/huqinghao/xzyao/images/deltaserve.sif


python3 vllm/tools/optimize_io_bitblas.py --input .idea/models/llama2-chat-70b.4b75s128g-bitblas-unopt-1 --tp-size 8

BITWIDTH=4 TP_SIZE=8 USE_BITBLAS=1 python3 benchmarks/deltas/benchmark_70b.py

python3 triteia/tools/pre_compile.py

sbatch .artifact/scripts/pjlab/13b_ours_x4_plus.slurm