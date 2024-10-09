python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 --swap-modules delta-1=/scratch/xiayao/models/full/vicuna-7b-v1.5-1 delta-2=/scratch/xiayao/models/full/vicuna-7b-v1.5-2 delta-3=/scratch/xiayao/models/full/vicuna-7b-v1.5-3 delta-4=/scratch/xiayao/models/full/vicuna-7b-v1.5-4 delta-5=/scratch/xiayao/models/full/vicuna-7b-v1.5-5 delta-6=/scratch/xiayao/models/full/vicuna-7b-v1.5-6 delta-7=/scratch/xiayao/models/full/vicuna-7b-v1.5-7 delta-8=/scratch/xiayao/models/full/vicuna-7b-v1.5-8 delta-9=/scratch/xiayao/models/full/vicuna-7b-v1.5-9 delta-10=/scratch/xiayao/models/full/vicuna-7b-v1.5-10 delta-11=/scratch/xiayao/models/full/vicuna-7b-v1.5-11 delta-12=/scratch/xiayao/models/full/vicuna-7b-v1.5-12 delta-13=/scratch/xiayao/models/full/vicuna-7b-v1.5-13 delta-14=/scratch/xiayao/models/full/vicuna-7b-v1.5-14 delta-15=/scratch/xiayao/models/full/vicuna-7b-v1.5-15 delta-16=/scratch/xiayao/models/full/vicuna-7b-v1.5-16 --tensor-parallel-size 2 --enforce-eager --max-deltas 0 --enable-swap --max-swap-slots 1 --max-cpu-models 4