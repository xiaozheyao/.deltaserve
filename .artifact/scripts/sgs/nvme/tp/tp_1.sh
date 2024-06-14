USE_MARLIN=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-delta --max-model-len 2048 --disable-log-requests --gpu-memory-utilization 0.85 \
--delta-modules delta-1=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-1 delta-2=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-2 delta-3=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-3 delta-4=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-4 delta-5=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-5 delta-6=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-6 delta-7=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-7 delta-8=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-8 delta-9=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-9 delta-10=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-10 delta-11=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-11 delta-12=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-12 delta-13=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-13 delta-14=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-14 delta-15=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-15 delta-16=/scratch/xiayao/models/7b/4bit-tp1/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g-16  \
--max-deltas 2 --max-cpu-deltas 32 --tensor-parallel-size 1 \
--enforce-eager