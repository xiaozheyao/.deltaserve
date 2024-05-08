USE_BITBLAS=1 BITWIDTH=4 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-13b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.95 \
--delta-modules delta-1=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.1 delta-2=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.2 delta-3=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.3 delta-4=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.4 delta-5=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.5 delta-6=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.6 delta-7=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.7 delta-8=/scratch/xiayao/models/13b/4bit/13b.4b75s.tp_2.8 \
--max-deltas 3 --max-cpu-deltas 32 --tensor-parallel-size 2 \
--enforce-eager --max-model-len 2048 --enable-prefetch --scheduler-policy deltaserve