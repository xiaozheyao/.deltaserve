UNOPTIMIZED_DELTA=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.95 \
--delta-modules delta-1=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-1 delta-2=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-2 delta-3=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-3 delta-4=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-4 delta-5=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-5 delta-6=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-6 delta-7=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-7 delta-8=/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-8 \
--max-deltas 1 --max-cpu-deltas 1 --tensor-parallel-size 2 \
--enforce-eager