python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 \
--swap-modules vicuna-7b-1=.idea/models/vicuna-7b-v1.5-1 vicuna-7b-2=vicuna-7b-v1.5-2 vicuna-7b-3=vicuna-7b-v1.5-3 vicuna-7b-4=vicuna-7b-v1.5-4 vicuna-7b-5=vicuna-7b-v1.5-5 vicuna-7b-6=vicuna-7b-v1.5-6 vicuna-7b-7=vicuna-7b-v1.5-7 vicuna-7b-8=vicuna-7b-v1.5-8 \
--max-deltas 2 --tensor-parallel-size 2 \
--enforce-eager