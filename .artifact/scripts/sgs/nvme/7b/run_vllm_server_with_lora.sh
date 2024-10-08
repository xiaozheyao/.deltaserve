python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --enable-lora --disable-log-requests --gpu-memory-utilization 0.85 \
--lora-modules lora-1=/mnt/scratch/xiayao/cache/models/lora/llama-7b-hf-lora-r16-1 lora-2=/mnt/scratch/xiayao/cache/models/lora/llama-7b-hf-lora-r16-2 lora-3=/mnt/scratch/xiayao/cache/models/lora/llama-7b-hf-lora-r16-3 lora-4=/mnt/scratch/xiayao/cache/models/lora/llama-7b-hf-lora-r16-4 \
--max-loras 4 --max-cpu-loras 32 --tensor-parallel-size 1 \
--enforce-eager