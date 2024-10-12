python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --host 0.0.0.0 --enable-lora --disable-log-requests --gpu-memory-utilization 0.85 --lora-modules lora-1=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-1 lora-2=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-2 lora-3=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-3 lora-4=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-4 lora-5=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-5 lora-6=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-6 lora-7=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-7 lora-8=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-8 lora-9=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-9 lora-10=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-10 lora-11=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-11 lora-12=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-12 lora-13=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-13 lora-14=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-14 lora-15=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-15 lora-16=/scratch/xiayao/models/lora/llama-7b-hf-lora-r32-16 --max-loras 8 --max-cpu-loras 64 --tensor-parallel-size 2 --enforce-eager --port 8081 --max-lora-rank 32