USE_MARLIN=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-13b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 \
--delta-modules lmsys/vicuna-13b-v1.5=/scratch/xiayao/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g/ \
--max-deltas 1 --max-cpu-deltas 32 --tensor-parallel-size 2 --max-logprobs 100 \
--enforce-eager