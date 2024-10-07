USE_MARLIN=1 python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.5 --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 \
--max-deltas 1 --max-cpu-deltas 32 --tensor-parallel-size 2 --max-logprobs 100 \
--enforce-eager