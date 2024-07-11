USE_MARLIN=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 \
--delta-modules lmsys/vicuna-7b-v1.5=/scratch/xiayao/models/7b/4bit/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g.1 \
--max-deltas 1 --max-cpu-deltas 32 --tensor-parallel-size 2 --max-logprobs 100 \
--enforce-eager