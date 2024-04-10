python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-delta \
--delta-modules vicuna-7b-1=.idea/models/vicuna-7b-4b0.75s-optimize_io-tp_2-1 vicuna-7b-2=.idea/models/vicuna-7b-4b0.75s-optimize_io-tp_2-2 vicuna-7b-3=.idea/models/vicuna-7b-4b0.75s-optimize_io-tp_2-3 vicuna-7b-4=.idea/models/vicuna-7b-4b0.75s-optimize_io-tp_2-4 \
--max-deltas 4 --tensor-parallel-size 2 \
--enforce-eager