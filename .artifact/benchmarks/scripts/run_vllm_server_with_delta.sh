python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 \
--delta-modules delta-1=.idea/models/vicuna-7b-4b0.75s-tp_2-1 delta-2=.idea/models/vicuna-7b-4b0.75s-tp_2-2 delta-3=.idea/models/vicuna-7b-4b0.75s-tp_2-3 delta-4=.idea/models/vicuna-7b-4b0.75s-tp_2-4 delta-5=.idea/models/vicuna-7b-4b0.75s-tp_2-5 delta-6=.idea/models/vicuna-7b-4b0.75s-tp_2-6 delta-7=.idea/models/vicuna-7b-4b0.75s-tp_2-7 delta-8=.idea/models/vicuna-7b-4b0.75s-tp_2-8 \
--max-deltas 1 --tensor-parallel-size 2 \
--enforce-eager