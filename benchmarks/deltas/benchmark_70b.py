import os
import torch
from vllm import LLM, SamplingParams
from vllm.delta.request import DeltaRequest

tp_size = int(os.environ.get("TP_SIZE", "8"))

print(f"Benchmarking with tensor parallel size={tp_size}")
llm = LLM(
    model=".idea/full_models/llama-70b",
    enable_delta=True,
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    gpu_memory_utilization=0.75,
    max_context_len_to_capture=64,
    max_model_len=64,
    max_deltas=1,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=64,
    seed=42,
)

delta_path = f".idea/models/llama2-chat-70b.4b75s128g-bitblas-unopt-1"

prompts = [
    "USER: Write a letter to the city council to complain the noise in the city.\nASSISTANT:",
]
outputs = llm.generate(
    prompts, sampling_params, delta_request=DeltaRequest("vicuna", 1, delta_path)
)
print(f"with delta: {outputs[0].outputs[0].text}")
print(outputs)
