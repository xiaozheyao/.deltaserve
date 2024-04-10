import os
import torch
from vllm import LLM, SamplingParams
from vllm.delta.request import DeltaRequest

tp_size = int(os.environ.get("TP_SIZE", "1"))
bits = int(os.environ.get("BITS", "4"))

# print(f"Benchmarking with tensor parallel size={tp_size} and bitwidth={bits}")
torch.cuda.nvtx.range_push(f"tp_size = {tp_size}")
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_delta=True,
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    max_context_len_to_capture=64,
    max_model_len=32,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=32,
)
delta_path = f".idea/models/vicuna-7b-4b0.75s-decom-1-optimize_io-tp_{tp_size}"

prompts = [
    "USER: Who is Alan Turing?\nASSISTANT:",
    # "USER: Who is Alan Turing?\n ASSISTANT: ",
]
outputs = llm.generate(
    prompts,
    sampling_params,
)
print(f"without delta: {outputs[0].outputs[0].text}")
print(outputs)
outputs = llm.generate(prompts,
                       sampling_params,
                       delta_request=DeltaRequest("vicuna", 1, delta_path))
torch.cuda.nvtx.range_pop()
print(f"with delta: {outputs[0].outputs[0].text}")
print(outputs)
