import os
from vllm import LLM, SamplingParams
from vllm.swap.request import SwapRequest

tp_size = int(os.environ.get("TP_SIZE", "1"))
use_bitblas = os.environ.get("USE_BITBLAS", "0") == "1"
os.environ["NUMEXPR_MAX_THREADS"] = "32"

print(f"Benchmarking with tensor parallel size={tp_size}")

llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    enable_delta=True,
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    gpu_memory_utilization=0.95,
    max_context_len_to_capture=64,
    max_model_len=64,
    max_deltas=0,
    max_swap_slots=1,  # swap==1 means only the original model will be swapped
    enable_swap=True,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=64,
    seed=42,
)
swap_model_path = "/scratch/xiayao/models/13b/full/vicuna-13b-v1.5-1/"

prompts = [
    "USER: Why did my parent not invite me to their wedding?\nASSISTANT:",
    "USER: Why did my parent not invite me to their wedding?\nASSISTANT:",
    "USER: Where is the capital of China?\nASSISTANT:",
    "USER: Where is the capital of France?\nASSISTANT:",
]

outputs = llm.generate(
    prompts, sampling_params, swap_request=SwapRequest("vicuna", 1, swap_model_path)
)
print(f"with swap: {outputs[0].outputs[0].text}")
print(outputs)
