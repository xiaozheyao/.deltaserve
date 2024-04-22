import os
from vllm import LLM, SamplingParams
from vllm.delta.request import DeltaRequest

tp_size = int(os.environ.get("TP_SIZE", "1"))
use_unoptimized_delta = os.environ.get("UNOPTIMIZED_DELTA", "0") == "1"
use_bitblas = os.environ.get("USE_BITBLAS", "0") == "1"
os.environ["NUMEXPR_MAX_THREADS"] = "32"

print(f"Benchmarking with tensor parallel size={tp_size}")

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_delta=True,
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    max_context_len_to_capture=64,
    max_model_len=64,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=64,
    seed=42,
)
if use_bitblas:
    delta_path = f"/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-bitblas-unopt-1"
elif use_unoptimized_delta:
    delta_path = f"/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-1"
else:
    delta_path = f"/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_{tp_size}-1"

prompts = [
    "USER: Write a letter to the city council to complain the noise in the city.\nASSISTANT:",
]
outputs = llm.generate(
    prompts, sampling_params, delta_request=DeltaRequest("vicuna", 1, delta_path)
)
print(f"with delta: {outputs[0].outputs[0].text}")
print(outputs)
