import os
from vllm import LLM, SamplingParams
from vllm.delta.request import DeltaRequest

tp_size = int(os.environ.get("TP_SIZE", "1"))
use_unoptimized_delta = os.environ.get("UNOPTIMIZED_DELTA", "0") == "1"
use_bitblas = os.environ.get("USE_BITBLAS", "0") == "1"
os.environ["NUMEXPR_MAX_THREADS"] = "32"
bitwidth = int(os.environ.get("BITWIDTH", "4"))

print(f"Benchmarking with tensor parallel size={tp_size}")

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_delta=True,
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    gpu_memory_utilization=0.9,
    max_context_len_to_capture=64,
    max_model_len=64,
    max_deltas=1,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=64,
    seed=42,
)
if bitwidth == 4:
    if use_bitblas:
        if tp_size == 1:
            delta_path = f"/scratch/xiayao/models/vicuna-7b-4b0.75s-unopt-bitblas-1"
        else:
            delta_path = f"/scratch/xiayao/models/7b/4bit/4b75s.bitblas.tp_2.1"
    elif use_unoptimized_delta:
        delta_path = f"/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_2-unopt-1"
    else:
        delta_path = f"/scratch/xiayao/models/vicuna-7b-4b0.75s-tp_{tp_size}-1"

elif bitwidth == 2:
    if use_bitblas:
        if tp_size == 2:
            delta_path = f"/scratch/xiayao/models/2bit/2b50s.bitblas.tp_2.1/"
else:
    raise ValueError(f"Unsupported bitwidth: {bitwidth}")

prompts = [
    "USER: Write a letter to the city council to complain the noise in the city.\nASSISTANT:",
    "USER: Who is Alan Turing?\nASSISTANT:",
    "USER: What is the capital of France?\nASSISTANT:",
    "USER: What is the capital of China?\nASSISTANT:",
]
outputs = llm.generate(
    prompts, sampling_params, delta_request=DeltaRequest("vicuna", 1, delta_path)
)
outputs = llm.generate(
    prompts, sampling_params, delta_request=DeltaRequest("vicuna", 1, delta_path)
)
print(f"with delta: {outputs[0].outputs[0].text}")
print(outputs)