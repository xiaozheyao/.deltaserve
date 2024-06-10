from vllm import LLM, SamplingParams
from vllm.delta.request import DeltaRequest
import os

os.environ['USE_MARLIN'] = "1"
tp_size = int(os.environ.get("TP_SIZE", "1"))

llm = LLM(
    model="meta-llama/Llama-2-13b-hf",
    enable_delta=True,
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    swap_space=8,
    max_context_len_to_capture=64,
    max_model_len=128,
)
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=128,
)
delta_path = ".idea/models/13b/4bit/deltazip.lmsys.vicuna-13b-v1.5.4bn2m4-1g"

prompts = [
    "USER: Why did my parent not invite me to their wedding?\nASSISTANT:",
    "USER: What is the difference between OpenCL and CUDA?\nASSISTANT:",
]

outputs = llm.generate(
    prompts,
    sampling_params,
)
print(f"without delta: {outputs[0].outputs[0].text}")
print(outputs)

# warmup
# outputs = llm.generate(
#     prompts,
#     sampling_params,
#     delta_request=DeltaRequest("vicuna", 1, delta_path)
# )

outputs = llm.generate(
    prompts,
    sampling_params,
    delta_request=DeltaRequest("vicuna", 1, delta_path)
)
print(f"with delta: {outputs[0].outputs[0].text}")
print(outputs)