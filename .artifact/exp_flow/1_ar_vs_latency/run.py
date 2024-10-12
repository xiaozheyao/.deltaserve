import os
import numpy as np
import subprocess
import time
from typing import List
import requests
from vllm.tools.gpu import get_processes

model_path = "/scratch/xiayao/models/"
hf_path = "/mnt/scratch/xiayao/cache/HF"
num_models = 8

def start_server(base_model: str, engine: str, num_models:int, tp_size: int):
    module_list = ""
    for i in range(1, num_models+1):
        if engine=="delta":
            module_list += f" delta-{i}=/models/delta/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g.tp{tp_size}-{i}"
        elif engine=="swap":
            module_list += f" delta-{i}=/models/full/vicuna-7b-v1.5-{i}"
        elif engine=="lora":
            module_list += f" lora-{i}=/models/lora/llama-7b-hf-lora-r16-{i}"
    if engine=="delta":
        cmd = f"singularity run --nv --env USE_MARLIN=1 --env HF_HOME=/hf --env HF_TOKEN={hf_token} --bind {model_path}:/models --bind {hf_path}:/hf .artifact/images/deltaserve.dev.sif  python -m vllm.entrypoints.openai.api_server --model {base_model} --host 0.0.0.0 --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 --delta-modules {module_list} --max-deltas 8 --max-cpu-deltas 16 --tensor-parallel-size 2 --enforce-eager --port 8080"
    elif engine == "swap":
        cmd = f"singularity run --nv --env HF_HOME=/hf --env HF_TOKEN={hf_token} --bind {model_path}:/models --bind {hf_path}:/hf .artifact/images/deltaserve.dev.sif python -m vllm.entrypoints.openai.api_server --model {base_model} --host 0.0.0.0 --disable-log-requests --gpu-memory-utilization 0.85 --swap-modules {module_list} --max-swap-slots 2 --max-cpu-models 16 --tensor-parallel-size 2 --enforce-eager --port 8080"
    elif engine == "lora":
        cmd = f"singularity run --nv --env HF_HOME=/hf --env HF_TOKEN={hf_token} --bind {model_path}:/models --bind {hf_path}:/hf .artifact/images/deltaserve.dev.sif python -m vllm.entrypoints.openai.api_server --model {base_model} --host 0.0.0.0 --enable-lora --disable-log-requests --gpu-memory-utilization 0.85 --lora-modules {module_list} --max-loras 8 --max-cpu-loras 16 --tensor-parallel-size 2 --enforce-eager --port 8080"
    else:
        raise NotImplementedError
    p = subprocess.Popen(cmd, shell=True, )
    return p

def start_router(port=8080):
    cmd = f"python .artifact/tools/router.py --upstreams http://localhost:{port}"
    p = subprocess.Popen(cmd, shell=True)
    return p

def kill_all(processes: List[int]):
    cmd = "kill -9 "+" ".join([str(p.pid) for p in processes])
    os.system(cmd)
    # ensure there's no process left
    has_remaining_process = True
    while has_remaining_process:
        processes = get_processes()
        for k, v in processes.items():
            if k in [int(x) for x in cuda_devices.split(",")]:
                if len(v)>0:
                    for process in v:
                        os.system(f"kill -9 {process['pid']}")
        has_remaining_process = False
        for k in cuda_devices.split(","):
            if len(processes[int(k)])==0:
                has_remaining_process = True

def issue_request(ar: float, mode: str):
    if mode=="swap":
        request_mode = "delta"
    else:
        request_mode = mode
    cmd = f"python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/gen_fixed/mode={request_mode},models=8,distribution=uniform,ar={ar},duration=60.0.jsonl --output .artifact/benchmarks/results/v2/ablation_ar --endpoints http://localhost:3001"
    os.system(cmd)
    
if __name__=="__main__":
    cuda_devices="2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    hf_token = os.environ["HF_TOKEN"]
    base_model = "meta-llama/Llama-2-7b-hf"

    modes = ['swap', 'lora', 'delta']
    total_trial = 5
    ars = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    for trial in range(total_trial):
        for mode in modes:
            for ar in ars:
                server_process = start_server(base_model, mode, num_models, 2)
                router_process = start_router()
                issue_request(ar, mode)
                kill_all([server_process, router_process])
                print(f"Finished ar={ar}, mode={mode}, trial={trial}, sleeping for 30s")
                time.sleep(30)