import os
import numpy as np
import subprocess
import time
from typing import List

model_path = "/scratch/xiayao/models/"
hf_path = "/mnt/scratch/xiayao/cache/HF"
num_models = 8

def start_server(base_model: str, engine: str, num_models:int, tp_size: int):
    module_list = ""
    for i in range(1, num_models+1):
        module_list += f" {engine}-{i}=/models/deltazip.lmsys.vicuna-7b-v1.5.4bn2m4-1g.tp{tp_size}-{i}"
    if engine=="delta":
        cmd = f"singularity run --nv --env USE_MARLIN=1 --env HF_HOME=/hf --env HF_TOKEN={hf_token} --bind {model_path}:/models --bind {hf_path}:/hf .artifact/images/deltaserve.dev.sif  python -m vllm.entrypoints.openai.api_server --model {base_model} --host 0.0.0.0 --enable-delta --disable-log-requests --gpu-memory-utilization 0.85 --delta-modules {module_list} --max-deltas 8 --max-cpu-deltas 32 --tensor-parallel-size 2 --enforce-eager --port 8080"
    else:
        raise NotImplementedError
    p = subprocess.Popen(cmd, shell=True, )
    return p

def start_router(port=8080):
    cmd = f"python .artifact/tools/router.py --upstreams http://localhost:{port}"
    p = subprocess.Popen(cmd, shell=True)
    return p

def kill_all(processes: List[int]):
    pass

if __name__=="__main__":
    cuda_devices="2,3"
    base_model = "meta-llama/Llama-2-7b-hf"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    hf_token = os.environ["HF_TOKEN"]
    ars = np.arange(0.1, 1.1, 0.1)
    
    # for ar in ars:
    server_process = start_server(base_model, "delta", num_models, 2)
    router_process = start_router()
    time.sleep(30)
    router_process.kill()
    server_process.kill()