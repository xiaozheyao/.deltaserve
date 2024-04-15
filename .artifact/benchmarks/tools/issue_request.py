import os
import json
import uuid
import argparse
from core import run, get_sys_info

def before_benchmark(args):
    with open(args.workload, "r") as f:
        workload = [json.loads(line) for line in f]
    
    warmup = args.warmup_strategy
    sysinfo = get_sys_info(args.endpoints[0])
    return args.endpoints, workload, warmup, sysinfo

def generate_annotation(endpoints, sysinfo, workload):
    tp_degree = sysinfo['tensor_parallel_size']
    rp_degree = len(endpoints)
    annotations = f"{workload},tp_degree={tp_degree},rp_degree={rp_degree}"
    return annotations

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="workload.json")
    parser.add_argument("--base-model", type=str, default='gpt2')
    parser.add_argument("--warmup-strategy", type=str, default="random", choices=["random", "none"])
    parser.add_argument("--endpoints", default=['http://localhost:8000'], nargs='+')
    parser.add_argument("--output", type=str, default="outputs/")
    parser.add_argument("--manual-reload", action="store_true", default=False)
    args = parser.parse_args()
    
    endpoints, workload, warmup, sysinfo = before_benchmark(args)
    workload_annotation = args.workload.split("/")[-1].split(".")[0]
    annotations = generate_annotation(args.endpoints, sysinfo, workload_annotation)
    outputs = run(endpoints, workload, warmup, args.base_model, sysinfo)
    new_unique_name = str(uuid.uuid4())
    output_file = os.path.join(args.output, f"{new_unique_name}.jsonl")
    
    with open(output_file, "a") as f:
        meta = {
            "workload": args.workload,
            "endpoints": args.endpoints,
            "warmup_strategy": args.warmup_strategy,
            "annotations": annotations,
            "sys_info": sysinfo,
        }
        f.write(json.dumps(meta))
        f.write("\n")
        for output in outputs:
            f.write(json.dumps(output))
            f.write("\n")

"""
Example: 
python .artifact/benchmarks/tools/issue_request.py --workload .artifact/workloads/distribution=uniform,ar=3.0,duration=30.0.jsonl --base-model meta-llama/Llama-2-7b-hf --output .artifact/benchmarks/results
"""