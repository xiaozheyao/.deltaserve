import json
import argparse
from core import run


def before_benchmark(args):
    with open(args.workload, "r") as f:
        workload = [json.loads(line) for line in f]
    
    warmup = args.warmup_strategy
    return args.endpoint, workload, warmup


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, default="workload.json")
    parser.add_argument("--basemodel", type=str, default='gpt2')
    parser.add_argument("--warmup-strategy", type=str, default="random", choices=["random", "none"])
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/completions")
    parser.add_argument("--output", type=str, default="outputs.jsonl")
    parser.add_argument("--annotations", type=str, default="tp=1,max_tokens-512")
    args = parser.parse_args()
    endpoint, workload, warmup = before_benchmark(args)
    outputs = run(endpoint, workload, warmup)
    with open(args.output, "a") as f:
        meta = {
            "workload": args.workload,
            "endpoint": args.endpoint,
            "warmup_strategy": args.warmup_strategy,
            "annotations": args.annotations,
        }
        f.write(json.dumps(meta))
        f.write("\n")
        for output in outputs:
            output["annotations"] = args.annotations
            f.write(json.dumps(output))
            f.write("\n")