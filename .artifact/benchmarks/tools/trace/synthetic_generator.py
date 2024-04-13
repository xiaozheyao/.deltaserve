import json
import argparse
import datasets
import pandas as pd
import numpy as np
from loguru import logger
from .arrival import PoissonProcess

to_eval_models = [
    "vicuna-7b-1",
    "vicuna-7b-2",
    "vicuna-7b-3",
    "vicuna-7b-4",
    "vicuna-7b-5",
    "vicuna-7b-6",
    "vicuna-7b-7",
    "vicuna-7b-8",
    "meta-llama/Llama-2-7b-hf"
]

def format_lmsys(prompt):
    return f"USER: {prompt}\nASSISTANT:"

def generate_model_distribution(distribution, num_queries, distribution_args):
    if distribution == "uniform":
        return np.random.choice(to_eval_models, num_queries)
    if distribution == "zipf":
        return np.random.zipf(distribution_args, num_queries)
    
def get_dialogs():
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
    all_dialogs = []
    for idx, item in enumerate(trace):
        all_dialogs.append(format_lmsys(item["conversation_a"][0]["content"]))
    return all_dialogs

def generate_synthetic(args):
    print(args)
    logger.info("Using Poisson arrival process")
    poisson_ticks = PoissonProcess(args.arrival_rate).generate_arrivals(
        start=0, duration=args.duration
    )
    traces_data = []
    dialogs = get_dialogs()
    
    
    for idx in range(len(poisson_ticks)):
        traces_data.append(
            {
                "id": idx,
                "prompt": dialogs[idx],
                "timestamp": poisson_ticks[idx],
                "model": mapped_models[idx],
                "max_tokens": args.max_tokens,
            }
        )
        
    with open(args.output, "w") as fp:
        for datum in traces_data:
            json.dump(datum, fp)
            fp.write("\n")

def main(args):
    generate_synthetic(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution", type=str, default="uniform", choices=["uniform", "identical", "zipf"])
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--gen-tokens", type=int, default=512)
    parser.add_argument("--arrival-rate", type=float, default=0)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    main(args)
