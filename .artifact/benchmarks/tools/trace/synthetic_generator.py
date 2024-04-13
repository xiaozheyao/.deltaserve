import os
import json
import argparse
import datasets
import numpy as np
from loguru import logger
from arrival import PoissonProcess

to_eval_models = [
    "meta-llama/Llama-2-7b-hf",
    "vicuna-7b-1",
    "vicuna-7b-2",
    "vicuna-7b-3",
    "vicuna-7b-4",
    "vicuna-7b-5",
    "vicuna-7b-6",
    "vicuna-7b-7",
    "vicuna-7b-8",
]

def format_lmsys(prompt):
    return f"USER: {prompt}\nASSISTANT:"

def generate_model_distribution(distribution, num_queries):
    if distribution == "uniform":
        return np.random.choice(to_eval_models, num_queries)
    if distribution.startswith("zipf"):
        alpha = float(distribution.split(":")[1])
        assert alpha > 1, "alpha must be greater than 1"
        probs = [i**alpha for i in range(1, len(to_eval_models) + 1)]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(to_eval_models, num_queries, p=probs)
    raise ValueError("Invalid distribution")

def get_dialogs():
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
    all_dialogs = []
    for idx, item in enumerate(trace):
        all_dialogs.append(format_lmsys(item["conversation_a"][0]["content"]))
    return all_dialogs

def generate_synthetic(args):
    print(args)
    poisson_ticks = PoissonProcess(args.arrival_rate).generate_arrivals(
        start=0, duration=args.duration
    )
    logger.info(f"Using Poisson arrival process, total_requests={len(poisson_ticks)}")
    traces_data = []
    dialogs = get_dialogs()
    models = generate_model_distribution(args.distribution, len(poisson_ticks))
    for idx in range(len(poisson_ticks)):
        traces_data.append(
            {
                "id": idx,
                "prompt": dialogs[idx],
                "timestamp": poisson_ticks[idx],
                "model": models[idx],
                "gen_tokens": args.gen_tokens,
            }
        )
    output_file = os.path.join(args.output, f"distribution={args.distribution},ar={args.arrival_rate},duration={args.duration}.jsonl")
    with open(output_file, "w") as fp:
        for datum in traces_data:
            json.dump(datum, fp)
            fp.write("\n")

def main(args):
    generate_synthetic(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution", type=str, default="uniform")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--gen-tokens", type=int, default=512)
    parser.add_argument("--arrival-rate", type=float, default=0)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    main(args)

"""
Examples:
python .artifact/benchmarks/tools/trace/synthetic_generator.py --distribution uniform --gen-tokens 512 --arrival-rate 3 --duration 30 --output .artifact/workloads/
"""