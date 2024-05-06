import os
from tqdm import tqdm

def plot_all(args):

    base_url = args.dir

    files = [x for x in os.listdir(args.dir) if x.endswith(".jsonl")]

    for filename in tqdm(files):
        os.system(
            f"python .artifact/benchmarks/tools/plot/plot_latency_per_request.py --output .artifact/benchmarks/results/figures/ --input {base_url}/{filename}"
        )

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    plot_all(args)