import os
files = [x for x in os.listdir(".artifact/benchmarks/results/") if x.endswith(".jsonl")]

for filename in files:
    os.system(f"python .artifact/benchmarks/tools/plot/plot_latency_per_request.py --output .artifact/benchmarks/results/figures --input .artifact/benchmarks/results/{filename}")