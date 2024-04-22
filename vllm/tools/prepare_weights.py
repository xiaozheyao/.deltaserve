import os
SRC=".idea/models/"
DST="/scratch/xiayao/models/"

NUM_MODELS = 8
PREFIXS = [
    'vicuna-7b-4b0.75s-tp_2',
    'vicuna-7b-4b0.75s-tp_2-unopt-1/'
]

for prefix in PREFIXS:
    for i in range(1, NUM_MODELS+1):
        src = f"{SRC}{prefix}-{i}"
        dst = f"{DST}{prefix}-{i}"
        # os.system(f"cp -r {src} {dst}")
        print(f"cp -r {src} {dst}")