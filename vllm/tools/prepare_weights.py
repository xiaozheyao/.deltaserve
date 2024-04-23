import os

SRC = ".idea/models/"
DST = "/scratch/xiayao/models/"

os.makedirs(DST, exist_ok=True)

NUM_MODELS = 8
PREFIXS = [
    "awq-vicuna-7b-v1.5-4b128g", 
    "vicuna-7b-v1.5"
]

for prefix in PREFIXS:
    for i in range(1, NUM_MODELS + 1):
        src = f"{SRC}{prefix}-{i}"
        dst = f"{DST}{prefix}-{i}"
        # check if src exists
        if not os.path.exists(src):
            print(f"{src} does not exist")
            continue
        # check if dst exists
        if os.path.exists(dst):
            continue
        os.system(f"cp -r {src} {dst}")
        # print(f"cp -r {src} {dst}")
