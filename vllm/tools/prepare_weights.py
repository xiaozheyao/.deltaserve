import os
SRC=".idea/models/"
DST="/dev/shm/xiayao/models/"

os.makedirs(DST)

NUM_MODELS = 8
PREFIXS = [
    'llama2-chat-70b.4b75s128g-1-tp_8',
    'llama2-chat-70b.4b75s128g-unopt'
]

for prefix in PREFIXS:
    for i in range(1, NUM_MODELS+1):
        src = f"{SRC}{prefix}-{i}"
        dst = f"{DST}{prefix}-{i}"
        # check if src exists
        if not os.path.exists(src):
            print(f"{src} does not exist")
            continue
        # check if dst exists
        if os.path.exists(dst):
            continue
        # os.system(f"cp -r {src} {dst}")
        print(f"cp -r {src} {dst}")