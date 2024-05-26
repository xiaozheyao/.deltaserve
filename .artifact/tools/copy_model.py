import os

NUM_COPIES = 32
START_ID = 33
SRC_FOLDER = "/mnt/petrelfs/huqinghao/xzyao/code/vllm/.idea/full_models/Llama-2-13b-hf"
# SRC_FOLDER = "/mnt/petrelfs/huqinghao/xzyao/code/vllm/.idea/models/lmsys.vicuna-13b-v1.5.4b75s128g"

for i in range(NUM_COPIES):
    cmd = f"cp -r {SRC_FOLDER} {SRC_FOLDER}-{START_ID+i}"
    print(cmd)
