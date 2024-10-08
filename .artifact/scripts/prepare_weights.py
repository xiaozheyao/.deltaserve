"""This script downloads the weights for a given model, and duplicates multiple versions"""
from huggingface_hub import snapshot_download
import os
import shutil
def download(args):
    print(args)
    hf_name = args.hf_id.split("/")[-1]
    snapshot_download(
        repo_id=args.hf_id,
        repo_type="model",
        local_dir=os.path.join(args.local_dir, hf_name),
    )
    # rename the directory to the model name
    shutil.rmtree(os.path.join(args.local_dir, hf_name, ".cache"))
    # rename to -1
    shutil.move(os.path.join(args.local_dir, hf_name), os.path.join(args.local_dir, hf_name + "-1"))
    # copy the model to replicas
    for i in range(1, args.replicas):
        shutil.copytree(os.path.join(args.local_dir, hf_name + "-1"), os.path.join(args.local_dir, hf_name + f"-{i+1}"))
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", type=str, required=True)
    parser.add_argument("--local-dir", type=str, required=True)
    parser.add_argument("--replicas", type=int, default=1) 
    download(parser.parse_args())