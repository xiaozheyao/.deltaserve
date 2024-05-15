import argparse
import pandas as pd
from vllm.tools.utils import parse_data, get_short_system_name

def peek_perf(args):
    metadata, results = parse_data(args.input)
    short_sys_name = get_short_system_name(metadata)
    print(f"--- {short_sys_name} ---")
    results = pd.DataFrame(results)
    types = results["type"].unique()
    for type in types:
        # average over all runs
        type_results = results[results["type"]==type]
        type_results = type_results['time'].mean()
        print(f"{type}    \t{type_results:.2f} sec")
        
        
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args()
    peek_perf(args)