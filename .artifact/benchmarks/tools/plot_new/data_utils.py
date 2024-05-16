import os
import pandas as pd
from vllm.tools.utils import parse_data, get_short_system_name

DEFAULT_PATH=".artifact/benchmarks/results/pjlab/ready"

def prepare_df(input_file):
    if os.path.isdir(input_file):
        inputs = [os.path.join(input_file, f) for f in os.listdir(input_file) if f.endswith(".jsonl")]
    else:
        inputs = [input_file]
    results_df = pd.DataFrame([])
    for input_file in inputs:
        metadata, results = parse_data(input_file)
        short_sys_name, order = get_short_system_name(metadata)
        results = pd.DataFrame(results)
        results['sys_name'] = short_sys_name
        results['order'] = order
        results['distribution'] = metadata['distribution']
        results['ar'] = metadata['ar']
        results['tp_size'] = metadata['tp_size']
        results['policy'] = metadata['policy']
        results['total_models'] = metadata['total_models']
        results_df = pd.concat([results_df, results])
    return results_df

if __name__=="__main__":
    input_file = ".artifact/benchmarks/results/pjlab/ready"
    df = prepare_df(input_file)
    ars = [1,2,3,6,9]
    distribution = ['uniform', 'zipf:1.5', 'azure']
    for distribution in distribution:
        for ar in ars:
            for tp_size in [1, 2, 4, 8]:
                for policy in ['lru', 'fifo', 'lfu']:
                    print(f"{distribution}, {ar}, {tp_size}, {policy}")
                    print(df[(df['distribution'] == distribution) & (df['ar'] == ar) & (df['tp_size'] == tp_size) & (df['policy'] == policy)].groupby(['sys_name', 'order', 'type'])['time'].mean())
                    print()