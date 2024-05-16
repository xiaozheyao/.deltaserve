import os
import pandas as pd
from vllm.tools.utils import parse_data, get_short_system_name


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
        results_df = pd.concat([results_df, results])
    return results_df

if __name__=="__main__":
    input_file = ".artifact/benchmarks/results/pjlab/ready"
    df = prepare_df(input_file)
    print(df)