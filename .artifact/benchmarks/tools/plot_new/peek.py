import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH

def peek_perf(args):
    full_df = prepare_df(args.path, order=False)
    metrics = ["E2E Latency", "TTFT"]
    result_df = []
    for metric in metrics:
        sub_df = full_df[full_df['type'] == metric]
        filenames = sub_df['filename'].unique()
        for filename in filenames:
            df = sub_df[sub_df['filename'] == filename]
            # find per-model latency, then find mean
            min_per_model = df.groupby('model')['time'].min()
            max_per_model = df.groupby('model')['time'].max()
            result_df.append({
                "metric": metric,
                "mean": df['time'].mean(),
                "filename": filename.removeprefix(args.path+"/").removesuffix(".jsonl"),
                "min": min_per_model.min(),
                "max": max_per_model.max(),
            })
    df = pd.DataFrame(result_df)
    
    print(df)        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    
    args = parser.parse_args()
    peek_perf(args)