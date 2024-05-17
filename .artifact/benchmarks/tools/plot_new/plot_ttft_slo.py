import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH

df = prepare_df(DEFAULT_PATH)

total_models = 33
total_max_deltas = 24

df = df[df['type'] == "E2E Latency"]
df = df[df['total_models'] == total_models]
# for each row, if max_deltas>0, then choose the row with max_deltas

distributions = df['distribution'].unique()
ars = df['ar'].unique()
for dist in distributions:
    if dist != "zipf:1.5" and dist != "distinct":
        for ar in ars:
            sub_df = df[df['distribution'] == dist]
            sub_df = sub_df[sub_df['ar'] == ar]
            systems = sub_df['sys_name'].unique()
            max_deltas = sub_df['max_deltas'].unique()
            for system in systems:
                sub_df_sys = sub_df[sub_df['sys_name'] == system]
                if system == "Baseline-1":
                    sub_df_sys = sub_df_sys[sub_df_sys['max_deltas'] == 0]
                else:
                    sub_df_sys = sub_df_sys[sub_df_sys['max_deltas'] == total_max_deltas]

                # generate slo for each system
                # compute average, median and max
                print(f"System: {system}, Distribution: {dist}, AR: {ar}")
                print(f"MEDIAN: {sub_df_sys['time'].median():.2f}, MAX: {sub_df_sys['time'].max():.2f}, MEAN: {sub_df_sys['time'].mean():.2f}")
                
                