import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH

df = prepare_df(DEFAULT_PATH)

total_models = 65

df = df[df['type'] == "TTFT"]
df = df[df['total_models'] == total_models]

distributions = df['distribution'].unique()
ars = df['ar'].unique()
for dist in distributions:
    if dist != "zipf:1.5":
        for ar in ars:
            sub_df = df[df['distribution'] == dist]
            sub_df = sub_df[sub_df['ar'] == ar]
            systems = sub_df['sys_name'].unique()
            for system in systems:
                sub_df_sys = sub_df[sub_df['sys_name'] == system]
                # generate slo for each system
                # compute average, median and max
                print(f"System: {system}, Distribution: {dist}, AR: {ar}")
                print(f"MEDIAN: {sub_df_sys['time'].median():.2f}, MAX: {sub_df_sys['time'].max():.2f}, MEAN: {sub_df_sys['time'].mean():.2f}")
                
                