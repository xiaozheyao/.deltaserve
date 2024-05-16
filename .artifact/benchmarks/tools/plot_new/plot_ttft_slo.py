import pandas as pd
from data_utils import prepare_df, DEFAULT_PATH

df = prepare_df(DEFAULT_PATH)
total_models = 33
df = df[df['type'] == "TTFT"]
df = df[df['total_models'] == total_models]

distributions = df['distribution'].unique()
ar = df['ar'].unique()

print(df)