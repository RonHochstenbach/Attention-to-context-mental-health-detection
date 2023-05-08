import pandas as pd

from data_loader import load_data

task = "Depression"

writings_df = load_data(task)

print(writings_df)

print(len(writings_df['subject'].unique()))





