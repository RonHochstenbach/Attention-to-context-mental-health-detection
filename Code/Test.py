import pandas as pd

from resource_loader import load_NRC, load_LIWC, load_stopwords
from hyperparameters import hyperparams_features

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"
task = "Self-Harm"


writings_df = pd.read_pickle(root_dir +  "/Processed Data/tokenized_df_" + task + ".pkl")
pd.set_option('display.max_columns', None)
#print(writings_df.head(100))

liwc_dict = load_LIWC(hyperparams_features['liwc_path'])
liwc_categories = set(liwc_dict.keys())
categories = [c for c in liwc_categories if c in writings_df.columns]

print(categories)