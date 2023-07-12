import pandas as pd
import numpy as np
from hyperparameters import hyperparams_features, hyperparams
from data_loader import load_erisk_data
import multiprocessing
from data_generator import DataGenerator_BERT, DataGenerator_Base
from feature_encoders import encode_stopwords, encode_liwc_categories, encode_emotions
from resource_loader import load_stopwords
import time
from resource_loader import load_LIWC, load_NRC

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

task = "Self-Harm"

writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")
# print(writings_df.keys())
# print(writings_df.shape)
#
# #CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 0.7,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)
# #subject41530000
# example=user_level_data['subject41530000']
# print(example.keys())
# print(example['texts'][0])
# print(example['texts'][0][0])
# print(type(example['texts'][0][0]))





