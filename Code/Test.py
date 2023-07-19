import pandas as pd
import numpy as np
from hyperparameters import hyperparams_features, hyperparams
from data_loader import load_erisk_data
import multiprocessing
from data_generator import DataGenerator_BERT, DataGenerator_Base, DataGenerator_BERT_TEST
from feature_encoders import encode_stopwords, encode_liwc_categories, encode_emotions
from resource_loader import load_stopwords
import time
from resource_loader import load_LIWC, load_NRC

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

task = "Self-Harm"
print("test")

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

data_gen = DataGenerator_BERT_TEST(user_level_data, subjects_split, set_type='train',
                                          hyperparams_features=hyperparams_features, model_type="HAN_BERT",
                                          seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                          posts_per_group=hyperparams['posts_per_group'],
                                          post_groups_per_user=None,
                                          max_posts_per_user=hyperparams['posts_per_user'],
                                          compute_liwc=True,
                                          ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                          ablate_liwc='liwc' in hyperparams['ignore_layer'])

for i, (x,y) in enumerate(data_gen):
    print(i)
    if i==0:
        break







