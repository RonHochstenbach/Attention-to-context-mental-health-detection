import pandas as pd
import numpy as np
from hyperparameters import hyperparams_features, hyperparams
from data_loader import load_erisk_data
import multiprocessing
from data_generator import DataGenerator_BERT, DataGenerator_Base
from feature_encoders import encode_stopwords
from resource_loader import load_stopwords

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

task = "Anorexia"

writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 0.7,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)

data_generator = DataGenerator_Base(user_level_data, subjects_split, set_type='train',
                                                  hyperparams_features=hyperparams_features, #model_type= "HAN_RoBERTa",
                                                  seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                  posts_per_group=hyperparams['posts_per_group'],
                                                  post_groups_per_user=hyperparams['post_groups_per_user'],
                                                  max_posts_per_user=hyperparams['posts_per_user'],
                                                  compute_liwc=True,
                                                  ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                                  ablate_liwc='liwc' in hyperparams['ignore_layer'])



count=0
for i, (x,y) in enumerate(data_generator):
    if i>0:
        break
    print(len(x))
    categorical = x[1]              #emotions, pronouns, liwc
    sparse = x[2]                   #stopwords

    for k in range(sparse.shape[0]):
        for l in range(sparse.shape[1]):
            if np.sum(sparse[k,l,:])>0:
                print(categorical[k,l,:])
                print(sparse[k,l,:])
                print(x[0][k,l,:])
                count+=1
                break
    if count>5:
        break



