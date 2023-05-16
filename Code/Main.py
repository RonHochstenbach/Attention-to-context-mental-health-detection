import pandas as pd
import logging
import csv
import pickle
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
import time

from tensorflow.keras import optimizers

from hyperparameters import hyperparams_features, hyperparams
from resource_loader import load_NRC, readDict, load_stopwords
from data_loader import load_erisk_data
from auxilliary_functions import tokenize_fields, tokenize_tweets
from data_generator import DataGenerator
from models import build_hierarchical_model
from train import  train
from feature_encoders import encode_liwc_categories

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

logger = logging.getLogger('training')
tf.config.list_physical_devices('GPU')

hyperparams['optimizer'] = optimizers.Adam(lr=hyperparams['lr'], beta_1=0.9, beta_2=0.999, epsilon=0.0001)

#IMPORT DATA
task = "Depression"
# writings_df = pd.read_pickle(root_dir +  "/Processed Data/df_" + task + ".pkl")
# writings_df = tokenize_fields(writings_df, tokenize_fct=tokenize_tweets, columns=['text', 'title'])
# writings_df.to_pickle("/Users/ronhochstenbach/Desktop/Thesis/Data/Processed Data/tokenized_df_" + task + ".pkl")

writings_df = pd.read_pickle(root_dir +  "/Processed Data/tokenized_df_" + task + ".pkl")

#IMPORT RESOURCES
# nrc_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
# emotions = list(nrc_lexicon.keys())
# #print(emotions)
#
# liwc_dict = {}
# for (w, c) in readDict(root_dir + '/Resources/LIWC2007.dic'):
#     if c not in liwc_dict:
#         liwc_dict[c] = []
#     liwc_dict[c].append(w)

#
# stopword_list = load_stopwords(root_dir + '/Resources/stopwords.txt')

#print(len(categories))

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,
                                                           hyperparams_features=hyperparams_features,
                                                                                logger=None,
                                                              by_subset=True
                                                                               )

models, history = train(user_level_data, subjects_split,
          hyperparams=hyperparams, hyperparams_features=hyperparams_features,
          dataset_type=task,
          validation_set='valid',
          version=0, epochs=2, start_epoch=0
                                       )


