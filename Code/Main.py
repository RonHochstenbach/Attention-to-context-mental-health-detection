import pandas as pd
import logging
import csv
import pickle
from tqdm import tqdm

from tensorflow.keras import optimizers
from nltk.corpus import stopwords

from hyperparameters import hyperparams_features, hyperparams
from resource_loader import load_NRC, readDict
from data_loader import load_erisk_data
from auxilliary_functions import tokenize_fields, tokenize_tweets
from data_generator import DataGenerator
from models import build_hierarchical_model
from train import initialize_experiment, train

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

logger = logging.getLogger('training')
tf.test.is_gpu_available()


hyperparams['optimizer'] = optimizers.Adam(lr=hyperparams['lr'], beta_1=0.9, beta_2=0.999, epsilon=0.0001)


#IMPORT DATA
task = "Depression"
#writings_df = pd.read_csv(root_dir +  "/Processed Data/df_" + task)
#writings_df = tokenize_fields(writings_df, tokenize_fct=tokenize_tweets, columns=['text'])
#writings_df.to_csv("/Users/ronhochstenbach/Desktop/Thesis/Data/Processed Data/tokenized_df_" + task)

writings_df = pd.read_csv(root_dir +  "/Processed Data/tokenized_df_" + task)

#print("Average number of posts per user", writings_df.groupby('subject').count().title.mean())
#print("Average number of comments per user", writings_df.groupby('subject').count().text.mean())

#IMPORT RESOURCES
nrc_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
emotions = list(nrc_lexicon.keys())
#print(emotions)

liwc_dict = {}
for (w, c) in readDict(root_dir + '/Resources/LIWC2007.dic'):
    if c not in liwc_dict:
        liwc_dict[c] = []
    liwc_dict[c].append(w)

categories = set(liwc_dict.keys())

stopword_list = stopwords.words("english")

#print(len(categories))

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,
                                                            voc_size=20002,
                                                           emotion_lexicon=nrc_lexicon,
                                                           emotions=emotions,
                                                        logger = logger,
                                                          liwc_categories= categories
                                                           )







