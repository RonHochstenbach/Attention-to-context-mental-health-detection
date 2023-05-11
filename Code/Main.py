import pandas as pd
import logging
import csv
import pickle
from tqdm import tqdm

from resource_loader import load_NRC, readDict
from data_loader import load_erisk_data
from auxilliary_functions import tokenize_fields, tokenize_tweets
from data_generator import DataGenerator

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

logger = logging.getLogger('training')

hyperparams_features = {
    "max_features": 20002,
    "embedding_dim": 300,
    "vocabulary_path": root_dir + '/Resources/vocab.pickle',
    "nrc_lexicon_path" : root_dir + "/Resources/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",
    "liwc_path": root_dir + '/Resources/LIWC2007.dic',
    "stopwords_path": root_dir + '/Resources/stopwords.txt',
    "embeddings_path": "Resources/glove.840B.300d.txt"#,
    #"liwc_words_cached": "data/liwc_categories_for_vocabulary_erisk_clpsych_stop_20K.pkl",
    #"pretrained_model_path": "models/lstm_symanto_hierarchical64"
}

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
#print(len(categories))

#CREATE VOCABULARY
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,
                                                            voc_size=20002,
                                                           emotion_lexicon=nrc_lexicon,
                                                           emotions=emotions,
                                                        logger = logger,
                                                          liwc_categories= categories
                                                           )



x_data = {'train': [], 'valid': [], 'test': []}
y_data = {'train': [], 'valid': [], 'test': []}
for set_type in ['train', 'valid', 'test']:
    total_positive = 0
    for x, y in  tqdm(DataGenerator(user_level_data=user_level_data, subjects_split=subjects_split, set_type='train',
                 batch_size=32, seq_len=512, hyperparams_features=hyperparams_features,
                 post_groups_per_user=None, posts_per_group=10, post_offset = 0,
                 pronouns=["i", "me", "my", "mine", "myself"],
                 compute_liwc=False,
                 max_posts_per_user=None,
                 shuffle=True, keep_last_batch=True)):
#         total_positive += pd.Series(y).sum()
        x_data[set_type].append(x)
        y_data[set_type].append(y)
        logger.info("%s %s positive examples\n" % (total_positive, set_type))

print(x_data)
print(y_data)

