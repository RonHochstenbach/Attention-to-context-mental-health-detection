import pandas as pd
import logging
import csv

from resource_loader import load_NRC, readDict
from data_loader import load_erisk_data
from auxilliary_functions import tokenize_fields, tokenize_tweets
from data_generator import DataGenerator

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

logger = logging.getLogger('training')

#IMPORT DATA
task = "Depression"
#writings_df = pd.read_csv(root_dir +  "/Processed Data/df_" + task)
#writings_df = tokenize_fields(writings_df, tokenize_fct=tokenize_tweets, columns=['text'])
#writings_df.to_csv("/Users/ronhochstenbach/Desktop/Thesis/Data/Processed Data/tokenized_df_" + task)

writings_df = pd.read_csv(root_dir +  "/Processed Data/tokenized_df_" + task)

#print("Average number of posts per user", writings_df.groupby('subject').count().title.mean())
#print("Average number of comments per user", writings_df.groupby('subject').count().text.mean())

#IMPORT RESOURCES
nrc_lexicon = load_NRC(root_dir + "/Resources/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
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
    for x, y in DataGenerator(user_level_data, subjects_split, sample_seqs=False, max_posts_per_user=None,
                                          set_type=set_type, hierarchical=True, post_groups_per_user=1,
                              posts_per_group=50, shuffle=False,
                             sampling_distr='exp', liwc_words_for_categories=liwc_words_for_categories,
                             compute_liwc=True, classes=3):
#         total_positive += pd.Series(y).sum()
        x_data[set_type].append(x)
        y_data[set_type].append(y)
        logger.info("%s %s positive examples\n" % (total_positive, set_type))

print(x_data)
print(y_data)

