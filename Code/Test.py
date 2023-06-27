import pandas as pd
import matplotlib.pyplot as plt
from resource_loader import load_NRC, load_LIWC, load_stopwords
from hyperparameters import hyperparams_features, hyperparams
from data_loader import load_erisk_data
from transformers import BertTokenizerFast

from data_generator import DataGenerator_Base, DataGenerator_BERT

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

task = "Anorexia"

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
#
# tokens = ""
#
# encodings = tokenizer(tokens, add_special_tokens=True, max_length=hyperparams['maxlen'],
#                       padding='max_length', truncation=True,
#                       return_attention_mask=True,
#                       # return_tensors='tf'
#                       )
# encoded_token_ids = encodings['input_ids']
# encoded_token_attnmasks = encodings['attention_mask']

writings_df = pd.read_pickle(root_dir +  "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 0.7,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)

data_generator = DataGenerator_BERT(user_level_data, subjects_split, set_type='train',
                                          hyperparams_features=hyperparams_features,
                                          seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                          posts_per_group=hyperparams['posts_per_group'],
                                          post_groups_per_user=hyperparams['post_groups_per_user'],
                                          max_posts_per_user=hyperparams['posts_per_user'],
                                          compute_liwc=True,
                                          ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                          ablate_liwc='liwc' in hyperparams['ignore_layer'])


iter = 0

for i in data_generator:
    if iter >0:
        break
    iter+=1
#
#




