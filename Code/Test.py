import pandas as pd
from hyperparameters import hyperparams_features, hyperparams
from data_loader import load_erisk_data
import multiprocessing
from data_generator import DataGenerator_BERT

print(multiprocessing.cpu_count())


root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

task = "Anorexia"

writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 0.7,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)


data_generator = DataGenerator_BERT(user_level_data, subjects_split, set_type='train',
                                                  hyperparams_features=hyperparams_features, model_type= "RoBERTa",
                                                  seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                                  posts_per_group=hyperparams['posts_per_group'],
                                                  post_groups_per_user=hyperparams['post_groups_per_user'],
                                                  max_posts_per_user=hyperparams['posts_per_user'],
                                                  compute_liwc=True,
                                                  ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                                  ablate_liwc='liwc' in hyperparams['ignore_layer'])

for i, (x,y) in enumerate(data_generator):
    print(x)

    if i>1:
        break

