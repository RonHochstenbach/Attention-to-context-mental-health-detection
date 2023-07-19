import pandas as pd

from load_save_model import load_saved_model_weights, load_params
from data_generator import DataGenerator_Base
from data_loader import load_erisk_data

#root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"
root_dir = "/content/drive/MyDrive/Thesis/Data"  #when cloning for colab
#root_dir = "/content/drive/MyDrive/Thesis/Data"  #when cloning for colab
saved_path = root_dir + '/Saved Models/Old/Self-Harm_HAN_2023-07-09 22:30:25.765997'

hyperparams, hyperparams_features = load_params(saved_path)

task = "Self-Harm"          #"Self-Harm" - "Anorexia" - "Depression"
model_type = "HAN"          #"HAN" - "HAN_BERT"
print(f"Running {task} task using the {model_type} model!")

#IMPORT DATA AND CREATE DATAGENERATOR
writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 1,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None,by_subset=True)


print(f"There are {len(user_level_data)} subjects, of which {len(subjects_split['train'])} train and {len(subjects_split['test'])} test.")

data_generator_test = DataGenerator_Base(user_level_data, subjects_split, set_type='test',
                                     hyperparams_features=hyperparams_features,
                                     seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                     posts_per_group=hyperparams['posts_per_group'],
                                     post_groups_per_user=hyperparams['post_groups_per_user'],
                                     max_posts_per_user=hyperparams['posts_per_user'],
                                     compute_liwc=True,
                                     ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                     ablate_liwc='liwc' in hyperparams['ignore_layer'])


#LOAD MODEL AND EVALUATE
model = load_saved_model_weights(saved_path, hyperparams, hyperparams_features, model_type, h5=True)

model.evaluate(data_generator_test)



