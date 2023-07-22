import pandas as pd

from load_save_model import load_saved_model_weights, load_params
from data_generator import DataGenerator_Base, DataGenerator_BERT
from data_loader import load_erisk_data
from metrics_decision_based import evaluate_for_subjects

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"
#root_dir = "/content/drive/MyDrive/Thesis/Data"  #when cloning for colab

saved_path = '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Self-Harm_HSAN_2023-07-20 21:01:52.538001'

hyperparams, hyperparams_features = load_params(saved_path)

task = "Self-Harm"          #"Self-Harm" - "Anorexia" - "Depression"
model_type = "HSAN"          #"HAN" - "HAN_BERT"
print(f"Running {task} task using the {model_type} model!")

analysis_type = "Custom"  #"Custom" or "Keras"

#IMPORT DATA AND CREATE DATAGENERATOR
writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 1,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None,by_subset=True)

print(f"There are {len(user_level_data)} subjects, of which {len(subjects_split['train'])} train and {len(subjects_split['test'])} test.")

if model_type == "HAN" or model_type == "HSAN":
    data_gen_class = DataGenerator_Base
    data_generator_test = DataGenerator_Base(user_level_data, subjects_split, set_type='test',
                                              hyperparams_features=hyperparams_features,
                                              seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                              posts_per_group=hyperparams['posts_per_group'], post_groups_per_user=None,
                                              max_posts_per_user=hyperparams['posts_per_user'],
                                              compute_liwc=True,
                                              ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                              ablate_liwc='liwc' in hyperparams['ignore_layer'])
elif model_type == "HAN_BERT" or model_type == "HAN_RoBERTa":
    data_gen_class = DataGenerator_BERT
    data_generator_test = DataGenerator_BERT(user_level_data, subjects_split, set_type='test',
                                              hyperparams_features=hyperparams_features, model_type=model_type,
                                              seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                              posts_per_group=hyperparams['posts_per_group'],
                                              post_groups_per_user=None,
                                              max_posts_per_user=hyperparams['posts_per_user'],
                                              compute_liwc=True,
                                              ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                              ablate_liwc='liwc' in hyperparams['ignore_layer'])
else:
    raise Exception("Unknown type!")

#LOAD MODEL AND EVALUATE
model = load_saved_model_weights(saved_path, hyperparams, hyperparams_features, model_type, h5=True)

if analysis_type == "Keras":
    model.evaluate(data_generator_test)
elif analysis_type == "Custom":
    results = evaluate_for_subjects(model, data_gen_class, subjects_split['test'], user_level_data, hyperparams, hyperparams_features,
                              alert_threshold=0.5, rolling_window=0)
    for metric, value in results.items():
        print(f"{metric}: {value}")
else:
    Exception("Not a valid analysis type!")



