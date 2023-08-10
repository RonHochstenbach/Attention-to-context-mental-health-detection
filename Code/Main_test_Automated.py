import pandas as pd
from tensorflow.keras import optimizers
from keras.models import load_model
import json

from hyperparameters import hyperparams, hyperparams_features
from load_save_model import load_saved_model_weights, load_params
from data_generator import DataGenerator_Base, DataGenerator_BERT
from data_loader import load_erisk_data
from metrics_decision_based import evaluate_for_subjects

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"
#root_dir = "/content/drive/MyDrive/Thesis/Data"  #when cloning for colab

saved_paths_SelfHarm = {
    "HAN" : '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Self-Harm/Self-Harm_HAN_2023-07-18 11:13:44.567461',
    "HAN_BERT": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Self-Harm/Self-Harm_HAN_TinyBERT_2023-07-30 18:32:12.354982',
    "HSAN": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Self-Harm/Self-Harm_HSAN_2023-07-30 16:14:20.950111',
    "Con_HAN": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Self-Harm/Self-Harm_Con_HAN_2023-07-27 11:02:20.050187'
    }

saved_paths_Anorexia = {
    "HAN" : '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Anorexia/Anorexia_HAN_2023-07-18 16:34:09.218490',
    "HAN_BERT": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Anorexia/Anorexia_HAN_TinyBERT_2023-07-28 19:38:33.513781',
    "HSAN": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Anorexia/Anorexia_HSAN_2023-07-30 20:48:15.391809',
    "Con_HAN": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Anorexia/Anorexia_Con_HAN_2023-07-27 22:01:18.128520'
    }

saved_paths_Depression = {
    "HAN" : '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Depression/Depression_HAN_2023-07-31 23:35:39.671308',
    "HAN_BERT": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Depression/Depression_HAN_BERT_2023-08-01 18:29:44.243405',
    "HSAN": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Depression/Depression_HSAN_2023-08-01 20:34:48.793109',
    "Con_HAN": '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Final Trained Models (10 epochs)/Depression/Depression_Con_HAN_2023-07-28 23:52:39.351657'
    }
#hyperparams, hyperparams_features = load_params(saved_path)
hyperparams['optimizer'] = optimizers.legacy.Adam(learning_rate=hyperparams['lr'],
                                                  decay = hyperparams['decay'])

task = "Depression"          #"Self-Harm" - "Anorexia" - "Depression"
#analysis_type = "Keras"  #"Custom" or "Keras"

for analysis_type in ["Custom", "Keras"]:
    for model_type in ["Con_HAN", "HAN", "HAN_BERT", "HSAN"]:
        try:
            if task == "Self-Harm":
                saved_path = saved_paths_SelfHarm[model_type]
            elif task == "Anorexia":
                saved_path = saved_paths_Anorexia[model_type]
            elif task == "Depression":
                saved_path = saved_paths_Depression[model_type]
            else:
                raise Exception("Invalid Model Type!")

            print(f"Running {task} task using the {model_type} model!")

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
            elif model_type == "HAN_BERT" or model_type == "Con_HAN":
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
            results = {}
            if analysis_type == "Keras":
                metrics = model.evaluate(data_generator_test)
                metrics_names = model.metrics_names
                for i, metr in enumerate(metrics_names):
                    results[metr] = metrics[i]

            elif analysis_type == "Custom":
                results = evaluate_for_subjects(model, data_gen_class, subjects_split['test'], user_level_data, hyperparams, hyperparams_features, model_type,
                                          alert_threshold=0.5, rolling_window=0)
                for metric, value in results.items():
                    print(f"{metric}: {value}")
            else:
                Exception("Not a valid analysis type!")

            save_place = root_dir + '/Test Results/' + task + '_' + model_type + '_' + analysis_type + '.json'

            with open(save_place, 'w') as fp:
                json.dump(results, fp)

        except Exception as e:
            print(f"Something went wrong when testing the {task} task with the {model_type} model and {analysis_type} method.")
            print(e)
            continue
