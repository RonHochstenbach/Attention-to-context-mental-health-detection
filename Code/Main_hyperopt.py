from comet_ml import Optimizer
import numpy as np
import pandas as pd
import os
import logging
import json
from keras import backend as K

import tensorflow as tf
from collections import Counter
import time
import multiprocessing
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from keras import callbacks
from callbacks import WeightsHistory, LRHistory

from hyperparameters import hyperparams_features, hyperparams
from data_generator import DataGenerator_Base, DataGenerator_BERT
from models import build_HAN, build_HAN_BERT, build_HSAN
from train import train_model
from data_loader import load_erisk_data
from auxilliary_functions import feature_sizes

from datetime import datetime
from load_save_model import save_model_and_params


root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"
#root_dir = "/content/drive/MyDrive/Thesis/Data"  #when cloning for colab


logger = logging.getLogger('training')

if not tf.config.list_physical_devices('GPU'):
    print(tf.config.list_physical_devices())
    raise Exception("NO GPU DETECTED")
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print("GPU found!")

hyperparams['optimizer'] = optimizers.legacy.Adam(learning_rate=hyperparams['lr'], beta_1=0.9, beta_2=0.999, epsilon=0.001)

#IMPORT DATA
task = "Self-Harm"                #"Self-Harm" - "Anorexia" - "Depression"
model_type = "HAN_BERT"                #"HAN" - "HAN_BERT" - "HAN_RoBERTa" - "HSAN"
print(f"Running Hyperopt for the {task} task using the {model_type} model!")

writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 0.7,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)

print(f"There are {len(user_level_data)} subjects, of which {len(subjects_split['train'])} train and {len(subjects_split['test'])} test.")

with tf.device('GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'):
    print(f"Training on {'GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'}!")

    max_hyperopt_trials = 10
    tune_epochs = 10

    #Defining the hyperparameter search space and setting up the experiment
    # parameter_space = {
    #     "lstm_units": {"type": "discrete", "values": [128]},                                         # Fixed
    #     "lstm_units_user": {"type": "discrete", "values": [32]},                                        # Fixed
    #     "dense_bow_units": {"type": "discrete", "values": [20]},                                        # Fixed
    #     "dense_numerical_units": {"type": "discrete", "values": [20]},  # Fixed
    #     "lr": {"type": "float", "min": 0.0005, "max": 0.2, "scalingType": "loguniform"},
    #     "l2_dense": {"type": "float", "min": 0.0000001, "max": 0.1, "scalingType": "loguniform"},
    #     "l2_embeddings": {"type": "float", "min": 0.00000001, "max": 0.2, "scalingType": "loguniform"},
    #     "dropout": {"type": "float", "min": 0, "max": 0.5, "scalingType": "uniform"},
    #     "norm_momentum": {"type": "float", "min": 0, "max": 0.99, "scalingType": "uniform"},
    #     "batch_size": {"type": "discrete", "values" :[2,4,8,16,32,64]},           #ADAPT BASED ON ALGO AND WHERE RUNNING!
    #     "positive_class_weight": {"type": "integer", "min": 2, "max": 10},
    #     "trainable_embeddings": {"type": "discrete", "values": [True, False]},
    #     "sample_seqs": {"type": "discrete", "values": [True, False]},
    #     "freeze_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
    #     "lr_reduce_factor": {"type": "float", "min": 0.0001, "max": 0.8},
    #     "scheduled_lr_reduce_factor": {"type": "float", "min": 0.0001, "max": 0.8},
    #     "lr_reduce_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
    #     "scheduled_lr_reduce_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
    #     "early_stopping_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
    #     "decay": {"type": "float", "min": 0.00000001, "max": 0.5, "scalingType": "loguniform"},
    #     "sampling_distr": {"type": "categorical", "values": ["exp", "uniform"]},
    #     "posts_per_group": {"type": "discrete", "values": [50]},                                        # Fixed
    #     "maxlen": {"type": "discrete", "values": [256]},                                                 # Fixed
    # }

    parameter_space = {
        "lstm_units": {"type": "discrete", "values": [128]},                                         # Fixed
        "lstm_units_user": {"type": "discrete", "values": [32]},                                        # Fixed
        "dense_bow_units": {"type": "discrete", "values": [20]},                                        # Fixed
        "dense_numerical_units": {"type": "discrete", "values": [20]},  # Fixed
        "lr": {"type": "float", "min": 0.00005, "max": 0.001, "scalingType": "loguniform"},
        "l2_dense": {"type": "discrete", "values": [0.00001]},
        "l2_embeddings": {"type": "discrete", "values": [0.00001]},
        "dropout": {"type": "discrete", "values": [0.3]},
        "norm_momentum": {"type": "discrete", "values": [0.1]},
        "batch_size": {"type": "discrete", "values" :[32]},           #ADAPT BASED ON ALGO AND WHERE RUNNING!
        "positive_class_weight": {"type": "discrete", "values" :[2]},
        "trainable_embeddings": {"type": "discrete", "values": [True]},
        "sample_seqs": {"type": "discrete", "values": [False]},
        "freeze_patience": {"type": "discrete", "values": [2000]},
        "lr_reduce_factor": {"type": "discrete", "values": [0.8]},
        "scheduled_lr_reduce_factor": {"type": "discrete", "values": [0.5]},
        "lr_reduce_patience": {"type": "discrete", "values": [55]},
        "scheduled_lr_reduce_patience": {"type": "discrete", "values": [95]},
        "early_stopping_patience": {"type": "discrete", "values": [5]},
        "decay": {"type": "discrete", "values": [0.001]},
        "sampling_distr": {"type": "categorical", "values": ["exp"]},
        "posts_per_group": {"type": "discrete", "values": [50]},                                        # Fixed
        "maxlen": {"type": "discrete", "values": [256]},                                                 # Fixed
    }

    #Add hyperparams specific to HSAN
    if model_type == "HSAN":
        parameter_space["num_heads"] = {"type": "integer", "min": 1, "max": 4}
        parameter_space["key_dim"] = {"type": "integer", "min": 30, "max": 200}
        parameter_space["num_layers"] = {"type": "integer", "min": 1, "max": 4}
        parameter_space["use_positional_encodings"] = {"type": "discrete", "values": [True, False]}

    if model_type == "HAN_BERT":
        parameter_space["sum_layers"] = {"type": "integer", "min": 1, "max": 4}
        parameter_space['trainable_bert_layer'] = {"type": "discrete", "values": [False]}

    config = {
        "algorithm": "bayes",
        "parameters": parameter_space,
        "spec": {
            "metric": "val_loss",
            "objective": "minimize",
        },
    }

    optimizer = Optimizer(config, api_key="ospb2AMYTC4fka83XrIL3fXdj")

    val_losses = []
    best_hyperparams = {}
    num_trials = -1
    for experiment in optimizer.get_experiments(project_name="masterThesis"):

        num_trials +=1
        #Perform maximum of "max_hyperopt_trials" trials
        if num_trials >= max_hyperopt_trials:
            break
        #If more than 5 trials done and loss hasnt improved in the last 5 trials, break
        if num_trials > 5:
            if all(element >= min(val_losses) for element in val_losses[-5:]):
                break

        experiment.add_tag("tune")
        experiment.add_tag(task)
        experiment.add_tag(model_type)

        #Create an input dict for the hyperparameters of this experiment
        experiment_hyperparams = {key: None for key in parameter_space}
        for key in experiment_hyperparams:
            experiment_hyperparams[key] = experiment.get_parameter(key)

        #Always use same optimizer:
        experiment_hyperparams['optimizer'] = optimizers.legacy.Adam(learning_rate=experiment_hyperparams['lr'],
                                                                     decay=experiment_hyperparams['decay'])

        print(experiment_hyperparams)

        # Create DataGenerators
        if model_type == "HAN" or model_type == "HSAN":
            data_generator_train = DataGenerator_Base(user_level_data, subjects_split, set_type='train',
                                                      hyperparams_features=hyperparams_features,
                                                      seq_len=experiment_hyperparams['maxlen'],
                                                      batch_size=experiment_hyperparams['batch_size'],
                                                      posts_per_group=experiment_hyperparams['posts_per_group'],
                                                      post_groups_per_user=None,
                                                      max_posts_per_user=None,
                                                      compute_liwc=True)

            data_generator_valid = DataGenerator_Base(user_level_data, subjects_split, set_type='valid',
                                                      hyperparams_features=hyperparams_features,
                                                      seq_len=experiment_hyperparams['maxlen'],
                                                      batch_size=experiment_hyperparams['batch_size'],
                                                      posts_per_group=experiment_hyperparams['posts_per_group'],
                                                      post_groups_per_user=1,
                                                      max_posts_per_user=None,
                                                      shuffle=False,
                                                      compute_liwc=True)
        elif model_type == "HAN_BERT" or model_type == "HAN_RoBERTa":
            data_generator_train = DataGenerator_BERT(user_level_data, subjects_split, set_type='train',
                                                      hyperparams_features=hyperparams_features, model_type=model_type,
                                                      seq_len=experiment_hyperparams['maxlen'],
                                                      batch_size=experiment_hyperparams['batch_size'],
                                                      posts_per_group=experiment_hyperparams['posts_per_group'],
                                                      post_groups_per_user=None,
                                                      max_posts_per_user=None,
                                                      compute_liwc=True)

            data_generator_valid = DataGenerator_BERT(user_level_data, subjects_split, set_type='valid',
                                                      hyperparams_features=hyperparams_features, model_type=model_type,
                                                      seq_len=experiment_hyperparams['maxlen'],
                                                      batch_size=experiment_hyperparams['batch_size'],
                                                      posts_per_group=experiment_hyperparams['posts_per_group'],
                                                      post_groups_per_user=1,
                                                      max_posts_per_user=None,
                                                      shuffle=False,
                                                      compute_liwc=True)
        else:
            raise Exception("Unknown type!")

        try:
            #Build model
            if model_type == 'HAN':
                model = build_HAN(experiment_hyperparams, hyperparams_features,
                                  feature_sizes()['emotions_dim'], feature_sizes()['stopwords_dim'],
                                  feature_sizes()['liwc_categories_dim'],
                                  ignore_layer=hyperparams['ignore_layer'])
            elif model_type == 'HAN_BERT' or model_type == "HAN_RoBERTa":
                model = build_HAN_BERT(experiment_hyperparams, hyperparams_features, model_type,
                                       feature_sizes()['emotions_dim'], feature_sizes()['stopwords_dim'],
                                       feature_sizes()['liwc_categories_dim'],
                                       ignore_layer=hyperparams['ignore_layer'])
            elif model_type == 'HSAN':
                model = build_HSAN(experiment_hyperparams, hyperparams_features,
                                   feature_sizes()['emotions_dim'], feature_sizes()['stopwords_dim'],
                                   feature_sizes()['liwc_categories_dim'],
                                   ignore_layer=hyperparams['ignore_layer'])
            else:
                Exception("Unknown model!")

            model.summary()

            #Set up callbacks
            weights_history = WeightsHistory()
            lr_history = LRHistory()
            reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=experiment_hyperparams['lr_reduce_factor'],
                                                    patience=experiment_hyperparams['lr_reduce_patience'], min_lr=0.000001, verbose=1)
            lr_schedule = callbacks.LearningRateScheduler(lambda epoch, lr:
                                                          lr if (epoch + 1) % experiment_hyperparams[
                                                              'scheduled_lr_reduce_patience'] != 0 else
                                                          lr * experiment_hyperparams['scheduled_lr_reduce_factor'], verbose=1)

            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=experiment_hyperparams['early_stopping_patience'],
                                                     start_from_epoch=7)

            callbacks_dict = {'weights_history': weights_history,
                              'lr_history': lr_history,
                              'reduce_lr_plateau': reduce_lr,
                              'lr_schedule': lr_schedule,
                              'early_stopping': early_stopping
                              }

            #Fit model
            history = model.fit(data_generator_train,
                                epochs=tune_epochs, initial_epoch=0,
                                class_weight={0:1, 1:experiment_hyperparams['positive_class_weight']},
                                validation_data=data_generator_valid,
                                verbose=1,
                                workers=min(4, multiprocessing.cpu_count()),
                                callbacks=callbacks_dict.values(),
                                use_multiprocessing=False)

            # Log the loss
            if num_trials == 0:
                auc = history.history['auc'][-1]
                precision = history.history['precision'][-1]
                recall = history.history['recall'][-1]

                val_auc = history.history['val_auc'][-1]
                val_precision = history.history['val_precision'][-1]
                val_recall = history.history['val_recall'][-1]

            else:
                auc = history.history['auc_' + str(num_trials)][-1]
                precision = history.history['precision_' + str(num_trials)][-1]
                recall = history.history['recall_' + str(num_trials)][-1]

                val_auc = history.history['val_auc_' + str(num_trials)][-1]
                val_precision = history.history['val_precision_' + str(num_trials)][-1]
                val_recall = history.history['val_recall_' + str(num_trials)][-1]

            loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            f1 = (2 * precision * recall) / (precision + recall + K.epsilon())
            val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall + K.epsilon())

            experiment.log_metric("loss", loss)
            experiment.log_metric("auc", auc)
            experiment.log_metric("precision", precision)
            experiment.log_metric("recall", recall)
            experiment.log_metric("f1", f1)

            experiment.log_metric("val_loss", val_loss)
            experiment.log_metric("val_auc", val_auc)
            experiment.log_metric("val_precision", val_precision)
            experiment.log_metric("val_recall", val_recall)
            experiment.log_metric("val_f1", val_f1)

            val_losses.append(val_loss)

            # If loss is best yet, save hyperparameters to a CSV
            if val_loss <= min(val_losses):
                del experiment_hyperparams['optimizer']
                with open(root_dir + "/HyperOpt/" + task + "_" + model_type + "_hyperparamsForLoss_" + str(
                        round(val_loss, 4)) + ".json", 'w') as f:
                    json.dump(experiment_hyperparams, f)
                best_hyperparams = experiment_hyperparams

        #Continue if Resource exhausted error
        except tf.errors.ResourceExhaustedError as e:
            print("Resource Error, continuing next try!")
            print(e)
            continue
        except:
            print("Another Error, continuing next try!")
            continue

print(f"Best loss was {min(val_losses)}!")
print("Best hyperparameters:")
print(best_hyperparams)
