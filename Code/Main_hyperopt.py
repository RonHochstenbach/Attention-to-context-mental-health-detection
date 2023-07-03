from comet_ml import Optimizer
import numpy as np
import pandas as pd
import os
import logging
import json

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
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 0.2,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)

print(f"There are {len(user_level_data)} subjects, of which {len(subjects_split['train'])} train and {len(subjects_split['test'])} test.")

with tf.device('GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'):
    print(f"Training on {'GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'}!")

    max_hyperopt_trials = 20
    tune_epochs = 25

    #Defining the hyperparameter search space and setting up the experiment
    parameter_space = {
        "lstm_units": {"type": "discrete", "values": [128]},                                         # Fixed
        "lstm_units_user": {"type": "discrete", "values": [32]},                                        # Fixed
        "dense_bow_units": {"type": "discrete", "values": [20]},                                        # Fixed
        "dense_numerical_units": {"type": "discrete", "values": [20]},  # Fixed
        "lr": {"type": "float", "min": 0.0005, "max": 0.2, "scalingType": "loguniform"},
        "l2_dense": {"type": "float", "min": 0.0000001, "max": 0.1, "scalingType": "loguniform"},
        "l2_embeddings": {"type": "float", "min": 0.00000001, "max": 0.2, "scalingType": "loguniform"},
        "dropout": {"type": "float", "min": 0, "max": 0.5, "scalingType": "uniform"},
        "norm_momentum": {"type": "float", "min": 0, "max": 0.99, "scalingType": "uniform"},
        "batch_size": {"type": "discrete", "values" :[2,4,8,16,32,64]},           #ADAPT BASED ON ALGO AND WHERE RUNNING!
        "positive_class_weight": {"type": "integer", "min": 2, "max": 10},
        "trainable_embeddings": {"type": "discrete", "values": [True, False]},
        "sample_seqs": {"type": "discrete", "values": [True, False]},
        "freeze_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
        "lr_reduce_factor": {"type": "float", "min": 0.0001, "max": 0.8},
        "scheduled_lr_reduce_factor": {"type": "float", "min": 0.0001, "max": 0.8},
        "lr_reduce_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
        "scheduled_lr_reduce_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
        "early_stopping_patience": {"type": "integer", "min": 2, "max": tune_epochs + 1},
        "decay": {"type": "float", "min": 0.00000001, "max": 0.5, "scalingType": "loguniform"},
        "sampling_distr": {"type": "categorical", "values": ["exp", "uniform"]},
        "posts_per_group": {"type": "discrete", "values": [50]},                                        # Fixed
        "maxlen": {"type": "discrete", "values": [256]},                                                 # Fixed
    }

    #Add hyperparams specific to HSAN
    if model_type == "HSAN":
        parameter_space["num_heads"] = {"type": "integer", "min": 1, "max": 5}
        parameter_space["key_dim"] = {"type": "integer", "min": 30, "max": 300}
        parameter_space["num_layers"] = {"type": "integer", "min": 1, "max": 5}
        parameter_space["use_positional_encodings"] = {"type": "discrete", "values": [True, False]}

    config = {
        "algorithm": "bayes",
        "parameters": parameter_space,
        "spec": {
            "metric": "loss",
            "objective": "minimize",
        },
    }

    optimizer = Optimizer(config, api_key="KYZajTCAyJ8SglIRMzNAAib73")

    losses = []
    num_trials = 0
    for experiment in optimizer.get_experiments(project_name="masterThesis"):

        num_trials +=1
        #Perform maximum of "max_hyperopt_trials" trials
        if num_trials >= max_hyperopt_trials:
            break
        #If more than 5 trials done and loss hasnt improved in the last 5 trials, break
        if num_trials > 5:
            if all(element >= min(losses) for element in losses[-5:]):
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
                                                          lr * experiment_hyperparams['scheduled_reduce_lr_factor'], verbose=1)

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

        #Continue if Resource exhausted error
        except tf.errors.ResourceExhaustedError as e:
            print(e)
            continue

        #Log the loss
        loss = history.history['loss'][-1]
        experiment.log_metric("loss", loss)

        #If loss is best yet, save hyperparameters to a CSV
        if loss < min(losses):
            with open(root_dir + "/HyperOpt/" + task + "_" + model_type + "_hyperparamsForLoss_" + str(round(loss,4)) + ".json", 'w') as f:
                json.dump(experiment_hyperparams,f)

        losses.append(loss)

