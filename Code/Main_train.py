import numpy as np
import pandas as pd
import os
import logging
import csv
import pickle
from tqdm import tqdm
import tensorflow as tf
from collections import Counter
import time
import multiprocessing
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers

from hyperparameters import hyperparams_features, hyperparams
from data_loader import load_erisk_data
from train import initialize_datasets, train
from feature_encoders import encode_liwc_categories
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

hyperparams['optimizer'] = optimizers.legacy.Adam(learning_rate=hyperparams['lr'],
                                                  decay = hyperparams['decay'])

#IMPORT DATA
task = "Depression"                     #"Self-Harm" - "Anorexia" - "Depression"
model_type = "HSAN"                 #"HAN" - "HAN_BERT" - "HSAN" - "Con_HAN"
print(f"Running {task} task using the {model_type} model!")

save = True
if save:
    print("Model will be saved!")
else:
    print("Model will NOT be saved!")

if model_type == "HAN" and not hyperparams['lr'] == 0.0001:
    raise Exception("Wrong LR!")
if model_type == "HAN_BERT" and not hyperparams['lr'] == 0.0002:
    raise Exception("Wrong LR!")
if model_type == "HSAN" and not hyperparams['lr'] == 0.0005:
    raise Exception("Wrong LR!")
if model_type == "Con_HAN" and not hyperparams['lr'] == 0.0005:
    raise Exception("Wrong LR!")


save_epoch = False
continue_from_saved = False
saved_path = '/Users/ronhochstenbach/Desktop/Thesis/Data/Saved Models/Depression_HAN_BERT_2023-07-31 00:41:50.541090_weights.h5'
writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 1,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)

print(f"There are {len(user_level_data)} subjects, of which {len(subjects_split['train'])} train and {len(subjects_split['test'])} test.")


with tf.device('GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'):
    print(f"Training on {'GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'}!")

    store_path = root_dir + '/Saved Models/' + task + '_' + model_type + '_' + str(datetime.now())

    data_generator_train, data_generator_valid = initialize_datasets(user_level_data, subjects_split,
                                                                     hyperparams, hyperparams_features, model_type,
                                                                     validation_set='valid')

    model, history = train(user_level_data, subjects_split, save, save_epoch, store_path,
                              continue_from_saved, saved_path,
                              hyperparams=hyperparams, hyperparams_features=hyperparams_features,
                              epochs = 10,
                              dataset_type=task,
                              model_type=model_type,
                              validation_set='valid',start_epoch=0)

    if save:
        logger.info("Saving model...\n")
        save_model_and_params(model, store_path, hyperparams, hyperparams_features)
