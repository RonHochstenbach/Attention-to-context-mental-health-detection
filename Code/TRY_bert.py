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
from resource_loader import load_NRC, readDict, load_stopwords
from data_loader import load_erisk_data
from auxilliary_functions import tokenize_fields, tokenize_tweets, build_vocabulary, tokenize_fields_bert
from train import train
from feature_encoders import encode_liwc_categories
from datetime import datetime
from load_save_model import save_model_and_params
from transformers import BertTokenizer

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
task = "Anorexia"                #"Self-Harm" - "Anorexia" - "Depression"
model_type = "HAN_BERT"          #"HAN" - "HAN_BERT"
print(f"Running {task} task using the {model_type} model!")

save = True
if save:
    print("Model will be saved!")
else:
    print("Model will NOT be saved!")

# writings_df = pd.read_pickle(root_dir + "/Processed Data/df_" + task + ".pkl")
# writings_df = tokenize_fields_bert(writings_df, columns=['text', 'title'])
# writings_df.to_pickle("/Users/ronhochstenbach/Desktop/Thesis/Data/Processed Data/tokenized_df_BERT_" + task + ".pkl")

writings_df = pd.read_pickle(root_dir + "/Processed Data/tokenized_df_" + task + ".pkl")

#CREATE VOCABULARY, PROCESS DATA, DATAGENERATOR
user_level_data, subjects_split, vocabulary = load_erisk_data(writings_df,train_prop= 0.7,
                                                           hyperparams_features=hyperparams_features,
                                                           logger=None, by_subset=True)

print(f"There are {len(user_level_data)} subjects, of which {len(subjects_split['train'])} train and {len(subjects_split['test'])} test.")

with tf.device('GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'):
    print(f"Training on {'GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'}!")

    store_path = root_dir + '/Saved Models/' + task + '_' + model_type + '_' + str(datetime.now())

    model, history = train(user_level_data, subjects_split, save, store_path,
          hyperparams=hyperparams, hyperparams_features=hyperparams_features,
          dataset_type=task,
          model_type=model_type,
          validation_set='valid',
          version=0, epochs=15, start_epoch=0)

    if save:
        logger.info("Saving model...\n")
        save_model_and_params(model, store_path, hyperparams, hyperparams_features)
