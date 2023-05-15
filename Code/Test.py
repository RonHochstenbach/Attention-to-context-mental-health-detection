import pandas as pd
import logging
import csv
import pickle
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras import optimizers

from hyperparameters import hyperparams_features, hyperparams
from resource_loader import load_NRC, readDict, load_stopwords
from data_loader import load_erisk_data
from auxilliary_functions import tokenize_fields, tokenize_tweets
from data_generator import DataGenerator
from models import build_hierarchical_model
from train import  train, initialize_model

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

logger = logging.getLogger('training')
tf.config.list_physical_devices('GPU')

hyperparams['optimizer'] = optimizers.Adam(lr=hyperparams['lr'], beta_1=0.9, beta_2=0.999, epsilon=0.0001)

#IMPORT DATA
task = "Depression"

writings_df = pd.read_csv(root_dir +  "/Processed Data/tokenized_df_" + task)

model = initialize_model(hyperparams, hyperparams_features,
                         session=None, transfer=False)





