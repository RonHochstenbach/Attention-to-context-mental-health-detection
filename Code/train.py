from callbacks import FreezeLayer, WeightsHistory, LRHistory
from keras import callbacks
from metrics import Metrics
import logging, sys, os
import pickle
from data_generator import DataGenerator
from models import build_HAN, build_HAN_BERT
from resource_loader import load_NRC, load_LIWC, load_stopwords
import tensorflow as tf
import keras
import multiprocessing

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # When cudnn implementation not found, run this
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Note: when starting kernel, for gpu_available to be true, this needs to be run
# only reserve 1 GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

root_dir = "/Users/ronhochstenbach/Desktop/Thesis"
def initialize_datasets(user_level_data, subjects_split, hyperparams, hyperparams_features,
                        validation_set, session=None):

    data_generator_train = DataGenerator(user_level_data, subjects_split, set_type='train',
                                        hyperparams_features=hyperparams_features,
                                        seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                        posts_per_group=hyperparams['posts_per_group'], post_groups_per_user=hyperparams['post_groups_per_user'],
                                        max_posts_per_user=hyperparams['posts_per_user'],
                                         compute_liwc=True,
                                         ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                         ablate_liwc='liwc' in hyperparams['ignore_layer'])

    data_generator_valid = DataGenerator(user_level_data, subjects_split, set_type=validation_set,
                                            hyperparams_features=hyperparams_features,
                                        seq_len=hyperparams['maxlen'], batch_size=hyperparams['batch_size'],
                                        posts_per_group=hyperparams['posts_per_group'],
                                         post_groups_per_user=1,
                                        max_posts_per_user=None,
                                        shuffle=False,
                                         compute_liwc=True,
                                         ablate_emotions='emotions' in hyperparams['ignore_layer'],
                                         ablate_liwc='liwc' in hyperparams['ignore_layer'])

    return data_generator_train, data_generator_valid


def initialize_model(hyperparams, hyperparams_features, model_type,
                     logger=None, session=None, transfer=False):
    if not logger:
        logger = logging.getLogger('training')
        ch = logging.StreamHandler(sys.stdout)
        # create formatter
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
    logger.info("Initializing model...\n")
    if 'emotions' in hyperparams['ignore_layer']:
        emotions_dim = 0
    else:
        emotions = load_NRC(hyperparams_features['nrc_lexicon_path'])
        emotions_dim = len(emotions)
    if 'liwc' in hyperparams['ignore_layer']:
        liwc_categories_dim = 0
    else:
        liwc_categories = load_LIWC(hyperparams_features['liwc_path'])
        liwc_categories_dim = len(liwc_categories)
    if 'stopwords' in hyperparams['ignore_layer']:
        stopwords_dim = 0
    else:
        stopwords_list = load_stopwords(hyperparams_features['stopwords_path'])
        stopwords_dim = len(stopwords_list)

    # Initialize model

    if model_type == 'HAN':
        model = build_HAN(hyperparams, hyperparams_features,
                                         emotions_dim, stopwords_dim, liwc_categories_dim,
                                         ignore_layer=hyperparams['ignore_layer'])
    else:
        model = build_HAN_BERT

    model.summary()
    return model


def train_model(model, hyperparams, save, store_path,
                data_generator_train, data_generator_valid,
                epochs, class_weight, start_epoch=0, workers=multiprocessing.cpu_count(),
                callback_list=[], logger=None,

                model_path='/tmp/model',
                validation_set='valid',
                verbose=1):

    if not logger:
        logger = logging.getLogger('training')
        ch = logging.StreamHandler(sys.stdout)
        # create formatter
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
    logger.info("Initializing callbacks...\n")
    # Initialize callbacks
    freeze_layer = FreezeLayer(patience=hyperparams['freeze_patience'], set_to=not hyperparams['trainable_embeddings'])
    weights_history = WeightsHistory()

    lr_history = LRHistory()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=hyperparams['reduce_lr_factor'],
                                            patience=hyperparams['reduce_lr_patience'], min_lr=0.000001, verbose=1)
    lr_schedule = callbacks.LearningRateScheduler(lambda epoch, lr:
                                                  lr if (epoch + 1) % hyperparams['scheduled_reduce_lr_freq'] != 0 else
                                                  lr * hyperparams['scheduled_reduce_lr_factor'], verbose=1)
    callbacks_dict = {}

    # callbacks_dict = {'weights_history': weights_history,
    #                   'lr_history': lr_history,
    #                   #'freeze_layer': freeze_layer,
    #                   'reduce_lr_plateau': reduce_lr,
    #                   'lr_schedule': lr_schedule}

    if save:
        callbacks_dict['csv_logger'] = keras.callbacks.CSVLogger(store_path + 'metricHistory.csv',separator=",",append=True)

    logging.info('Train...\n')

    history = model.fit(data_generator_train,
                                  epochs=epochs, initial_epoch=start_epoch,
                                  class_weight=class_weight,
                                  validation_data=data_generator_valid,
                                  verbose=verbose,
                                  workers=workers,
                                  callbacks=callbacks_dict.values(),
                                  use_multiprocessing=False)

    return model, history


def train(user_level_data, subjects_split, save, store_path,
          hyperparams, hyperparams_features,
          dataset_type,
          model_type,
          logger=None,
          validation_set='valid',
          version=0, epochs=50, start_epoch=0,
          session=None, model=None, transfer_layer=False):
    if not logger:
        logger = logging.getLogger('training')
        ch = logging.StreamHandler(sys.stdout)
        # create formatter
        formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)

        network_type = 'lstm'
        hierarch_type = 'hierarchical'
        for feature in ['LIWC', 'emotions', 'numerical_dense_layer', 'sparse_feat_dense_layer', 'user_encoded']:
            if feature in hyperparams['ignore_layer']:
                network_type += "no%s" % feature

        model_path = 'models/%s_%s_%s%d' % (network_type, dataset_type, hierarch_type, version)

        logger.info("Initializing datasets...\n")
        data_generator_train, data_generator_valid = initialize_datasets(user_level_data, subjects_split,
                                                                         hyperparams, hyperparams_features,
                                                                         validation_set=validation_set)

        model = initialize_model(hyperparams, hyperparams_features, model_type,
                                    session=session, transfer=transfer_layer)

        print(model_path)
        logger.info("Training model...\n")

        model, history = train_model(model, hyperparams, save, store_path,
                                     data_generator_train, data_generator_valid,
                                     epochs=epochs, start_epoch=start_epoch,
                                     class_weight={0: 1, 1: hyperparams['positive_class_weight']},
                                     model_path=model_path, workers=1,
                                     validation_set=validation_set)
        return model, history
