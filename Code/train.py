from callbacks import FreezeLayer, WeightsHistory,LRHistory
from tensorflow.keras import callbacks
from metrics import Metrics
import logging, sys, os
import pickle
from data_generator import DataGenerator
from models import build_hierarchical_model
from resource_loader import load_NRC, load_LIWC, load_stopwords

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # When cudnn implementation not found, run this
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Note: when starting kernel, for gpu_available to be true, this needs to be run
# only reserve 1 GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

def get_network_type(hyperparams):
    if 'lstm' in hyperparams['ignore_layer']:
        network_type = 'cnn'
    else:
        network_type = 'lstm'
    if 'user_encoded' in hyperparams['ignore_layer']:
        if 'bert_layer' not in hyperparams['ignore_layer']:
            network_type = 'bert'
        else:
            network_type = 'extfeatures'
    if hyperparams['hierarchical']:
        hierarch_type = 'hierarchical'
    else:
        hierarch_type = 'seq'
    return network_type, hierarch_type


def train_model(model, hyperparams,
                data_generator_train, data_generator_valid,
                epochs, class_weight, start_epoch=0, workers=1,
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
    callbacks_dict = {'freeze_layer': freeze_layer, 'weights_history': weights_history,
                      'lr_history': lr_history,
                      'reduce_lr_plateau': reduce_lr,
                      'lr_schedule': lr_schedule}

    logging.info('Train...')

    history = model.fit_generator(data_generator_train,
                                  # steps_per_epoch=100,
                                  epochs=epochs, initial_epoch=start_epoch,
                                  class_weight=class_weight,
                                  validation_data=data_generator_valid,
                                  verbose=verbose,
                                  #               validation_split=0.3,
                                  workers=workers,
                                  use_multiprocessing=False,
                                  # max_queue_size=100,

                                  callbacks=[
                                                # callbacks.ModelCheckpoint(filepath='%s_best.h5' % model_path, verbose=1,
                                                #                           save_best_only=True, save_weights_only=True),
                                                # callbacks.EarlyStopping(patience=hyperparams['early_stopping_patience'],
                                                #                        restore_best_weights=True)
                                            ] + [
                                                callbacks_dict[c] for c in [
                                          # 'weights_history',
                                      ]])
    return model, history


