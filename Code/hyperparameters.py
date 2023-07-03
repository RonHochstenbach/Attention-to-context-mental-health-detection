root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"
#root_dir = "/content/drive/MyDrive/Thesis/Data"  #when cloning for colab

import json
hyperparams_features = {
    "max_features": 20002,
    "embedding_dim": 300,
    "vocabulary_path": root_dir + '/Resources/vocabulary.pkl',
    "nrc_lexicon_path" : root_dir + "/Resources/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",
    "liwc_path": root_dir + '/Resources/LIWC2007.dic',
    "stopwords_path": root_dir + '/Resources/stopwords.txt',
    "embeddings_path": root_dir + "/Resources/glove.840B.300d.txt",
    #"liwc_words_cached": "data/liwc_categories_for_vocabulary_erisk_clpsych_stop_20K.pkl"
    "BERT_path": root_dir + '/Resources/BERT-base-uncased/'
}

hyperparams = {
    "trainable_embeddings": True,

    "lstm_units": 128,

    "dense_bow_units": 20,
    "dense_numerical_units": 20,

    "lstm_units_user": 32,

    "transfer_units": 20,

    "num_heads": 3,
    "key_dim": 128,
    "num_layers":4,
    "use_positional_encodings": True,

    "dropout": 0.3,             #Appendix uban
    "l2_dense": 0.00001,        #Appendix uban (?)
    "l2_embeddings": 0.00001,   #Appendix uban (?)
    "norm_momentum": 0.1,

    "ignore_layer": ["bert_layer"],

    "decay": 0.001,
    "lr": 0.0003,                   #appendix uban 0.0001
    "reduce_lr_factor": 0.8,        #originally 0.5
    "reduce_lr_patience": 5,        #originally 55
    "scheduled_reduce_lr_freq": 5,  #originally: 95
    "scheduled_reduce_lr_factor": 0.5,
    "freeze_patience": 2000,
    "threshold": 0.5,
    "early_stopping_patience": 5,

    "positive_class_weight": 3,     #github history https://github.com/ananana/mental-disorders/commit/8a7f12d470ede3c850eb344c6e53fd660ea32908 (2_

    "maxlen": 512,
    "posts_per_user": None,
    "post_groups_per_user": None,
    "posts_per_group": 50,
    "batch_size": 32,   #normally 32
    "padding": "pre",
    "hierarchical": True,
    "sample_seqs": False,
    "sampling_distr": "exp",

}

# with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Resources/config.json', 'w') as file:
#     json.dump(hyperparams_features, file)
