import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Lambda, BatchNormalization, TimeDistributed, \
    Input, concatenate, Flatten, RepeatVector, Activation, Multiply, Permute, MultiHeadAttention
from keras_nlp.layers import SinePositionEncoding
from keras import regularizers
from keras import backend as K
from keras.metrics import AUC, Precision, Recall
from metrics import Metrics
from resource_loader import load_embeddings
from transformers import TFBertModel, TFRobertaModel
from auxilliary_functions import create_embeddings


def build_HAN(hyperparams, hyperparams_features,
                             emotions_dim, stopwords_list_dim, liwc_categories_dim,
                             ignore_layer=[]):

    embedding_matrix = load_embeddings(hyperparams_features['embeddings_path'],
                                       hyperparams_features['embedding_dim'],
                                       hyperparams_features['vocabulary_path'])

    # Post/sentence representation - word sequence
    tokens_features = Input(shape=(hyperparams['maxlen'],), name='word_seq')
    embedding_layer = Embedding(hyperparams_features['max_features'],
                                hyperparams_features['embedding_dim'],
                                input_length=hyperparams['maxlen'],
                                embeddings_regularizer=regularizers.l2(hyperparams['l2_embeddings']),
                                weights=[embedding_matrix],
                                trainable=hyperparams['trainable_embeddings'],
                                name='embeddings_layer')(tokens_features)

    embedding_layer = Dropout(hyperparams['dropout'], name='embedding_dropout')(embedding_layer)

    lstm_layers = LSTM(hyperparams['lstm_units'],
                       return_sequences='attention' not in ignore_layer,
                       name='LSTM_layer')(embedding_layer)

    # Attention
    if 'attention' not in ignore_layer:
        attention_layer = Dense(1, activation='tanh', name='attention')
        attention = attention_layer(lstm_layers)
        attention = Flatten()(attention)
        attention_output = Activation('softmax')(attention)
        attention = RepeatVector(hyperparams['lstm_units'])(attention_output)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([lstm_layers, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units'],)
                                     )(sent_representation)
    else:
        sent_representation = lstm_layers

    if 'batchnorm' not in ignore_layer:
        sent_representation = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                 name='sent_repr_norm')(sent_representation)

    sent_representation = Dropout(hyperparams['dropout'], name='sent_repr_dropout')(sent_representation)

    # Other features
    numerical_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        emotions_dim + 1 + liwc_categories_dim
    ), name='numeric_input_hist')  # emotions and pronouns
    sparse_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        stopwords_list_dim
    ), name='sparse_input_hist')  # stopwords

    posts_history_input = Input(shape=(hyperparams['posts_per_group'],
                                       hyperparams['maxlen']
                                       ), name='hierarchical_word_seq_input')

    # Hierarchy
    sentEncoder = Model(inputs=tokens_features,
                        outputs=sent_representation, name='sentEncoder')
    sentEncoder.summary()

    user_encoder = TimeDistributed(sentEncoder, name='user_encoder')(posts_history_input)

    dense_layer_sparse = Dense(units=hyperparams['dense_bow_units'],
                               name='sparse_feat_dense_layer', activation='relu',
                               kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                               )
    dense_layer_sparse_user = TimeDistributed(dense_layer_sparse,
                                              name='sparse_dense_layer_user')(sparse_features_history)

    dense_layer_numerical = Dense(units=hyperparams['dense_numerical_units'],
                                  name='numerical_feat_dense_layer', activation='relu',
                                  kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                                  )
    dense_layer_numerical_user = TimeDistributed(dense_layer_numerical,
                                                 name='numerical_dense_layer_user')(numerical_features_history)


    # Concatenate features
    if 'batchnorm' not in ignore_layer:
        dense_layer_numerical_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                        name='numerical_features_norm')(dense_layer_numerical_user)

        dense_layer_sparse_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                     name='sparse_features_norm')(dense_layer_sparse_user)


    all_layers = {
        'user_encoded': user_encoder,

        'numerical_dense_layer': dense_layer_numerical_user,

        'sparse_feat_dense_layer': dense_layer_sparse_user,
    }

    layers_to_merge = [l for n, l in all_layers.items() if n not in ignore_layer]
    if len(layers_to_merge) == 1:
        merged_layers = layers_to_merge[0]
    else:
        merged_layers = concatenate(layers_to_merge)

    lstm_user_layers = LSTM(hyperparams['lstm_units_user'],
                            return_sequences='attention_user' not in ignore_layer,
                            name='LSTM_layer_user')(merged_layers)

    # Attention
    if 'attention_user' not in ignore_layer:
        attention_user_layer = Dense(1, activation='tanh', name='attention_user')
        attention_user = attention_user_layer(lstm_user_layers)
        attention_user = Flatten()(attention_user)
        attention_user_output = Activation('softmax')(attention_user)
        attention_user = RepeatVector(hyperparams['lstm_units_user'])(attention_user_output)
        attention_user = Permute([2, 1])(attention_user)

        user_representation = Multiply()([lstm_user_layers, attention_user])
        user_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units_user'],))(user_representation)

    else:
        user_representation = lstm_user_layers

    user_representation = Dropout(hyperparams['dropout'], name='user_repr_dropout')(user_representation)

    output_layer = Dense(1, activation='sigmoid',
                         name='output_layer',
                         kernel_regularizer=regularizers.l2(hyperparams['l2_dense'])
                         )(user_representation)

    hierarchical_model = Model(inputs=[posts_history_input,
                                       numerical_features_history, sparse_features_history,
                                       ],
                               outputs=output_layer)

    # metrics_class = Metrics(threshold=hyperparams['threshold'])
    # hierarchical_model.compile(hyperparams['optimizer'], K.binary_crossentropy,
    #                            metrics=[AUC(), Precision(),metrics_class.precision_m, Recall(),metrics_class.recall_m, metrics_class.f1_m ])
    hierarchical_model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[AUC(), Precision(), Recall()])


    return hierarchical_model

def build_HAN_BERT(hyperparams, hyperparams_features, model_type,
                             emotions_dim, stopwords_list_dim, liwc_categories_dim,
                             ignore_layer=[]):

    # Post/sentence representation - word sequence
    tokens_features_ids = Input(shape=(hyperparams['maxlen'],), name='word_seq_ids',dtype=tf.int32)
    tokens_features_attnmasks = Input(shape=(hyperparams['maxlen'],), name='word_seq_attnmasks',dtype=tf.int32)

    #extracting the last four hidden states and summing them
    if model_type == "HAN_BERT":
        BERT_embedding_layer = TFBertModel.from_pretrained('bert-small')(
                                                            tokens_features_ids, attention_mask=tokens_features_attnmasks,
                                                            output_hidden_states=True, return_dict=True)[
                                                                                   'hidden_states'][-4:]
    elif model_type == "HAN_RoBERTa":
        BERT_embedding_layer = TFRobertaModel.from_pretrained('roberta-base')(
                                                            tokens_features_ids, attention_mask=tokens_features_attnmasks,
                                                            output_hidden_states=True, return_dict=True)[
                                                                                   'hidden_states'][-4:]
    else:
        Exception("Unknown model type!")

    #embedding_layer = Lambda(lambda x: tf.add_n([layer for layer in x]))(BERT_embedding_layer)
    embedding_layer = Lambda(lambda x: tf.add_n(x))(BERT_embedding_layer)

    embedding_layer = Dropout(hyperparams['dropout'], name='embedding_dropout')(embedding_layer)

    lstm_layers = LSTM(hyperparams['lstm_units'],
                       return_sequences='attention' not in ignore_layer,
                       name='LSTM_layer')(embedding_layer)

    # Attention
    if 'attention' not in ignore_layer:
        attention_layer = Dense(1, activation='tanh', name='attention')
        attention = attention_layer(lstm_layers)
        attention = Flatten()(attention)
        attention_output = Activation('softmax')(attention)
        attention = RepeatVector(hyperparams['lstm_units'])(attention_output)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([lstm_layers, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units'],)
                                     )(sent_representation)
    else:
        sent_representation = lstm_layers

    if 'batchnorm' not in ignore_layer:
        sent_representation = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                 name='sent_repr_norm')(sent_representation)

    sent_representation = Dropout(hyperparams['dropout'], name='sent_repr_dropout')(sent_representation)

    # Other features
    numerical_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        emotions_dim + 1 + liwc_categories_dim
    ), name='numeric_input_hist')  # emotions and pronouns
    sparse_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        stopwords_list_dim
    ), name='sparse_input_hist')  # stopwords

    post_history_ids = Input(shape=(hyperparams['posts_per_group'],
                                       hyperparams['maxlen']
                                       ), name='hierarchical_word_seq_input_ids')
    post_history_attnmasks = Input(shape=(hyperparams['posts_per_group'],
                                       hyperparams['maxlen']
                                       ), name='hierarchical_word_seq_input_attnmasks')

    # Hierarchy
    sentEncoder = Model(inputs=[tokens_features_ids,tokens_features_attnmasks],
                        outputs=sent_representation, name='sentEncoder')

    #Set BERT/RoBERTa model to non-trainable
    [setattr(layer, 'trainable', False) for layer in sentEncoder.layers if layer._name == 'tf_bert_model' or layer._name == 'tf_roberta_model']

    sentEncoder.summary()

    user_encoder = TimeDistributed(sentEncoder, name='user_encoder')([post_history_ids, post_history_attnmasks])

    dense_layer_sparse = Dense(units=hyperparams['dense_bow_units'],
                               name='sparse_feat_dense_layer', activation='relu',
                               kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                               )
    dense_layer_sparse_user = TimeDistributed(dense_layer_sparse,
                                              name='sparse_dense_layer_user')(sparse_features_history)

    dense_layer_numerical = Dense(units=hyperparams['dense_numerical_units'],
                                  name='numerical_feat_dense_layer', activation='relu',
                                  kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                                  )
    dense_layer_numerical_user = TimeDistributed(dense_layer_numerical,
                                                 name='numerical_dense_layer_user')(numerical_features_history)


    # Concatenate features
    if 'batchnorm' not in ignore_layer:
        dense_layer_numerical_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                        name='numerical_features_norm')(dense_layer_numerical_user)

        dense_layer_sparse_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                     name='sparse_features_norm')(dense_layer_sparse_user)


    all_layers = {
        'user_encoded': user_encoder,

        'numerical_dense_layer': dense_layer_numerical_user,

        'sparse_feat_dense_layer': dense_layer_sparse_user,
    }

    layers_to_merge = [l for n, l in all_layers.items() if n not in ignore_layer]
    if len(layers_to_merge) == 1:
        merged_layers = layers_to_merge[0]
    else:
        merged_layers = concatenate(layers_to_merge)

    lstm_user_layers = LSTM(hyperparams['lstm_units_user'],
                            return_sequences='attention_user' not in ignore_layer,
                            name='LSTM_layer_user')(merged_layers)

    # Attention
    if 'attention_user' not in ignore_layer:
        attention_user_layer = Dense(1, activation='tanh', name='attention_user')
        attention_user = attention_user_layer(lstm_user_layers)
        attention_user = Flatten()(attention_user)
        attention_user_output = Activation('softmax')(attention_user)
        attention_user = RepeatVector(hyperparams['lstm_units_user'])(attention_user_output)
        attention_user = Permute([2, 1])(attention_user)

        user_representation = Multiply()([lstm_user_layers, attention_user])
        user_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units_user'],))(user_representation)

    else:
        user_representation = lstm_user_layers

    user_representation = Dropout(hyperparams['dropout'], name='user_repr_dropout')(user_representation)

    output_layer = Dense(1, activation='sigmoid',
                         name='output_layer',
                         kernel_regularizer=regularizers.l2(hyperparams['l2_dense'])
                         )(user_representation)

    hierarchical_model_BERT = Model(inputs=[post_history_ids, post_history_attnmasks,
                                       numerical_features_history, sparse_features_history,
                                       ],
                               outputs=output_layer)

    hierarchical_model_BERT.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[AUC(), Precision(), Recall()])

    return hierarchical_model_BERT

def build_HSAN(hyperparams, hyperparams_features,
                             emotions_dim, stopwords_list_dim, liwc_categories_dim,
                             ignore_layer=[]):

    embedding_matrix = load_embeddings(hyperparams_features['embeddings_path'],
                                       hyperparams_features['embedding_dim'],
                                       hyperparams_features['vocabulary_path'])

    # Post/sentence representation - word sequence
    tokens_features = Input(shape=(hyperparams['maxlen'],), name='word_seq')
    embedding_layer = Embedding(hyperparams_features['max_features'],
                                hyperparams_features['embedding_dim'],
                                input_length=hyperparams['maxlen'],
                                embeddings_regularizer=regularizers.l2(hyperparams['l2_embeddings']),
                                weights=[embedding_matrix],
                                trainable=hyperparams['trainable_embeddings'],
                                name='embeddings_layer')(tokens_features)

    embedding_layer = Dropout(hyperparams['dropout'], name='embedding_dropout')(embedding_layer)

    lstm_layers = LSTM(hyperparams['lstm_units'],
                       return_sequences='attention' not in ignore_layer,
                       name='LSTM_layer')(embedding_layer)

    # Attention
    if 'attention' not in ignore_layer:
        attention_layer = Dense(1, activation='tanh', name='attention')
        attention = attention_layer(lstm_layers)
        attention = Flatten()(attention)
        attention_output = Activation('softmax')(attention)
        attention = RepeatVector(hyperparams['lstm_units'])(attention_output)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([lstm_layers, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units'],)
                                     )(sent_representation)
    else:
        sent_representation = lstm_layers

    if 'batchnorm' not in ignore_layer:
        sent_representation = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                 name='sent_repr_norm')(sent_representation)

    sent_representation = Dropout(hyperparams['dropout'], name='sent_repr_dropout')(sent_representation)

    # Other features
    numerical_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        emotions_dim + 1 + liwc_categories_dim
    ), name='numeric_input_hist')  # emotions and pronouns
    sparse_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        stopwords_list_dim
    ), name='sparse_input_hist')  # stopwords

    posts_history_input = Input(shape=(hyperparams['posts_per_group'],
                                       hyperparams['maxlen']
                                       ), name='hierarchical_word_seq_input')

    # Hierarchy
    sentEncoder = Model(inputs=tokens_features,
                        outputs=sent_representation, name='sentEncoder')
    sentEncoder.summary()

    user_encoder = TimeDistributed(sentEncoder, name='user_encoder')(posts_history_input)

    dense_layer_sparse = Dense(units=hyperparams['dense_bow_units'],
                               name='sparse_feat_dense_layer', activation='relu',
                               kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                               )
    dense_layer_sparse_user = TimeDistributed(dense_layer_sparse,
                                              name='sparse_dense_layer_user')(sparse_features_history)

    dense_layer_numerical = Dense(units=hyperparams['dense_numerical_units'],
                                  name='numerical_feat_dense_layer', activation='relu',
                                  kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                                  )
    dense_layer_numerical_user = TimeDistributed(dense_layer_numerical,
                                                 name='numerical_dense_layer_user')(numerical_features_history)


    # Concatenate features
    if 'batchnorm' not in ignore_layer:
        dense_layer_numerical_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                        name='numerical_features_norm')(dense_layer_numerical_user)

        dense_layer_sparse_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                     name='sparse_features_norm')(dense_layer_sparse_user)


    all_layers = {
        'user_encoded': user_encoder,

        'numerical_dense_layer': dense_layer_numerical_user,

        'sparse_feat_dense_layer': dense_layer_sparse_user,
    }

    layers_to_merge = [l for n, l in all_layers.items() if n not in ignore_layer]
    if len(layers_to_merge) == 1:
        merged_layers = layers_to_merge[0]
    else:
        merged_layers = concatenate(layers_to_merge)

    if hyperparams["use_positional_encodings"]:
        positional_encoding = SinePositionEncoding(name='positional_encoding')(merged_layers)

        attention_input = Lambda(lambda x: tf.add_n(x), name='attention_input')([merged_layers, positional_encoding])
    else:
        attention_input = merged_layers

    user_representation_attention = attention_input

    for layer in range(hyperparams["num_layers"]):
        user_representation_attention = MultiHeadAttention(num_heads = hyperparams["num_heads"],
                                                           key_dim=hyperparams["key_dim"],
                                                           dropout=hyperparams["dropout"],
                                                           name=f"MH-attention_layer_{layer}")(
                                                            user_representation_attention,user_representation_attention)

    # user_representation_attention = Lambda(lambda xin: K.sum(xin, axis=1),
    #                              output_shape=(hyperparams['lstm_units']+hyperparams["dense_bow_units"]+
    #                                            hyperparams["dense_numerical_units"],)
    #                              )(user_representation_attention)

    user_representation_attention = Permute([2,1])(user_representation_attention)
    user_representation_attention = Dense(1, activation='sigmoid',
                         name='user_att_dense',
                         kernel_regularizer=regularizers.l2(hyperparams['l2_dense'])
                         )(user_representation_attention)
    user_representation_attention = Lambda (lambda x: tf.squeeze(x, axis=2))(user_representation_attention)

    user_representation = Dropout(hyperparams['dropout'], name='user_repr_dropout')(user_representation_attention)

    output_layer = Dense(1, activation='sigmoid',
                         name='output_layer',
                         kernel_regularizer=regularizers.l2(hyperparams['l2_dense'])
                         )(user_representation)

    hierarchical_model = Model(inputs=[posts_history_input,
                                       numerical_features_history, sparse_features_history,
                                       ],
                               outputs=output_layer)

    hierarchical_model.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[AUC(), Precision(), Recall()])

    return hierarchical_model

def build_HAN_BERT_test(hyperparams, hyperparams_features, model_type,
                             emotions_dim, stopwords_list_dim, liwc_categories_dim,
                             ignore_layer=[]):

    # Post/sentence representation - word sequence
    tokens_features_ids = Input(shape=(hyperparams['maxlen'],), name='word_seq_ids',dtype=tf.int32)
    tokens_features_attnmasks = Input(shape=(hyperparams['maxlen'],), name='word_seq_attnmasks',dtype=tf.int32)

    #extracting the last four hidden states and summing them
    # if model_type == "HAN_BERT":
    #     BERT_embedding_layer = TFBertModel.from_pretrained('bert-base-uncased')(
    #                                                         tokens_features_ids, attention_mask=tokens_features_attnmasks,
    #                                                         output_hidden_states=True, return_dict=True)[
    #                                                                                'hidden_states'][-4:]
    # elif model_type == "HAN_RoBERTa":
    #     BERT_embedding_layer = TFRobertaModel.from_pretrained('roberta-base')(
    #                                                         tokens_features_ids, attention_mask=tokens_features_attnmasks,
    #                                                         output_hidden_states=True, return_dict=True)[
    #                                                                                'hidden_states'][-4:]
    # else:
    #     Exception("Unknown model type!")

    embedding_layer = Lambda(lambda x: create_embeddings(x, model_type))((tokens_features_ids,tokens_features_attnmasks))

    embedding_layer = Dropout(hyperparams['dropout'], name='embedding_dropout')(embedding_layer)

    lstm_layers = LSTM(hyperparams['lstm_units'],
                       return_sequences='attention' not in ignore_layer,
                       name='LSTM_layer')(embedding_layer)

    # Attention
    if 'attention' not in ignore_layer:
        attention_layer = Dense(1, activation='tanh', name='attention')
        attention = attention_layer(lstm_layers)
        attention = Flatten()(attention)
        attention_output = Activation('softmax')(attention)
        attention = RepeatVector(hyperparams['lstm_units'])(attention_output)
        attention = Permute([2, 1])(attention)

        sent_representation = Multiply()([lstm_layers, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units'],)
                                     )(sent_representation)
    else:
        sent_representation = lstm_layers

    if 'batchnorm' not in ignore_layer:
        sent_representation = BatchNormalization(axis=1, momentum=hyperparams['norm_momentum'],
                                                 name='sent_repr_norm')(sent_representation)

    sent_representation = Dropout(hyperparams['dropout'], name='sent_repr_dropout')(sent_representation)

    # Other features
    numerical_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        emotions_dim + 1 + liwc_categories_dim
    ), name='numeric_input_hist')  # emotions and pronouns
    sparse_features_history = Input(shape=(
        hyperparams['posts_per_group'],
        stopwords_list_dim
    ), name='sparse_input_hist')  # stopwords

    post_history_ids = Input(shape=(hyperparams['posts_per_group'],
                                       hyperparams['maxlen']
                                       ), name='hierarchical_word_seq_input_ids')
    post_history_attnmasks = Input(shape=(hyperparams['posts_per_group'],
                                       hyperparams['maxlen']
                                       ), name='hierarchical_word_seq_input_attnmasks')

    # Hierarchy
    sentEncoder = Model(inputs=[tokens_features_ids,tokens_features_attnmasks],
                        outputs=sent_representation, name='sentEncoder')

    #Set BERT/RoBERTa model to non-trainable
    [setattr(layer, 'trainable', False) for layer in sentEncoder.layers if layer._name == 'tf_bert_model' or layer._name == 'tf_roberta_model']

    sentEncoder.summary()

    user_encoder = TimeDistributed(sentEncoder, name='user_encoder')([post_history_ids, post_history_attnmasks])

    dense_layer_sparse = Dense(units=hyperparams['dense_bow_units'],
                               name='sparse_feat_dense_layer', activation='relu',
                               kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                               )
    dense_layer_sparse_user = TimeDistributed(dense_layer_sparse,
                                              name='sparse_dense_layer_user')(sparse_features_history)

    dense_layer_numerical = Dense(units=hyperparams['dense_numerical_units'],
                                  name='numerical_feat_dense_layer', activation='relu',
                                  kernel_regularizer=regularizers.l2(hyperparams['l2_dense']),
                                  )
    dense_layer_numerical_user = TimeDistributed(dense_layer_numerical,
                                                 name='numerical_dense_layer_user')(numerical_features_history)


    # Concatenate features
    if 'batchnorm' not in ignore_layer:
        dense_layer_numerical_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                        name='numerical_features_norm')(dense_layer_numerical_user)

        dense_layer_sparse_user = BatchNormalization(axis=2, momentum=hyperparams['norm_momentum'],
                                                     name='sparse_features_norm')(dense_layer_sparse_user)


    all_layers = {
        'user_encoded': user_encoder,

        'numerical_dense_layer': dense_layer_numerical_user,

        'sparse_feat_dense_layer': dense_layer_sparse_user,
    }

    layers_to_merge = [l for n, l in all_layers.items() if n not in ignore_layer]
    if len(layers_to_merge) == 1:
        merged_layers = layers_to_merge[0]
    else:
        merged_layers = concatenate(layers_to_merge)

    lstm_user_layers = LSTM(hyperparams['lstm_units_user'],
                            return_sequences='attention_user' not in ignore_layer,
                            name='LSTM_layer_user')(merged_layers)

    # Attention
    if 'attention_user' not in ignore_layer:
        attention_user_layer = Dense(1, activation='tanh', name='attention_user')
        attention_user = attention_user_layer(lstm_user_layers)
        attention_user = Flatten()(attention_user)
        attention_user_output = Activation('softmax')(attention_user)
        attention_user = RepeatVector(hyperparams['lstm_units_user'])(attention_user_output)
        attention_user = Permute([2, 1])(attention_user)

        user_representation = Multiply()([lstm_user_layers, attention_user])
        user_representation = Lambda(lambda xin: K.sum(xin, axis=1),
                                     output_shape=(hyperparams['lstm_units_user'],))(user_representation)

    else:
        user_representation = lstm_user_layers

    user_representation = Dropout(hyperparams['dropout'], name='user_repr_dropout')(user_representation)

    output_layer = Dense(1, activation='sigmoid',
                         name='output_layer',
                         kernel_regularizer=regularizers.l2(hyperparams['l2_dense'])
                         )(user_representation)

    hierarchical_model_BERT = Model(inputs=[post_history_ids, post_history_attnmasks,
                                       numerical_features_history, sparse_features_history,
                                       ],
                               outputs=output_layer)

    hierarchical_model_BERT.compile(hyperparams['optimizer'], K.binary_crossentropy,
                               metrics=[AUC(), Precision(), Recall()])

    return hierarchical_model_BERT

