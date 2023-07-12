import pandas as pd
import tensorflow as tf
from nltk.tokenize import TweetTokenizer
import re
from collections import Counter
import pickle
from transformers import TFBertModel, TFRobertaModel

from hyperparameters import hyperparams_features, hyperparams
from resource_loader import load_NRC, readDict, load_stopwords, load_LIWC

def tokenize_tweets(t, stop=True):
    tweet_tokenizer = TweetTokenizer()
    tokens = tweet_tokenizer.tokenize(t.lower())
    tokens_clean = [re.sub("^#", "", token) for token in tokens]
    tokens_clean = [token for token in tokens_clean
                    if re.match("^[a-z']*$", token)]
    if not stop:
        tokens_clean = [token for token in tokens
                        if (token not in sw)]
    return tokens_clean



def tokenize_fields(writings_df, tokenize_fct, columns=['title', 'text']):
    cnt = 0
    for c in columns:
        writings_df['tokenized_%s' % c] = writings_df['%s' % c].apply(lambda t: tokenize_fct(t)
                                                                if type(t)==str and t else None)

        writings_df['%s_len' % c] = writings_df['tokenized_%s' % c].apply(lambda t: len(t)
                                                                    if type(t)==list and t else None)

    return writings_df

def shorten_text(text, length):
    if isinstance(text, str) and len(text.split()) > length:
        print(f"shortened a string which was length {len(text.split())} Now it is length {len(' '.join(text.split()[:length]).split())}")
        print(' '.join(text.split()[:length]))
        return ' '.join(text.split()[:length])
    else:
        return text

def build_vocabulary(writings_df):
    # Build vocabulary
    vocabulary_all = {}
    word_freqs = Counter()

    for text in writings_df.tokenized_text:
        word_freqs.update(text)

    if 'tokenized_title' in writings_df.columns:
        for text in writings_df.tokenized_title:
            word_freqs.update(text)
    i = 1
    print(len(word_freqs))
    for w, f in word_freqs.most_common(hyperparams_features['max_features'] + 1000): #add some space for len<1 words
        if len(vocabulary_all) < hyperparams_features['max_features'] - 2:  # keeping voc_size-1 for unk
            if len(w) < 1:
                continue
            vocabulary_all[w] = i
            i += 1
        else:
            break

    print(len(vocabulary_all))
    print("Average number of posts per user", writings_df.groupby('subject').count().title.mean())
    print("Average number of comments per user", writings_df.groupby('subject').count().text.mean())

    with open("/Users/ronhochstenbach/Desktop/Thesis/Data/Resources/vocabulary.pkl", 'wb') as f:
        pickle.dump(vocabulary_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    return vocabulary_all

def feature_sizes():
    emotions = load_NRC(hyperparams_features['nrc_lexicon_path'])
    liwc_categories = load_LIWC(hyperparams_features['liwc_path'])
    stopwords_list = load_stopwords(hyperparams_features['stopwords_path'])

    return_dict = {"emotions_dim": len(emotions),
                   "liwc_categories_dim": len(liwc_categories),
                   "stopwords_dim": len(stopwords_list)}

    return return_dict

def create_embeddings(inputs, model_type):

    ids = inputs[0].numpy()
    masks = inputs[1].numpy()
    print(type(ids))

    #extracting the last four hidden states and summing them
    if model_type == "HAN_BERT":
        BERT_embedding_layer = TFBertModel.from_pretrained('bert-base-uncased')(
                                                            ids, attention_mask=masks,
                                                            output_hidden_states=True, return_dict=True)[
                                                                                   'hidden_states'][-4:]
    elif model_type == "HAN_RoBERTa":
        BERT_embedding_layer = TFRobertaModel.from_pretrained('roberta-base')(
                                                            ids, attention_mask=masks,
                                                            output_hidden_states=True, return_dict=True)[
                                                                                   'hidden_states'][-4:]
    else:
        Exception("Unknown model type!")

    embedding = tf.add_n(BERT_embedding_layer)

    return embedding




