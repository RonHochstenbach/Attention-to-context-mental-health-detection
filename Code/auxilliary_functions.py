import pandas as pd
from nltk.tokenize import TweetTokenizer
import re
from collections import Counter
import pickle
from transformers import BertTokenizer

from hyperparameters import hyperparams_features, hyperparams

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

def tokenize_fields_bert(writings_df, columns=['title', 'text']):
    cnt = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for c in columns:
        print('Tokenizing ids for %s' % c)
        writings_df['tokenized_%s_id' % c] = writings_df['%s' % c].apply(lambda t: tokenizer(t,
                                                                            add_special_tokens=True, max_length=hyperparams['maxlen'],
                                                                            padding='max_length', truncation=True,
                                                                            return_attention_mask=True,
                                                                            return_tensors='tf')['input_ids']
                                                                if type(t)==str and t else None)
        print('Tokenizing attention masks for %s' % c)
        writings_df['tokenized_%s_attnmask' % c] = writings_df['%s' % c].apply(lambda t: tokenizer(t,
                                                                            add_special_tokens=True, max_length=hyperparams['maxlen'],
                                                                            padding='max_length', truncation=True,
                                                                            return_attention_mask=True,
                                                                            return_tensors='tf')['attention_mask']
                                                                if type(t)==str and t else None)
        print('Appending lengths for %s' % c)
        writings_df['%s_len' % c] = writings_df['tokenized_%s_id' % c].apply(lambda t: len(t)
                                                                if type(t)==list and t else None)

    return writings_df

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
