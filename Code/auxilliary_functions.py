from nltk.tokenize import TweetTokenizer
import re
from collections import Counter
import pickle

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
    for c in columns:
        writings_df['tokenized_%s' % c] = writings_df['%s' % c].apply(lambda t: tokenize_fct(t)
                                                                if type(t)==str and t else None)
        writings_df['%s_len' % c] = writings_df['tokenized_%s' % c].apply(lambda t: len(t)
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
    for w, f in word_freqs.most_common(20002 - 2):  # keeping voc_size-1 for unk
        if len(w) < 1:
            continue
        vocabulary_all[w] = i
        i += 1
    print(len(vocabulary_all))
    print("Average number of posts per user", writings_df.groupby('subject').count().title.mean())
    print("Average number of comments per user", writings_df.groupby('subject').count().text.mean())

    with open("/Users/ronhochstenbach/Desktop/Thesis/Data/Resources/vocabulary.pkl", 'wb') as f:
        pickle.dump(vocabulary_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    return vocabulary_all
