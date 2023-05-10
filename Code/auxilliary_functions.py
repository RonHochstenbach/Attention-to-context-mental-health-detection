from nltk.tokenize import TweetTokenizer
import re

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

