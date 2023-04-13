from resource_loader import load_vocabulary, load_NRC

def load_erisk_data(writings_df, hyperparams_features, by_subset=True,
                    pronouns = ["i", "me", "my", "mine", "myself"],
                    train_prop=0.7, valid_prop=0.3, test_slice=2,
                    nr_slices=5,
                    min_post_len=3, min_word_len=1,
                    user_level=True, labelcol='label', label_index=None,
                   logger=None):

    vocabulary = load_vocabulary(hyperparams_features['vocabulary_path'])
    vocab_size = hyperparams_features(['max_features'])

    emotion_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
    emotions = list(emotion_lexicon.keys())

