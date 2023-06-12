import logging
from collections import Counter


from resource_loader import load_vocabulary, load_NRC, load_LIWC
from read_erisk_data import read_texts_2019, read_subject_writings, read_texts_2020

def load_data(task):
    root_dir = '/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data'

    if task == "Depression":
        datadirs_T1_2018 = {
            'train': ['train/positive_examples_anonymous_chunks/', 'train/negative_examples_anonymous_chunks/',
                      'test/'],
            'test': ['task 1 - depression (test split, train split is 2017 data)/']
        }
        datadir_root_T1_2018 = {
            'train': root_dir + '/2017/',
            'test': root_dir + '/2018/'
        }

        labels_files_T1_2018 = {
            'train': ['train/risk_golden_truth.txt', 'test/test_golden_truth.txt'],
            'test': ['task 1 - depression (test split, train split is 2017 data)/risk-golden-truth-test.txt']
        }

        writings_df = read_texts_2019(datadir_root_T1_2018,
                  datadirs_T1_2018,
                   labels_files_T1_2018,
                   chunked_subsets=['train', 'test'])

    elif task == "Anorexia":
        datadirs_T1_2019 = {
            'train': ['2018 test/', '2018 train/positive_examples/', '2018 train/negative_examples/'],
            'test': ['data/']
        }
        datadir_root_T1_2019 = {
            'train': root_dir + '/eRisk2019_T1/training data - t1/',
            'test': root_dir + '/eRisk2019_T1/test data - T1/'
        }

        labels_files_T1_2019 = {
            'train': ['2018 train/risk_golden_truth.txt', '2018 test/risk-golden-truth-test.txt'],
            'test': ['T1_erisk_golden_truth.txt']
        }

        writings_df = read_texts_2019(datadir_root_T1_2019,
                        datadirs_T1_2019,
                        labels_files_T1_2019)

    elif task == "Self-Harm":

        datadirs_T1_2020 = {
            'train': ['./data/'],
            'test': ['./DATA/']
        }
        datadir_root_T1_2020 = {
            'train': root_dir + '/eRISK2020_training_data/',
            'test': root_dir + '/T1/'
        }

        labels_files_T1_2020 = {
            'train': ['golden_truth.txt'],
            'test': ['T1_erisk_golden_truth.txt']
        }

        writings_df = read_texts_2020(datadir_root_T1_2020,
                                      datadirs_T1_2020,
                                      labels_files_T1_2020)

    else:
        raise Exception("Unknown task!")

    return writings_df

def load_erisk_data(writings_df, hyperparams_features, by_subset=True,
                    pronouns=["i", "me", "my", "mine", "myself"],
                    train_prop=0.7, valid_prop=0.3, test_slice=2,
                    nr_slices=5,
                    min_post_len=3, min_word_len=1,
                    user_level=True, labelcol='label', label_index=None,
                    logger=None):
    #     logger.debug("Loading data...\n")

    vocabulary = load_vocabulary(hyperparams_features['vocabulary_path'])
    #print(len(vocabulary))
    voc_size = hyperparams_features['max_features']
    emotion_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
    emotions = list(emotion_lexicon.keys())
    liwc_dict = load_LIWC(hyperparams_features['liwc_path'])
    liwc_categories = set(liwc_dict.keys())

    training_subjects = list(set(writings_df[writings_df['subset'] == 'train'].subject))
    test_subjects = list(set(writings_df[writings_df['subset'] == 'test'].subject))

    training_subjects = sorted(training_subjects)  # ensuring reproducibility
    valid_subjects_size = int(len(training_subjects) * valid_prop)
    valid_subjects = training_subjects[:valid_subjects_size]
    training_subjects = training_subjects[valid_subjects_size:]
    categories = [c for c in liwc_categories if c in writings_df.columns]
    #     logger.debug("%d training users, %d validation users, %d test users." % (
    #         len(training_subjects),
    #           len(valid_subjects),
    #           len(test_subjects)))
    subjects_split = {'train': training_subjects,
                      'valid': valid_subjects,
                      'test': test_subjects}

    user_level_texts = {}
    for row in writings_df.sort_values(by='date').itertuples():
        words = []
        raw_text = ""
        if hasattr(row, 'tokenized_title'):
            if row.tokenized_title:
                words.extend(row.tokenized_title)
                raw_text += row.title
        if hasattr(row, 'tokenized_text'):
            if row.tokenized_text:
                words.extend(row.tokenized_text)
                raw_text += row.text
        if not words or len(words) < min_post_len:
            #             logger.debug(row.subject)
            continue
        if labelcol == 'label':
            label = row.label
        liwc_categs = [getattr(row, categ) for categ in categories]
        if row.subject not in user_level_texts.keys():
            user_level_texts[row.subject] = {}
            user_level_texts[row.subject]['texts'] = [words]
            user_level_texts[row.subject]['label'] = label
            user_level_texts[row.subject]['liwc'] = [liwc_categs]
            user_level_texts[row.subject]['raw'] = [raw_text]
        else:
            user_level_texts[row.subject]['texts'].append(words)
            user_level_texts[row.subject]['liwc'].append(liwc_categs)
            user_level_texts[row.subject]['raw'].append(raw_text)

    return user_level_texts, subjects_split, vocabulary

#save datasets
# task = "Depression"
# load_data(task).to_pickle("/Users/ronhochstenbach/Desktop/Thesis/Data/Processed Data/df_" + task + ".pkl")


