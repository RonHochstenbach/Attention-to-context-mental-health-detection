import logging
from collections import Counter


from resource_loader import load_vocabulary, load_NRC, load_LIWC
from read_erisk_data import read_texts_2019

def load_data(task):
    root_dir = '/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data'

    if task == "Depression":
        datadirs_T1_2018 = {
            'train': ['train/positive_examples_anonymous_chunks/', 'train/positive_examples_anonymous_chunks/',
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



    return writings_df


def load_erisk_data(writings_df, voc_size, emotion_lexicon, liwc_categories, logger, emotions=
['anger', 'anticipation', 'disgust', 'fear', 'joy',
 'negative', 'positive', 'sadness', 'surprise', 'trust'], by_subset=True,
                    pronouns=["i", "me", "my", "mine", "myself"],
                    train_prop=0.7, valid_prop=0.3, test_slice=2,
                    nr_slices=5,
                    min_post_len=3, min_word_len=1,
                    user_level=True, vocabulary=None, labelcol='label', label_index=None):
    logger.debug("Loading data...\n")

    ## Build vocabulary
    vocabulary_all = {}
    word_freqs = Counter()

    for words in writings_df.tokenized_text:
        word_freqs.update(words)
    if 'tokenized_title' in writings_df.columns:
        for words in writings_df.tokenized_title:
            word_freqs.update(words)
    i = 1
    for w, f in word_freqs.most_common(voc_size - 2):  # keeping voc_size-1 for unk
        if len(w) < min_word_len:
            continue
        vocabulary_all[w] = i
        i += 1
    if not vocabulary:
        vocabulary = vocabulary_all
    else:
        logger.info("Words not found in the vocabulary: %d\n" % len(set(vocabulary_all.keys()).difference(
            set(vocabulary.keys()))))

    if labelcol != 'label' and not label_index:
        label_index = {}
        l = 0
        for label in set(writings_df[labelcol]):
            label_index[label] = l
            l += 1
        print("Label index", label_index)

    if by_subset and 'subset' in writings_df.columns:
        training_subjects = list(set(writings_df[writings_df['subset'] == 'train'].subject))
        test_subjects = list(set(writings_df[writings_df['subset'] == 'test'].subject))
    else:
        all_subjects = sorted(list(set(writings_df.subject)))
        training_subjects_size = int(len(all_subjects) * train_prop)
        test_subjects_size = len(all_subjects) - training_subjects_size
        logger.info("%d training subjects, %d test subjects\n" % (training_subjects_size, test_subjects_size))
        # Cross-validation, with fixed slice as input
        test_prop = 1 - train_prop
        test_slice = min(test_slice, nr_slices)
        logger.debug("start index: %f, from %f\n" % (
            len(all_subjects) * (1 / nr_slices) * test_slice, test_prop * test_slice))
        start_slice = int(len(all_subjects) * (1 / nr_slices) * test_slice)
        test_subjects = all_subjects[start_slice: start_slice + test_subjects_size]
        training_subjects = [s for s in all_subjects if s not in test_subjects]
    training_subjects = sorted(training_subjects)  # ensuring reproducibility
    valid_subjects_size = int(len(training_subjects) * valid_prop)
    valid_subjects = training_subjects[:valid_subjects_size]
    training_subjects = training_subjects[valid_subjects_size:]
    categories = [c for c in liwc_categories if c in writings_df.columns]
    logger.debug("%d training users, %d validation users, %d test users." % (
        len(training_subjects),
        len(valid_subjects),
        len(test_subjects)))
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
        else:
            label = label_index[getattr(row, labelcol)]
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

def load_erisk_server_data(dataround_json, tokenizer,
                           logger = None, verbose = 0):
    if verbose:
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
        logger.debug("Loading data...\n")

    subjects_split = {'test': []}
    user_level_texts = {}

    for datapoint in dataround_json:
        words = []
        raw_text = ""
        if "title" in datapoint:
            tokenized_title = tokenizer.tokenize(datapoint["title"])
            words.extend(tokenized_title)
            raw_text += datapoint["title"]
        if "content" in datapoint:
            tokenized_text = tokenizer.tokenize(datapoint["content"])
            words.extend(tokenized_text)
            raw_text += datapoint["content"]

        if datapoint["nick"] not in user_level_texts.keys():
            user_level_texts[datapoint["nick"]] = {}
            user_level_texts[datapoint["nick"]]['texts'] = [words]
            user_level_texts[datapoint["nick"]]['raw'] = [raw_text]
            subjects_split['test'].append(datapoint['nick'])
        else:
            user_level_texts[datapoint["nick"]]['texts'].append(words)
            user_level_texts[datapoint["nick"]]['raw'].append(raw_text)

    return user_level_texts, subjects_split

#save datasets
task = "Depression"
load_data(task).to_csv("/Users/ronhochstenbach/Desktop/Thesis/Data/Processed Data/df_" + task)
