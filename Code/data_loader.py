import logging

from resource_loader import load_vocabulary, load_NRC, load_LIWC
from read_erisk_data import read_texts_2019

def load_data(task):
    root_dir = '/Users/ronhochstenbach/Desktop/Thesis/Data'

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

def load_erisk_data(writings_df, hyperparams_features, by_subset=True,
                    pronouns = ["i", "me", "my", "mine", "myself"],
                    train_prop=0.7, valid_prop=0.3, test_slice=2,
                    nr_slices=5,
                    min_post_len=3, min_word_len=1,
                    user_level=True, labelcol='label', label_index=None,
                   logger=None):

    #importing vocabulary, lexicon and liwc
    vocabulary = load_vocabulary(hyperparams_features['vocabulary_path'])
    vocab_size = hyperparams_features(['max_features'])

    emotion_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
    emotions = list(emotion_lexicon.keys())

    liwc_dict = load_LIWC(hyperparams_features['liwc_path'])
    liwc_categories = set(liwc_dict.keys())

    #SEE ERISK.IPYNB FOR DIFFERENT VERSION
    #creating train, validation and test sets
    training_subjects = list(set(writings_df[writings_df['subset']=='train'].subject))
    test_subjects = list(set(writings_df[writings_df['subset'] == 'test'].subject))

    training_subjects = sorted(training_subjects) #ensure reproducability
    valid_subjects_size = int(len(training_subjects)*valid_prop)
    validation_subjects = training_subjects[:valid_subjects_size]
    training_subjects = training_subjects[valid_subjects_size:]

    categories = [c for c in liwc_categories if c in writings_df.columns]

    subjects_split = {'train': training_subjects,
                      'validation': validation_subjects,
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
load_data(task).to_csv("/Users/ronhochstenbach/Desktop/Thesis/Data/df_" + task)

