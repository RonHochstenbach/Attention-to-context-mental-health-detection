from read_erisk_data import read_subject_writings
from read_erisk_data import read_texts_2019
import os
import pandas as pd

root_dir = '/Users/ronhochstenbach/Desktop/Thesis/Data'

datadirs_T1_2018 = {
    'train': ['train/positive_examples_anonymous_chunks/', 'train/positive_examples_anonymous_chunks/', 'test/'],
    'test': ['task 1 - depression (test split, train split is 2017 data)/']
}
datadir_root_T1_2018 = {
    'train': root_dir + '/eRisk/data/2017/',
    'test': root_dir + '/eRisk/data/2018/'
}

labels_files_T1_2018 = {
    'train': ['train/risk_golden_truth.txt', 'test/test_golden_truth.txt'],
    'test': ['task 1 - depression (test split, train split is 2017 data)/risk-golden-truth-test.txt']
}

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




