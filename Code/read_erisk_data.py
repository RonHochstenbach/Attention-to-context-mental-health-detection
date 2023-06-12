import xml.etree.ElementTree as ET
import pandas as pd
import os

#opens an xml subject file, and converts it into a list, where each entry is a dictionary.
#Each dictionary has as keys Subject, Title, Text and Date. SOMETIMES TITLE AND TEXT MIXED UP?
def read_subject_writings(subject_file):
    writings = []
    with open(subject_file) as file:
        contents = file.read()
        root = ET.fromstring(contents)
        try:
            subject = root.findall('ID')[0].text.strip()
        except Exception:
            print('Cannot extract ID', contents[:500], '\n-------\n')
        for w in root.iter('WRITING'):
            subject_writings = {'subject': subject}
            for title in w.findall('TITLE'):
                subject_writings['title'] = title.text
            for text in w.findall('TEXT'):
                subject_writings['text'] = text.text
            for date in w.findall('DATE'):
                subject_writings['date'] = date.text
            writings.append(subject_writings)

    return writings


def read_texts_2019(datadir_root_T1_2019,
                    datadirs_T1_2019,
                    labels_files_T1_2019,
                    test_suffix='0000',
                    chunked_subsets='train'):
    writings = {'train': [], 'test': []}
    writings_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    for subset in ('train', 'test'):
        for subdir in [os.path.join(datadir_root_T1_2019[subset], subp) for subp in datadirs_T1_2019[subset]]:
            if subset in chunked_subsets:
                chunkdirs = [os.path.join(datadir_root_T1_2019[subset], subdir, chunkdir)
                             for chunkdir in os.listdir(subdir)]
            else:
                chunkdirs = [os.path.join(datadir_root_T1_2019[subset], subdir)]

            for chunkdir in chunkdirs:
                #print(chunkdir)
                if not os.path.isdir(chunkdir):
                    #print(chunkdir + " is not a directory")
                    continue
                for subject_file in os.listdir(chunkdir):

                    if not subject_file == '.DS_Store':
                        writings[subset].extend(read_subject_writings(os.path.join(chunkdir, subject_file)))
        writings_df_part = pd.DataFrame(writings[subset])
        # add a suffix for users in the test -- the numbers are duplicated with the ones in train
        #print(writings_df_part)
        #print(writings_df_part.keys())
        if subset == 'test':
            writings_df_part['subject'] = writings_df_part['subject'].apply(lambda s: s + test_suffix)
            #print(subset, writings_df_part.subject)
        writings_df_part['subset'] = subset
        writings_df = pd.concat([writings_df, writings_df_part])
        writings_df.reindex()

        for label_file in labels_files_T1_2019[subset]:
            labels = pd.read_csv(os.path.join(datadir_root_T1_2019[subset], label_file),
                                 delimiter='\s+', names=['subject', 'label'])
            # add a suffix for users in the test -- the numbers are duplicated with the ones in train
            if subset == 'test':
                labels['subject'] = labels['subject'].apply(lambda s: s + test_suffix)
            labels_df = pd.concat([labels_df, labels])
    labels_df = labels_df.drop_duplicates()
    labels_df = labels_df.set_index('subject')

    writings_df = writings_df.drop_duplicates()

    writings_df = writings_df.join(labels_df, on='subject')

    return writings_df


def read_texts_2020(datadir_root_T1_2020,
                    datadirs_T1_2020,
                    labels_files_T1_2020,
                    test_suffix='0000',
                    chunked_subsets=None):
    writings = {'train': [], 'test': []}
    writings_df = pd.DataFrame()
    labels_df = pd.DataFrame()
    for subset in ('train', 'test'):
        for subdir in [os.path.join(datadir_root_T1_2020[subset], subp) for subp in datadirs_T1_2020[subset]]:

            for subject_file in os.listdir(subdir):
                writings[subset].extend(read_subject_writings(os.path.join(subdir, subject_file)))
        writings_df_part = pd.DataFrame(writings[subset])
        # add a suffix for users in the test -- the numbers are duplicated with the ones in train
        if subset == 'test':
            writings_df_part['subject'] = writings_df_part['subject'].apply(lambda s: s + test_suffix)
            #print(subset, writings_df_part.subject)
        writings_df_part['subset'] = subset
        writings_df = pd.concat([writings_df, writings_df_part])
        writings_df.reindex()

        for label_file in labels_files_T1_2020[subset]:
            labels = pd.read_csv(os.path.join(datadir_root_T1_2020[subset], label_file),
                                 delimiter='\s+', names=['subject', 'label'])
            # add a suffix for users in the test -- the numbers are duplicated with the ones in train
            if subset == 'test':
                labels['subject'] = labels['subject'].apply(lambda s: s + test_suffix)
            labels_df = pd.concat([labels_df, labels])
    labels_df = labels_df.drop_duplicates()
    labels_df = labels_df.set_index('subject')

    writings_df = writings_df.drop_duplicates()

    writings_df = writings_df.join(labels_df, on='subject')

    return writings_df
