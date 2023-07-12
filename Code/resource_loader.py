import pickle
import numpy as np
import collections, sys, re


# Opens the vocabulary file and returns it as a dictionary
def load_vocabulary(path):
    vocabulary_list = pickle.load(open(path, 'rb'))
    vocabulary_dict = {}
    for i, w in enumerate(vocabulary_list):
        vocabulary_dict[w] = i
    return vocabulary_dict


# Opens the NRC emotion Lexicon and returns the emotion words as dictionary
def load_NRC(nrc_path):
    word_emotions = {}  # key = word, value = emotions it belongs to
    emotion_words = {}  # key = emotion, value = words with that emotion

    with open(nrc_path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            word, emotion, label = line.split()
            if word not in word_emotions:
                word_emotions[word] = set()
            if emotion not in emotion_words:
                emotion_words[emotion] = set()

            label = int(label)
            if label:  # if label = 1
                word_emotions[word].add(emotion)
                emotion_words[emotion].add(word)
    #print(emotion_words)
    return emotion_words


# Opens the LIWC lexicon and returns dict with keys categories, values words in cat
def load_LIWC(path):
    liwc_dict = {}

    for (w, c) in readDict(path):
        if c not in liwc_dict:
            liwc_dict[c] = []
        liwc_dict[c].append(w)
    #print(liwc_dict)
    return liwc_dict

def readDict(dictionaryPath):
    catList = collections.OrderedDict()
    catLocation = []
    wordList = {}
    finalDict = []

    # Check to make sure the dictionary is properly formatted
    with open(dictionaryPath, "r") as dictionaryFile:
        for idx, item in enumerate(dictionaryFile):
            if "%" in item:
                catLocation.append(idx)
        if len(catLocation) > 2:
            # There are apparently more than two category sections; throw error and die
            sys.exit("Invalid dictionary format. Check the number/locations of the category delimiters (%).")

    # Read dictionary as lines
    with open(dictionaryPath, "r") as dictionaryFile:
        lines = dictionaryFile.readlines()

    # Within the category section of the dictionary file, grab the numbers associated with each category
    for line in lines[catLocation[0] + 1:catLocation[1]]:
        catList[re.split(r'\t+', line)[0]] = [re.split(r'\t+', line.rstrip())[1]]

    # Now move on to the words
    for idx, line in enumerate(lines[catLocation[1] + 1:]):
        # Get each line (row), and split it by tabs (\t)
        workingRow = re.split('\t', line.rstrip())
        wordList[workingRow[0]] = list(workingRow[1:])

    # Merge the category list and the word list
    for key, values in wordList.items():
        for catnum in values:
            workingValue = catList[catnum][0]
            finalDict.append([key, workingValue])
    return finalDict

def load_stopwords(path):
    stopwords_list = []
    with open(path) as f:
        for line in f:
            stopwords_list.append(line.strip())
    return stopwords_list

def load_embeddings(path, embedding_dim, vocabulary_path):
    # random matrix with mean value = 0
    voc = load_vocabulary(vocabulary_path)
    embedding_matrix = np.random.random((len(voc) + 2, embedding_dim)) - 0.5  # voc + unk + pad value(0)
    cnt_inv = 0
    f = open(path, encoding='utf8')
    for i, line in enumerate(f):
        #         print(i)
        values = line.split()
        word = ''.join(values[:-embedding_dim])
        coefs = np.asarray(values[-embedding_dim:], dtype='float32')
        word_i = voc.get(word)
        if word_i is not None:
            embedding_matrix[word_i] = coefs
            cnt_inv += 1

    f.close()

    print('Total %s word vectors.' % len(embedding_matrix))
    print('Words not found in embedding space %d' % (len(embedding_matrix) - cnt_inv))

    return embedding_matrix
