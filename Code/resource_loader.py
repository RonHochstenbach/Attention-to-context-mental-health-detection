import pickle
import numpy as np


#Opens the vocabulary file and returns it as a dictionary
def load_vocabulary(path):
    vocabulary_list = pickle.load(open(path, 'rb'))
    vocabulary_dict = {}
    for i,w in enumerate(vocabulary_list):
        vocabulary_dict[w] = i
    return vocabulary_dict

#Opens the NRC emotion Lexicon and returns the emotion words as dictionary
def load_NRC(nrc_path):
    word_emotions = {}      #key = word, value = emotions it belongs to
    emotion_words = {}      #key = emotion, value = words with that emotion

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
            if label:                       #if label = 1
                word_emotions[word].add(emotion)
                emotion_words[emotion].add(word)

    return emotion_words

#Opens the LIWC lexicon and returns
def load_LIWC(path):
    liwc_dict = {}

    for 

