This repository contains the code written for the Master Thesis "Paying Attention to Context When Detecting Mental Health Disorders Through Natural Language Processing" by Ron Hochstenbach.


It is based upon (but heavily modifies) the code writtenfor the work "An emotion and cognitive based analysis of mental health disorders from social media data" by Ana-Sabina Uban, Berta Chulvi and Paolo Rosso, which can be found at
https://github.com/ananana/mental-disorders.

The most important scripts are Main train.py, Main hyperopt.py, Main test.py and Main test - -Automated.py. The former two can be used to run the training procedure and hyperparameter optimisation procedure, respectively. The latter two are used to compute metrics from a trained model on a test set, where the first one does this for one specified model and data set, and the second one can iterate over several models and data sets per run. Finally, Data analysis.ipynb can be used to perform data- and results analysis, and obtain the figures displayed throughout this work. The file hyperparamaters.py can be used to set the desired hyperparameters. Some external resources are necessary for running these scripts.

stopwords.txt can be obtained from https://github.com/ananana/mental-disorders/blob/clean/stopwords.txt
The NRC emotion lexicon can be obtained from
        https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
The LIWC lexicon can be obtained from
        https://www.liwc.app/
Pre-trained GloVe embeddings can be obtained from
        https://www.kaggle.com/datasets/takuok/glove840b300dtxt
The data sets used can be obtained by filling in and submitting the respective user agreements found at
    https://erisk.irlab.org/
