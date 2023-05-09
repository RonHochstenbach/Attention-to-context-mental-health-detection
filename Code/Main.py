import pandas as pd

from resource_loader import load_NRC, readDict

root_dir = "/Users/ronhochstenbach/Desktop/Thesis/Data"

#Import Data
task = "Depression"
writings_df = pd.read_csv(root_dir +  "/Processed Data/df_" + task)

#print("Average number of posts per user", writings_df.groupby('subject').count().title.mean())
#print("Average number of comments per user", writings_df.groupby('subject').count().text.mean())

#Import Resources
nrc_lexicon = load_NRC(root_dir + "/Resources/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt")
emotions = list(nrc_lexicon.keys())
#print(emotions)

liwc_dict = {}
for (w, c) in readDict(root_dir + '/resources/liwc.dic'):
    if c not in liwc_dict:
        liwc_dict[c] = []
    liwc_dict[c].append(w)

categories = set(liwc_dict.keys())
print(len(categories))




