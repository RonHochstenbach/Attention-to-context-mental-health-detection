import numpy as np


train = {}
train["Subjects"] = []
train["Labels"] = []

test = {}
test["Subjects"] = []
test["Labels"] = []

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/eRISK2020_training_data/golden_truth.txt', 'r') as file:
    for l in file:
        if len(l.split()) > 0:
            train["Subjects"].append(l.split()[0][-4:])
            train["Labels"].append(l.split()[1])


with open('//Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/T1/T1_erisk_golden_truth.txt', 'r') as file:
    for l in file:
        if len(l.split())>0:
            test["Subjects"].append(l.split()[0][-4:])
            test["Labels"].append(l.split()[1])

print(f"{len(train['Subjects'])} train subjects")
print(f"{len(test['Subjects'])} test subjects")


train["Labels"] = [int(i) for i in train["Labels"]]
test["Labels"] = [int(i) for i in test["Labels"]]

print(f"There are {len(np.intersect1d(train['Subjects'], test['Subjects']))} duplicates between train and test.")

print(f"Percentage positive in train: {np.sum(train['Labels'])/len(train['Labels'])}")
print(f"Percentage positive in test: {np.sum(test['Labels'])/len(test['Labels'])}")


