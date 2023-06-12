import numpy as np

test_2018 = {}
test_2018["Subjects"] = []
test_2018["Labels"] = []
test_2018["writings"] = []

train_2018 = {}
train_2018["Subjects"] = []
train_2018["Labels"] = []
train_2018["writings"] = []

test_2019 = {}
test_2019["Subjects"] = []
test_2019["Labels"] = []
test_2019["writings"] = []

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 2 - anorexia/test/writings-per-subject-all-test.txt', 'r')  as file:
    for l in file:
        if len(l.split())>0:
            test_2018["Subjects"].append(l.split()[0][-4:])
            test_2018["writings"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 2 - anorexia/test/risk-golden-truth-test.txt', 'r')  as file:
    for l in file:
        if len(l.split())>0:
            test_2018["Labels"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 2 - anorexia/train/risk_golden_truth.txt', 'r') as file:
    for l in file:
        if len(l.split()) > 0:
            train_2018["Subjects"].append(l.split()[0][-4:])
            train_2018["Labels"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 2 - anorexia/train/scripts evaluation/writings-per-subject-all-train.txt', 'r') as file:
    for l in file:
        if len(l.split()) > 0:
            train_2018["writings"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/eRisk2019_T1/test data - T1/T1_erisk_golden_truth.txt', 'r') as file:
    for l in file:
        if len(l.split())>0:
            test_2019["Subjects"].append(l.split()[0][-4:])
            test_2019["Labels"].append(l.split()[1])


print(f"{len(test_2018['Subjects'])} test subjects 2018")
print(f"{len(train_2018['Subjects'])} train subjects 2018")
print(f"{len(test_2019['Subjects'])} test subjects 2019")

test_2018["Labels"] = [int(i) for i in test_2018["Labels"]]
train_2018["Labels"] = [int(i) for i in train_2018["Labels"]]
test_2019["Labels"] = [int(i) for i in test_2019["Labels"]]

test_2018["writings"] = [int(i) for i in test_2018["writings"]]
train_2018["writings"] = [int(i) for i in train_2018["writings"]]
test_2019["writings"] = [int(i) for i in test_2019["writings"]]


print(f"There are {len(np.intersect1d(test_2018['Subjects'], train_2018['Subjects']))} duplicates between test18 and train18.")
print(f"There are {len(np.intersect1d(test_2018['Subjects'], test_2019['Subjects']))} duplicates between test18 and test19.")
print(f"There are {len(np.intersect1d(train_2018['Subjects'], test_2019['Subjects']))} duplicates between train18 and test19.")

print(f"Percentage positive in test 2018: {np.sum(test_2018['Labels'])/len(test_2018['Labels'])}")
print(f"Percentage positive in train 2018: {np.sum(train_2018['Labels'])/len(train_2018['Labels'])}")
print(f"Percentage positive in test 2019: {np.sum(test_2019['Labels'])/len(test_2019['Labels'])}")

print(f"Average number of posts in test 2018: {np.mean(test_2018['writings'])}")
print(f"Average number of posts in train 2018: {np.mean(train_2018['writings'])}")


