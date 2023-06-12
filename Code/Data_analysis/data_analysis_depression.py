import numpy as np

test_2017 = {}
test_2017["Subjects"] = []
test_2017["Labels"] = []
test_2017["writings"] = []

train_2017 = {}
train_2017["Subjects"] = []
train_2017["Labels"] = []
train_2017["writings"] = []

test_2018 = {}
test_2018["Subjects"] = []
test_2018["Labels"] = []
test_2018["writings"] = []

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/test/writings_all_test_users.txt', 'r')  as file:
    for l in file:
        if len(l.split())>0:
            test_2017["Subjects"].append(l.split()[0][-4:])
            test_2017["writings"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/test/test_golden_truth.txt', 'r')  as file:
    for l in file:
        if len(l.split())>0:
            test_2017["Labels"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/train/risk_golden_truth.txt', 'r') as file:
    for l in file:
        if len(l.split()) > 0:
            train_2017["Subjects"].append(l.split()[0][-4:])
            train_2017["Labels"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/train/scripts evaluation/writings-per-subject-all-train.txt', 'r') as file:
    for l in file:
        if len(l.split()) > 0:
            train_2017["writings"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 1 - depression (test split, train split is 2017 data)/risk-golden-truth-test.txt', 'r') as file:
    for l in file:
        if len(l.split())>0:
            test_2018["Subjects"].append(l.split()[0][-4:])
            test_2018["Labels"].append(l.split()[1])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 1 - depression (test split, train split is 2017 data)/writings-per-subject-all-test.txt', 'r') as file:
    for l in file:
        if len(l.split())>0:
            test_2018["writings"].append(l.split()[1])

print(f"{len(test_2017['Subjects'])} test subjects 2017")
print(f"{len(train_2017['Subjects'])} train subjects 2017")
print(f"{len(test_2018['Subjects'])} test subjects 2018")

test_2017["Labels"] = [int(i) for i in test_2017["Labels"]]
train_2017["Labels"] = [int(i) for i in train_2017["Labels"]]
test_2018["Labels"] = [int(i) for i in test_2018["Labels"]]

test_2017["writings"] = [int(i) for i in test_2017["writings"]]
train_2017["writings"] = [int(i) for i in train_2017["writings"]]
test_2018["writings"] = [int(i) for i in test_2018["writings"]]

print(f"There are {len(np.intersect1d(test_2017['Subjects'], train_2017['Subjects']))} duplicates between test17 and train17.")
print(f"There are {len(np.intersect1d(test_2017['Subjects'], test_2018['Subjects']))} duplicates between test17 and test18.")
print(f"There are {len(np.intersect1d(train_2017['Subjects'], test_2018['Subjects']))} duplicates between train17 and test18.")

print(f"Percentage positive in test 2017: {np.sum(test_2017['Labels'])/len(test_2017['Labels'])}")
print(f"Percentage positive in train 2017: {np.sum(train_2017['Labels'])/len(train_2017['Labels'])}")
print(f"Percentage positive in test 2018: {np.sum(test_2018['Labels'])/len(test_2018['Labels'])}")

print(f"Average number of posts in test 2017: {np.mean(test_2017['writings'])}")
print(f"Average number of posts in train 2017: {np.mean(train_2017['writings'])}")
print(f"Average number of posts in test 2018: {np.mean(test_2018['writings'])}")


