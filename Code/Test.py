import numpy as np



test_2017 = []
train_2017 = []
test_2018 = []

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/test/writings_all_test_users.txt', 'r')  as file:
    for l in file:
        if len(l.split())>0:

            test_2017.append(l.split()[0][-4:])

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/train/risk_golden_truth.txt', 'r') as file:
    for l in file:
        if len(l.split()) > 0:
            train_2017.append(l.split()[0][-4:])

print(f"{len(train_2017)} train subjects 2017")
print(f"{len(test_2017)} test subjects 2017")

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 1 - depression (test split, train split is 2017 data)/risk-golden-truth-test.txt', 'r') as file:
    for l in file:
        if len(l.split())>0:
            test_2018.append(l.split()[0][-4:])

print(f"{len(test_2018)} test subjects 2018")

test2017 = np.array(test_2017)
train2017 = np.array(train_2017)
test2018 =np.array(test_2018)

print(f"There are {len(np.intersect1d(test2017, train2017))} duplicates between test17 and train17.")
print(np.intersect1d(test2017, train2017))
print(f"There are {len(np.intersect1d(test2017, test2018))} duplicates between test17 and test18.")
print(np.intersect1d(test2017, test2018))
print(f"There are {len(np.intersect1d(train2017, test2018))} duplicates between train17 and test18.")
print(np.intersect1d(train2017, test2018))