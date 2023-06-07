

train_train=0
train_test = 0

test_2018 = 0

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/test/writings_all_test_users.txt', 'r')  as file:
    for l in file:
        if len(l)>0:
            train_test +=1

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2017/train/risk_golden_truth.txt', 'r') as file:
    for l in file:
        if len(l) > 0:
            train_train +=1

print(f"{train_train} train subjects 2017")
print(f"{train_test} test subjects 2017")

with open('/Users/ronhochstenbach/Desktop/Thesis/Data/Raw Data/2018/task 1 - depression (test split, train split is 2017 data)/risk-golden-truth-test.txt', 'r') as file:
    for l in file:
        if len(l)>0:
            test_2018+=1

print(f"{test_2018} test subjects 2018")